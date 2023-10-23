from typing import Union, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import numpy as np

import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from torch.utils.tensorboard import SummaryWriter

from .aggregate import aggregate
from .TensorboardStrategy import TensorboardStrategy
from utils.clustering_fn import make_clusters, print_clusters, compute_clustering_acc

import copy

class IFCA(TensorboardStrategy):
    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        fraction_evaluate,
        writer,
        on_fit_config_fn,
        n_clusters,
        model_init,
        total_num_clients,
        transforms,
        n_base_layers,
        ):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        on_fit_config_fn=on_fit_config_fn,
                        writer=writer,
                        total_num_clients=total_num_clients,
                        transforms=transforms)
        
        self.writer = writer
        self.n_clusters = n_clusters
        self.n_base_layers = n_base_layers

        self.parameters = []
        for i, m_init in enumerate(model_init):
            for j, (_,val) in enumerate(m_init.items()):
                # add entire model weights
                if i == 0:
                    self.parameters.append(val.cpu().numpy())
                # add only personalized layers
                elif j>n_base_layers-1:
                    self.parameters.append(val.cpu().numpy())

        self.n_pers_layers = (len(self.parameters) - self.n_base_layers) // self.n_clusters
        self.parameters = ndarrays_to_parameters(self.parameters)
        
        self.kmeans = None

    def __repr__(self) -> str:
        return "IFCA"


    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        if server_round == 1:
            self.init_scalars()
        
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        self.clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        partitions_conf = self.set_client_partitions(total_num_clients=self.total_num_clients, sample_size=sample_size, server_round=server_round)

        fit_configurations = []
        for i, (client, partition_conf) in enumerate(zip(self.clients, partitions_conf)):
            config["client_number"] = i
            config["n_clusters"] = self.n_clusters
            config["round"] = server_round
            config["n_base_layers"] = self.n_base_layers
            config["n_pers_layers"] = self.n_pers_layers
            partition_conf.update(config)
            fit_configurations.append((client, FitIns(self.parameters, copy.deepcopy(partition_conf))))

        # Return client/config pairs
        return fit_configurations


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Extract local model weights from results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        client_numbers = [
            fit_res.metrics["client_number"] for _, fit_res in results
        ]
        cluster_labels = [
            fit_res.metrics["cluster_id"] for _, fit_res in results
        ]
        cluster_truth = [
            fit_res.metrics["transform"] for _, fit_res in results
        ]

        ordered_indices = np.array(client_numbers).argsort()
        weights_results = np.array(weights_results, dtype=object)[ordered_indices]
        self.cluster_labels = np.array(cluster_labels, dtype=object)[ordered_indices]
        self.cluster_truth = np.array(cluster_truth, dtype=object)[ordered_indices]

        print_clusters(self.cluster_labels, self.cluster_truth, n_clusters=self.n_clusters)

        base_layers = [(param[:self.n_base_layers], n_examples) for param, n_examples in weights_results]
        base_layers_agg = aggregate(base_layers)

        # Group local model update per clusters
        personalized_layers = [[] for i in range(self.n_clusters)]
        for client_number, label in enumerate(self.cluster_labels):
            personalized_layers[label].append((weights_results[client_number][0][self.n_base_layers:], 
                                            weights_results[client_number][1]))

        # Compute personalized update for each cluster
        personalized_layers_aggs = []
        for i, weights in enumerate(personalized_layers):
            if len(weights) == 0:
                personalized_layers_agg = parameters_to_ndarrays(self.parameters)[self.n_base_layers:][self.n_pers_layers*i : self.n_pers_layers*(i+1)]
            else:
                personalized_layers_agg = aggregate(weights)
            # cluster_weights = base_layers_agg + personalized_layers_agg
            personalized_layers_aggs.append(personalized_layers_agg)
        
        self.parameters = []
        # Add base layers
        for val in base_layers_agg:
            self.parameters.append(val)
        for params in personalized_layers_aggs:
            for val in params:
                self.parameters.append(val)
        
        self.parameters = ndarrays_to_parameters(self.parameters)
            
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        
        return self.parameters, metrics_aggregated


    def set_model_parameters(self, model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Assigning model parameters wrt client clusters
        eval_configurations = []
        params = parameters_to_ndarrays(self.parameters)
        for i, client in enumerate(self.clients):
            cluster_id = self.cluster_labels[i]
            config = {"cluster_id": cluster_id}
            cluster_params = params[:self.n_base_layers] + params[self.n_base_layers:][self.n_pers_layers*cluster_id : self.n_pers_layers*(cluster_id+1)]
            eval_configurations.append((client, EvaluateIns(ndarrays_to_parameters(cluster_params), config)))

        # Return client/config pairs
        return eval_configurations


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round=server_round, 
            results=results, 
            failures=failures
        )

        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        accs = [metrics["accuracy"] for _, metrics in eval_metrics]
        clustering_acc = compute_clustering_acc(self.cluster_labels, self.cluster_truth, n_clusters=self.n_clusters)

        accs_per_cluster = [[] for i in range(self.n_clusters)]
        for _, metrics in eval_metrics:
            accs_per_cluster[metrics['cluster_id']].append(metrics['accuracy'])

        for i, accs in enumerate(accs_per_cluster):
            self.writer.add_scalar(f"Cluster/federated_accuracy_C{i}", np.average(accs), server_round)

        self.writer.add_scalar(f'Cluster/adjusted_rand_score', clustering_acc, server_round)

        return loss_aggregated, metrics_aggregated
