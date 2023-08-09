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
        model_init):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        on_fit_config_fn=on_fit_config_fn,
                        writer=writer)
        
        self.writer = writer
        self.n_clusters = n_clusters

        self.parameters = []
        for m_init in model_init:
            for _, val in m_init.items():
                self.parameters.append(val.cpu().numpy())
        self.param_size = len(self.parameters) // n_clusters
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

        fit_configurations = []
        for i, client in enumerate(self.clients):
            config["client_number"] = i
            config["n_clusters"] = self.n_clusters
            fit_configurations.append((client, FitIns(self.parameters, copy.deepcopy(config))))

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

        # Group local model weights per clusters
        weights_per_cluster = [[] for i in range(self.n_clusters)]
        for i, cluster_label in enumerate(self.cluster_labels):
            weights_per_cluster[cluster_label].append(weights_results[i])

        # Compute global update for each cluster
        parameters_aggs = []
        for i in range(self.n_clusters):
            # # If no results for a given model, keep the previous model weights
            if len(weights_per_cluster[i]) == 0:
                parameters_aggs.append(parameters_to_ndarrays(self.parameters)[self.param_size*i : self.param_size*(i+1)])
            else:
                parameters_aggs.append(aggregate(weights_per_cluster[i]))
        
        self.parameters = []
        for params in parameters_aggs:
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
            cluster_params = params[self.param_size*cluster_id : self.param_size*(cluster_id+1)]
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
