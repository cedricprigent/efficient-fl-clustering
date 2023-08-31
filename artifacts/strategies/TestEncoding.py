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

class TestEncoding(TensorboardStrategy):
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
        self.min_fit_clients = min_fit_clients
        self.n_base_layers = n_base_layers

        param = ndarrays_to_parameters([val.cpu().numpy() for _, val in model_init.items()])
        self.parameters = [param for i in range(self.n_clusters)]
        self.kmeans = None

    def __repr__(self) -> str:
        return "TestEncoding"


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

        if len(parameters.tensors) == 0:
            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            self.clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # Request low dimensional representation of client data
            config["task"] = "compute_low_dim"
            print(config)

            partitions_conf = self.set_client_partitions(total_num_clients=self.total_num_clients, sample_size=sample_size, server_round=server_round)

            fit_configurations = []
            for i, (client, partition_conf) in enumerate(zip(self.clients, partitions_conf)):
                config["client_number"] = i
                partition_conf.update(config)
                fit_configurations.append((client, FitIns(parameters, copy.deepcopy(partition_conf))))
        else:
            # Request local training
            config["task"] = "local_training"

            for i, label in enumerate(self.cluster_labels):
                print(f"Assigning model: {label} to client {i}")
            print(config)

            # Assigning model parameters wrt client clusters
            fit_configurations = []
            for i, client in enumerate(self.clients):
                config["client_number"] = i
                fit_configurations.append((client, FitIns(self.parameters[self.cluster_labels[i]], copy.deepcopy(config))))


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
        cluster_truth = [
            fit_res.metrics["transform"] for _, fit_res in results
        ]

        ordered_indices = np.array(client_numbers).argsort()
        weights_results = np.array(weights_results, dtype=object)[ordered_indices]

        # Compute global update of the base layers
        base_layers = [(param[:self.n_base_layers], n_examples) for param, n_examples in weights_results]
        base_layers_agg = aggregate(base_layers)

        # Group local model update per clusters
        personalized_layers = [[] for i in range(self.n_clusters)]
        for client_number, label in enumerate(self.cluster_labels):
            personalized_layers[label].append((weights_results[client_number][0][self.n_base_layers:], 
                                            weights_results[client_number][1]))

        # Compute global update for each cluster
        for i, weights in enumerate(personalized_layers):
            if len(weights) == 0:
                continue
            personalized_layers_agg = aggregate(weights)
            cluster_weights = base_layers_agg + personalized_layers_agg
            self.parameters[i] = ndarrays_to_parameters(cluster_weights)

        # # Group local model weights per clusters
        # weights_per_cluster = [[] for i in range(self.n_clusters)]
        # for client_number, label in enumerate(self.cluster_labels):
        #     weights_per_cluster[label].append(weights_results[client_number])

        # # Compute global update for each cluster
        # for i, weights in enumerate(weights_per_cluster):
        #     if len(weights) == 0:
        #         continue
        #     self.parameters[i] = ndarrays_to_parameters(aggregate(weights))
            
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        
        return self.parameters[0], metrics_aggregated


    def build_clusters(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Extract local data compression from results
        low_dims = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        low_dims = [np.array(sample).flatten() for sample in low_dims]

        cluster_truth = [
            fit_res.metrics["transform"] for _, fit_res in results
        ]
        client_numbers = [
            fit_res.metrics["client_number"] for _, fit_res in results
        ]

        ordered_indices = np.array(client_numbers).argsort()
        self.cluster_truth = np.array(cluster_truth)[ordered_indices]
        low_dims = np.array(low_dims, dtype=object)[ordered_indices]
        print(low_dims.shape)

        # Building clusters
        if self.kmeans is None:
            self.cluster_labels, self.cluster_centers, self.kmeans = make_clusters(low_dims, n_clusters=self.n_clusters, n_clients=self.min_fit_clients)
        else:
            self.cluster_labels, self.cluster_centers, self.kmeans = make_clusters(low_dims, n_clusters=self.n_clusters, kmeans=self.kmeans, n_clients=self.min_fit_clients)

        print_clusters(self.cluster_labels, self.cluster_truth, n_clusters=self.n_clusters)



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
        for i, client in enumerate(self.clients):
            config = {"cluster_id": int(self.cluster_labels[i])}
            eval_configurations.append((client, EvaluateIns(self.parameters[self.cluster_labels[i]], config)))

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