from typing import Union, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import numpy as np

import flwr as fl
import psutil
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
from utils.clustering_fn import make_clusters, print_clusters

import copy

class TestEncoding(TensorboardStrategy):
    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        fraction_evaluate,
        eval_fn,
        writer,
        on_fit_config_fn,
        n_clusters,
        model):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        writer=writer)
        
        self.writer = writer
        self.n_clusters = n_clusters
        #self.models = [copy.deepcopy(model) for _ in range(self.n_clusters)]

        param = ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
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

            fit_configurations = []
            for i, client in enumerate(self.clients):
                config["client_number"] = i
                fit_configurations.append((client, FitIns(parameters, copy.deepcopy(config))))
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
        weights_results = np.array(weights_results)[ordered_indices]
        print(client_numbers)
        print(np.array(cluster_truth)[ordered_indices])

        # Group local model weights per clusters
        weights_per_cluster = [[] for i in range(self.n_clusters)]
        for client_number, label in enumerate(self.cluster_labels):
            weights_per_cluster[label].append(weights_results[client_number])

        # Compute global update for each cluster
        parameters_aggs = [
            ndarrays_to_parameters(aggregate(weights_per_cluster[i])) for i in range(self.n_clusters)
        ]
        
        parameters_aggregated = aggregate(weights_results)
        
        # for model, param in zip(self.models, parameters_aggs):
        #     self.set_model_parameters(model, param)
        self.parameters = parameters_aggs
            
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        
        return parameters_aggs[0], metrics_aggregated


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
        cluster_truth = np.array(cluster_truth)[ordered_indices]
        low_dims = np.array(low_dims)[ordered_indices]

        # Building clusters
        if self.kmeans is None:
            self.cluster_labels, cluster_centers, self.kmeans = make_clusters(low_dims, n_clusters=self.n_clusters)
        else:
            self.cluster_labels, cluster_centers, self.kmeans = make_clusters(low_dims, n_clusters=self.n_clusters, kmeans=self.kmeans)

        print_clusters(self.cluster_labels, cluster_truth, n_clusters=self.n_clusters)



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