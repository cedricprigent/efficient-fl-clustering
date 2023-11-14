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
from sklearn.manifold import TSNE

import copy
import time

class ClusterEmbeddings(TensorboardStrategy):
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
        self.transforms = transforms

        param = ndarrays_to_parameters([val.cpu().numpy() for _, val in model_init.items()])
        self.parameters = [param for i in range(self.n_clusters)]

    def __repr__(self) -> str:
        return "ClusterEmbeddings"


    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        self.start_training_round = time.time()

        if server_round == 0:
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

        if len(parameters.tensors) == 0:
            self.start_clustering = time.time()
            # Request low dimensional representation of client data
            config["task"] = "compute_low_dim"
            print(config)

            partitions_conf = self.assign_all_partitions_for_clustering(total_num_clients=self.total_num_clients, sample_size=sample_size)

            fit_configurations = []
            for i, (client, partition_conf) in enumerate(zip(self.clients, partitions_conf)):
                config["client_number"] = i
                partition_conf.update(config)
                fit_configurations.append((client, FitIns(parameters, copy.deepcopy(partition_conf))))
        else:
            # Request local training
            config["task"] = "local_training"

            partitions_conf = self.set_client_partitions(total_num_clients=self.total_num_clients, sample_size=sample_size, server_round=server_round)

            print("Selecting partitions to use for the current training round")
            for i in range(self.n_clusters):
                print(f"CLUSTER {i}:")
                for conf in partitions_conf:
                    partition = conf["partition"]
                    transform = conf["transform"]
                    if self.cluster_labels[partition] == i:
                        print(f"Partition {partition} - {transform}")
                print(f"####")

            # Assigning model parameters wrt client clusters
            fit_configurations = []
            for i, (client, partition_conf) in enumerate(zip(self.clients, partitions_conf)):
                partition_conf["client_number"] = i
                partition_conf.update(config)
                partition = int(partition_conf["partition"])
                fit_configurations.append((client, FitIns(self.parameters[self.cluster_labels[partition]], copy.deepcopy(partition_conf))))

            self.last_partitions_conf = partitions_conf

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

        partition_numbers = np.array([
            fit_res.metrics["partition"] for _, fit_res in results
        ]).astype(int)
        cluster_truth = [
            fit_res.metrics["transform"] for _, fit_res in results
        ]

        # Compute global update of the base layers
        base_layers = [(param[:self.n_base_layers], n_examples) for param, n_examples in weights_results]
        base_layers_agg = aggregate(base_layers)

        # Group local model update per clusters
        personalized_layers = [[] for i in range(self.n_clusters)]
        for partition_number, truth, weights_result in zip(partition_numbers, cluster_truth, weights_results):
            label = self.cluster_labels[partition_number]
            print(f"Add partition {partition_number} - {truth} to cluster {label}")
            personalized_layers[label].append((weights_result[0][self.n_base_layers:], 
                                            weights_result[1]))

        # Compute global update for each cluster
        for i, weights in enumerate(personalized_layers):
            if len(weights) == 0:
                continue
            personalized_layers_agg = aggregate(weights)
            cluster_weights = base_layers_agg + personalized_layers_agg
            self.parameters[i] = ndarrays_to_parameters(cluster_weights)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        self.end_training_round = time.time()
        
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

        ld_shape = low_dims[0][0][0].shape[0]
        # low_dims = np.array(low_dims).reshape(-1, ld_shape)
        low_dims = np.concatenate(low_dims, axis=1)[0]
        print("low_dims shape: ", low_dims.shape)
        print(low_dims[0])
        # torch.save(low_dims, '/home/cprigent/Documents/notebook/ML/PFL/cifar_low_dims.pt')
        
        cluster_truth = np.concatenate([
            transforms.split(',') for transforms in [fit_res.metrics["transform"] for _, fit_res in results]
        ])
        partitions_numbers = np.concatenate([
            partitions.split(',') for partitions in [fit_res.metrics["partition"] for _, fit_res in results]
        ]).astype(int)

        ordered_indices = np.array(partitions_numbers).argsort()
        self.cluster_truth = np.array(cluster_truth)[ordered_indices]
        low_dims = np.array(low_dims, dtype=object)[ordered_indices]
        print(low_dims.shape)

        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(low_dims)
        # torch.save(tsne, '/home/cprigent/Documents/notebook/ML/PFL/tsne.pt')

        # Building clusters
        self.cluster_labels, self.cluster_centers, _ = make_clusters(low_dims, n_clusters=self.n_clusters, n_clients=len(low_dims), kmeans_type='kmeans')
        # torch.save(self.cluster_labels, '/home/cprigent/Documents/notebook/ML/PFL/labels.pt')

        print_clusters(self.cluster_labels, self.cluster_truth, n_clusters=self.n_clusters)

        self.end_clustering = time.time()


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
        for i, (client, partition_conf) in enumerate(zip(self.clients, self.last_partitions_conf)):
            label = int(self.cluster_labels[partition_conf["partition"]])
            config = {"cluster_id": label}
            eval_configurations.append((client, EvaluateIns(self.parameters[label], config)))

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
        self.writer.add_scalar(f"System/clustering_time", self.end_clustering - self.start_clustering, server_round)
        self.writer.add_scalar(f'System/training_round_time', self.end_training_round - self.start_training_round, server_round)

        return loss_aggregated, metrics_aggregated


    def assign_all_partitions_for_clustering(self, total_num_clients, sample_size):
        partitions_conf = []
        partitions = [i for i in range(total_num_clients)]
        n_partitions_per_client = total_num_clients // sample_size
        additional_partitions = total_num_clients % sample_size
        for _ in range(sample_size):
            config = {}
            n_part = n_partitions_per_client
            if additional_partitions > 0:
                n_part += 1
                additional_partitions -= 1
            client_partitions = partitions[:n_part]
            client_transforms = [self.transform_assignments[partition] for partition in client_partitions]
            config["partition"] = self.convert_to_string(client_partitions)
            config["transform"] = self.convert_to_string(client_transforms)
            del partitions[:n_part]
            partitions_conf.append(copy.deepcopy(config))
        print(partitions_conf)

        return partitions_conf

    def convert_to_string(self, array):
        new_string = ""
        for element in array:
            new_string += str(element)
            new_string += ','
        new_string = new_string[:-1]
        return new_string
        