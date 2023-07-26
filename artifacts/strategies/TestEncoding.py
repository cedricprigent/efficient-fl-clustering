from typing import Union, Dict, List, Optional, Tuple
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
        n_clusters):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        eval_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        writer=writer)
        
        self.writer = writer
        self.n_clusters = n_clusters
        # self.cluster_centers = None

    def __repr__(self) -> str:
        return "TestEncoding"


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
            (parameters_to_ndarrays(fit_res.parameters)[:-1], fit_res.num_examples)
            for _, fit_res in results
        ]

        # Extract local data compression from results
        low_dims = [
            parameters_to_ndarrays(fit_res.parameters)[-1:] for _, fit_res in results
        ]
        low_dims = [np.array(sample).flatten() for sample in low_dims]
        print("low_dims size: ", np.array(low_dims).shape)
        print(low_dims)

        cluster_truth = [
            fit_res.metrics for _, fit_res in results
        ]

        # Building clusters
        cluster_labels, cluster_centers = make_clusters(low_dims, n_clusters=self.n_clusters)
        # if self.cluster_centers is None:
        #     cluster_labels, self.cluster_centers = make_clusters(low_dims, n_clusters=self.n_clusters)
        # else:
        #     cluster_labels, self.cluster_centers = make_clusters(low_dims, n_clusters=self.n_clusters, cluster_centers=self.cluster_centers)

        print_clusters(cluster_labels, cluster_truth, n_clusters=self.n_clusters)

        parameters_aggregated = ndarrays_to_parameters(
            aggregate(weights_results)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        
        return parameters_aggregated, metrics_aggregated