from typing import Union, Dict, List, Optional, Tuple

import flwr as fl
import psutil
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import copy
import time
import torch

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
    parameters_to_ndarrays,
)


class TensorboardStrategy(fl.server.strategy.FedAvg):
    def __repr__(self) -> str:
        return "TensorboardStrategy"

    def __init__(
        self,
        min_fit_clients,
        min_available_clients,
        fraction_fit,
        fraction_evaluate,
        writer,
        on_fit_config_fn,
        total_num_clients,
        transforms,
        ):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        on_fit_config_fn=on_fit_config_fn)
        
        self.writer = writer
        self.bytes_recv_init_counter = psutil.net_io_counters().bytes_recv
        self.bytes_sent_init_counter = psutil.net_io_counters().bytes_sent
        self.total_num_clients = total_num_clients
        self.transform_assignments = self.build_transform_assignments(total_num_clients, transforms)
        self.encountered_clients = [False for _ in range(total_num_clients)]
        self.rng = random.Random(0)

    def configure_fit(
        self, server_round, parameters, client_manager
    ):
        """Configure the next round of training."""

        self.start_training_round = time.time()
        
        if server_round == 1:
            self.init_scalars()

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        self.clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        partitions_conf = self.set_client_partitions(total_num_clients=self.total_num_clients, sample_size=sample_size, server_round=server_round)

        fit_configurations = []
        for client, partition_conf in zip(self.clients, partitions_conf):
            partition_conf.update(config)
            fit_configurations.append((client, FitIns(parameters, copy.deepcopy(partition_conf))))


        # Return client/config pairs
        return fit_configurations


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results."""

        params, metrics = super().aggregate_fit(server_round, results, failures)
        self.end_training_round = time.time()

        return params, metrics


    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using an evaluation function."""
        
        # Write scalars
        self.writer.add_scalar("System/bytes_rcv", (psutil.net_io_counters().bytes_recv - self.bytes_recv_init_counter) / 1000000, server_round)
        self.writer.add_scalar("System/bytes_sent", (psutil.net_io_counters().bytes_sent - self.bytes_sent_init_counter) / 1000000, server_round)

        return None


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = np.average(
            [
                evaluate_res.loss for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        accs = [metrics["accuracy"] for _, metrics in eval_metrics]
        metrics_aggregated["accuracy"] = np.average(accs)
        
        # Write scalars
        self.writer.add_scalar("Training/federated_accuracy", metrics_aggregated["accuracy"], server_round)
        self.writer.add_scalar("Training/federated_std", np.std(accs), server_round)
        self.writer.add_scalar('System/total_time', time.time() - self.start_time, server_round)
        self.writer.add_scalar('System/training_round_time', self.end_training_round - self.start_training_round, server_round)

        return loss_aggregated, metrics_aggregated


    def build_transform_assignments(self, n_clients, transforms):
        transform_assignments = []
        n_transforms = len(transforms)
        if transforms[0] == 'pacs':
            n_clients_per_domain = n_clients // 4
            for i in range(4):
                for j in range(n_clients_per_domain):
                    transform_assignments.append(f'pacs-{i}')
        else:
            for i in range(n_clients):
                transform_assignments.append(transforms[i%n_transforms])
        return transform_assignments


    def set_client_partitions(self, total_num_clients, sample_size, server_round):
        partitions_conf = []
        partitions = self.rng.sample(range(total_num_clients), sample_size)
        for partition_idx in partitions:
            config = {}
            config["partition"] = partition_idx
            config["transform"] = self.transform_assignments[partition_idx]
            partitions_conf.append(copy.deepcopy(config))

        for partition_idx in partitions:
            self.encountered_clients[partition_idx] = True

        if self.writer is not None:
            self.writer.add_scalar("Training/total_num_clients", total_num_clients, server_round)
            self.writer.add_scalar("Training/total_num_sampled", sample_size, server_round)
            self.writer.add_scalar("Training/total_num_met", self.encountered_clients.count(True), server_round)

        return partitions_conf


    def init_scalars(self):
        self.writer.add_scalar("System/bytes_rcv", 0, 0)
        self.writer.add_scalar("System/bytes_sent", 0, 0)
        self.writer.add_scalar("Training/federated_accuracy", 0, 0)
        self.writer.add_scalar("Training/federated_std", 0, 0)
        self.writer.add_scalar('System/total_time', 0, 0)
        self.start_time = time.time()