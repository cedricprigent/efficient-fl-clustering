from typing import Union, Dict, List, Optional, Tuple

import flwr as fl
import psutil
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
        eval_fn,
        writer,
        on_fit_config_fn):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        evaluate_fn=eval_fn,
                        on_fit_config_fn=on_fit_config_fn)
        
        self.writer = writer
        self.bytes_recv_init_counter = psutil.net_io_counters().bytes_recv
        self.bytes_sent_init_counter = psutil.net_io_counters().bytes_sent

    def configure_fit(
        self, server_round, parameters, client_manager
    ):
        """Configure the next round of training."""
        clients_conf = super().configure_fit(server_round, parameters, client_manager)

        self.writer.add_scalar("Training/total_num_clients", len(clients_conf), server_round)

        # Return client/config pairs
        return clients_conf

    def evaluate(self, server_round, parameters):
        """Evaluate model parameters using an evaluation function."""
        #loss, metrics = super().evaluate(server_round, parameters)

        # Write scalars
        #self.writer.add_scalar("Training/test_c_loss", loss, server_round)
        #self.writer.add_scalar("Training/test_accuracy", metrics["accuracy"], server_round)
        self.writer.add_scalar("System/bytes_rcv", (psutil.net_io_counters().bytes_recv - self.bytes_recv_init_counter) / 1000000, server_round)
        self.writer.add_scalar("System/bytes_sent", (psutil.net_io_counters().bytes_sent - self.bytes_sent_init_counter) / 1000000, server_round)

        return None
        #return loss, metrics


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

        return loss_aggregated, metrics_aggregated
