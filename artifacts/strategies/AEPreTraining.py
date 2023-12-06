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
import os

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

from .TensorboardStrategy import TensorboardStrategy


class AEPreTraining(TensorboardStrategy):
    def __repr__(self) -> str:
        return "AEPreTraining"

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
        dataset
        ):

        super().__init__(min_fit_clients=min_fit_clients, 
                        min_available_clients=min_available_clients, 
                        fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        writer=writer,
                        on_fit_config_fn=on_fit_config_fn,
                        total_num_clients=total_num_clients,
                        transforms=transforms)

        self.dataset = dataset
        os.makedirs(f'./pre-trained/{self.dataset}/federated/', exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results."""

        params, metrics = super().aggregate_fit(server_round, results, failures)
        torch.save(parameters_to_ndarrays(params), f'./pre-trained/{self.dataset}/federated/params.pt')

        return params, metrics