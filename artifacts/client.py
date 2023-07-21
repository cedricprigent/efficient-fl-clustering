from collections import OrderedDict
from typing import List, Tuple
import argparse
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

from utils.datasets import load_partition
from utils.models import Net, LogisticRegression, AutoEncoder, EncoderNet
from utils.partition_data import Partition
from utils.function import train_standard_classifier, train_regression, test_standard_classifier, test_regression
import logging
import flwr as fl
from utils.clustering_fn import compute_low_dims_per_class

torch.manual_seed(0)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # local training
        if args.model == 'cnn':
            train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
        elif args.model == 'regression':
            train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        # compute low dimensional representation of local data
        ld = compute_low_dims_per_class(encoder, self.trainloader, device=DEVICE)

        # concatenating model weights and low dim to return to the server
        params = self.get_parameters()
        params.append(ld)

        return params, len(self.trainloader), {"transform": str(args.transform)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args.model == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args.model == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="cnn", help="Model to train: cnn, regression"
    )
    parser.add_argument(
        "--num", type=int, required=False, default=0, help="client number"
    )
    parser.add_argument(
        "--server_address", type=str, required=False, default="127.0.0.1:8080", help="gRPC server address"
    )
    parser.add_argument(
        "--transform", type=str, required=False, default=None, help="Transform to apply to input data"
    )
    args = parser.parse_args()

    model = LogisticRegression(input_size=28*28, num_classes=10).to(DEVICE)

    # Testing
    #encoder = AutoEncoder().to(DEVICE).encoder
    encoder = EncoderNet().to(DEVICE)
    encoder.load_state_dict(torch.load('/app/artifacts/enc_save_orig.pth'))

    if args.transform == "solarize":
        transform = transforms.Compose([transforms.RandomSolarize(threshold=200.0), transforms.ToTensor()])
    if args.transform == "elastic":
        transform = transforms.Compose([transforms.ElasticTransform(alpha=100.0), transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    trainloader, testloader, _ = load_partition(args.num, batch_size, transform=transform)

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    )