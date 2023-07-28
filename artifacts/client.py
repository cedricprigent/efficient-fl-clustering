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
from utils.transforms import Rotate, LabelFlip
import logging
import flwr as fl
from utils.clustering_fn import compute_low_dims_per_class

torch.manual_seed(0)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
logging.basicConfig(filename="log_traces/logfilename.log", level=logging.INFO)


class EncodingClient(fl.client.NumPyClient):
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
        if config["task"] == "compute_low_dim":
            # compressing local data
            print("CLIENT NUM: ", config["client_number"])
            ld = compute_low_dims_per_class(encoder, self.trainloader, output_size=z_dim, device=DEVICE)
            return [np.array(ld)], len(self.trainloader), {"transform": str(args.transform), "client_number": config["client_number"]}
        else:
            # local training
            print("CLIENT NUM: ", config["client_number"])
            self.set_parameters(parameters)
            if args.model == 'cnn':
                train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            elif args.model == 'regression':
                train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

            params = self.get_parameters()

            return params, len(self.trainloader), {"transform": str(args.transform), "client_number": config["client_number"]}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args.model == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args.model == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        print(f"Cluster {config['cluster_id']} EVAL model performance - accuracy: {accuracy}, loss: {loss} | transform {args.transform}")

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



class StandardClient(fl.client.NumPyClient):
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
        # local training
        self.set_parameters(parameters)
        if args.model == 'cnn':
            train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
        elif args.model == 'regression':
            train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        params = self.get_parameters()

        return params, len(self.trainloader), {"transform": str(args.transform)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args.model == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args.model == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        print(f"EVAL model performance - accuracy: {accuracy}, loss: {loss}")

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
    parser.add_argument(
        "--path_to_encoder_weights", type=str, required=False, default='/app/artifacts/enc_save_orig.pth', help="Path to encoder weights"
    )
    parser.add_argument(
        "--client", type=str, required=False, default='EncodingClient', help="EncoderClient, StandardClient"
    )
    args = parser.parse_args()

    if args.model == "regression":
        model = LogisticRegression(input_size=28*28, num_classes=10).to(DEVICE)
    elif args.model == "cnn":
        model = Net().to(DEVICE)
    else:
        try:
            raise ValueError('Invalid model name')
        except ValueError as err:
            logging.info('Invalid model name')
            raise

    # Testing
    # encoder = AutoEncoder().to(DEVICE).encoder
    #z_dim = 100
    encoder = EncoderNet().to(DEVICE)
    encoder.load_state_dict(torch.load(args.path_to_encoder_weights, map_location=torch.device(DEVICE)))
    z_dim = 2

    target_transform=None
    if args.transform == "solarize":
        transform = transforms.Compose([transforms.RandomSolarize(threshold=200.0), transforms.ToTensor()])
    elif args.transform == "elastic":
        transform = transforms.Compose([transforms.ElasticTransform(alpha=100.0), transforms.ToTensor()])
    elif args.transform == "rotate90":
        transform = transforms.Compose([Rotate(90), transforms.ToTensor()])
    elif args.transform == "rotate180":
        transform = transforms.Compose([Rotate(180), transforms.ToTensor()])
    elif args.transform == "rotate270":
        transform = transforms.Compose([Rotate(270), transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    if args.transform == "label_flip":
        target_transform = LabelFlip()

    trainloader, testloader, _ = load_partition(args.num, batch_size, transform=transform, target_transform=target_transform)

    if args.client == "EncodingClient":
        client=EncodingClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    else:
        client=StandardClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )