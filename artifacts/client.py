from collections import OrderedDict
from typing import List, Tuple
import argparse
import copy
import json
import sys
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
import random

from utils.datasets import load_partition
from utils.models import Net, LeNet_5_CIFAR, ResNet9, LogisticRegression, AutoEncoder, EncoderNet, EncoderNet_Cifar
from utils.partition_data import Partition
from utils.function import train_standard_classifier, train_regression, test_standard_classifier, test_regression
from utils.transforms import Rotate, LabelFlip, Invert, Equalize
import logging
import flwr as fl
from utils.clustering_fn import compute_low_dims_per_class

torch.manual_seed(0)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64


class StandardClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader=None, valloader=None, sim=True):
        self.model = model
        self.sim = sim
        if not sim:
            self.trainloader = trainloader
            self.valloader = valloader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if self.sim:
            self.load_partition(partition=config['partition'], transform_instruction=config['transform'])

        # local training
        self.set_parameters(parameters)
        if args["model"] == 'cnn':
            train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
        elif args["model"] == 'regression':
            train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        params = self.get_parameters()

        return params, len(self.trainloader), {"transform": str(args["transform"])}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args["model"] == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args["model"] == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        print(f"EVAL model performance - accuracy: {accuracy}, loss: {loss}")

        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def load_partition(self, partition, transform_instruction):
        print(f"Loading partition {partition} with transform {transform_instruction}")
        transform, target_transform = load_transform(transform_instruction)
        self.trainloader, self.valloader, _ = load_partition(partition, batch_size, transform=transform, target_transform=target_transform)


class EncodingClient(StandardClient):
    def __init__(self, model, trainloader=None, valloader=None, sim=True):
        super(EncodingClient, self).__init__(model, trainloader, valloader, sim)

    def fit(self, parameters, config):
        if config["task"] == "compute_low_dim":
            # load partition to use
            if self.sim:
                self.load_partition(partition=config['partition'], transform_instruction=config['transform'])
            
            # compressing local data
            print("CLIENT NUM: ", config["client_number"])
            sample_flat_dim = self.trainloader.dataset[0][0].flatten().size()[0]
            ld = compute_low_dims_per_class(encoder, 
                self.trainloader, 
                output_size=z_dim, 
                device=DEVICE, 
                style_extraction=style_extraction, 
                sample_size=sample_flat_dim
            )
            return [np.array(ld)], len(self.trainloader), {"transform": str(args["transform"]), "client_number": config["client_number"]}
        else:
            # local training
            print("CLIENT NUM: ", config["client_number"])
            self.set_parameters(parameters)
            if args["model"] == 'cnn':
                train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
            elif args["model"] == 'regression':
                train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

            params = self.get_parameters()

            return params, len(self.trainloader), {"transform": str(args["transform"]), "client_number": config["client_number"]}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args["model"] == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args["model"] == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        print(f"Cluster {config['cluster_id']} EVAL model performance - accuracy: {accuracy}, loss: {loss} | transform {args['transform']}")

        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "cluster_id": config['cluster_id']}



class IFCAClient(StandardClient):
    def __init__(self, model, trainloader=None, valloader=None, sim=True):
        super(IFCAClient, self).__init__(model, trainloader, valloader, sim)

    def fit(self, parameters, config):
        if self.sim:
            self.load_partition(partition=config['partition'], transform_instruction=config['transform'])

        # Estimating cluster identity
        if config["round"] == 1:
            cluster_id = random.randint(0, config["n_clusters"]-1)
        else:
            cluster_id = self.estimate_cluster_identity(parameters, n_clusters=config["n_clusters"])

        # Assigning corresponding model for local training
        print("Assigning client to cluster ", cluster_id)
        param_size = len(parameters) // config["n_clusters"]
        self.set_parameters(parameters[cluster_id*param_size:(cluster_id+1)*param_size])
        if args["model"] == 'cnn':
            train_standard_classifier(self.model, self.trainloader, config=config, device=DEVICE, args=args)
        elif args["model"] == 'regression':
            train_regression(self.model, self.trainloader, config=config, device=DEVICE, args=args)

        params = self.get_parameters()

        return params, len(self.trainloader), {"transform": str(args["transform"]), "cluster_id": cluster_id, "client_number": config["client_number"]}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if args["model"] == 'cnn':
            loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
        elif args["model"] == 'regression':
            loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)

        print(f"EVAL model performance - accuracy: {accuracy}, loss: {loss}")

        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "cluster_id": config['cluster_id']}

    def estimate_cluster_identity(self, parameters, n_clusters=2):
        id = 0
        param_size = len(parameters) // n_clusters
        for i in range(n_clusters):
            self.set_parameters(parameters[i*param_size:(i+1)*param_size])
            if args["model"] == 'cnn':
                loss, accuracy = test_standard_classifier(self.model, self.valloader, device=DEVICE)
            elif args["model"] == 'regression':
                loss, accuracy = test_regression(self.model, self.valloader, device=DEVICE)
            if i==0:
                best_loss = loss
            elif loss < best_loss:
                best_loss = loss
                id = i

        return id



def load_transform(transform_instruction):
    target_transform=None
    transform=None
    if transform_instruction == "solarize":
        transform = transforms.Compose([transforms.RandomSolarize(threshold=200.0), transforms.ToTensor()])
    elif transform_instruction == "invert":
        transform = transforms.Compose([Invert(), transforms.ToTensor()])
    elif transform_instruction == "equalize":
        transform = transforms.Compose([Equalize(), transforms.ToTensor()])
    elif transform_instruction == "elastic":
        transform = transforms.Compose([transforms.ElasticTransform(alpha=100.0), transforms.ToTensor()])
    elif transform_instruction == "rotate90":
        transform = transforms.Compose([Rotate(90), transforms.ToTensor()])
    elif transform_instruction == "rotate180":
        transform = transforms.Compose([Rotate(180), transforms.ToTensor()])
    elif transform_instruction == "rotate270":
        transform = transforms.Compose([Rotate(270), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        
    if args["transform"] == "label_flip_1":
        target_transform = LabelFlip(1)
    elif args["transform"] == "label_flip_2":
        target_transform = LabelFlip(2)
    elif args["transform"] == "label_flip_3":
        target_transform = LabelFlip(3)
    elif args["transform"] == "label_flip_4":
        target_transform = LabelFlip(4)

    return transform, target_transform


if __name__ == "__main__":
    error_handler = logging.FileHandler("error.log")
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(level=logging.ERROR)
    error_logger.addHandler(error_handler)

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
        "--path_to_encoder_weights", type=str, required=False, default='/app/artifacts', help="Path to encoder weights"
    )
    parser.add_argument(
        "--client", type=str, required=False, default='EncodingClient', help="EncodingClient, StandardClient"
    )
    parser.add_argument(
        "--compression", type=str, required=False, default='Triplet', help="Triplet, AE, StyleExtraction"
    )
    parser.add_argument(
        "--dataset", type=str, required=False, default="mnist", help="mnist, cifar10"
    )
    parser.add_argument(
        "--config_file", help="Path to json config file"
    )

    args = parser.parse_args()
    args = vars(args)
    if args["config_file"]:
        with open(args["config_file"], 'rt') as f:
            json_config = json.load(f)
        args.update(json_config)

    # Input size
    if args["dataset"] == "mnist":
        n_channels = 1
        im_size = 28
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet
        z_dim = 2
    elif args["dataset"] == "cifar10":
        n_channels = 3
        im_size = 32
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet_Cifar
        z_dim = 20

    if args["model"] == "regression":
        model = LogisticRegression(input_size=input_size, num_classes=10).to(DEVICE)
    elif args["model"] == "cnn":
        if args["dataset"] == "mnist":
            model = Net().to(DEVICE)
        elif args["dataset"] == "cifar10":
            model = LeNet_5_CIFAR().to(DEVICE)
    elif args["model"] == "resnet9":
        model = ResNet9().to('cpu')
    else:
        try:
            raise ValueError('Invalid model name')
        except ValueError as err:
            error_logger.info('Invalid model name')
            raise

    # Testing
    style_extraction = False
    encoder = None
    if args["compression"] == 'AE':
        ae = AutoEncoder(n_channels=n_channels, im_size=im_size).to(DEVICE)
        ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/ae.pt", map_location=torch.device(DEVICE)))
        encoder = ae.encoder
        z_dim = 100
    elif args["compression"] == 'Triplet':
        encoder = encoder_net().to(DEVICE)
        encoder.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/enc_save_orig.pth", map_location=torch.device(DEVICE)))
    elif args["compression"] == 'StyleExtraction':
        style_extraction = True
    else:
        raise NotImplementedError

    if args["sim"]:
        trainloader = None
        testloader = None
    else:
        transform, target_transform = load_transform(args["transform"])
        trainloader, testloader, _ = load_partition(args["num"], batch_size, transform=transform, target_transform=target_transform)

    if args["client"] == "EncodingClient":
        client=EncodingClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader
        )
    elif args["client"] == "IFCAClient":
        client=IFCAClient(
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

    try:
        fl.client.start_numpy_client(
            server_address=args["server_address"],
            client=client
        )
    except Exception as e:
        error_logger.error(traceback.format_exc())
        raise