from typing import Dict, Optional, Tuple
import argparse
import logging
import flwr as fl
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import copy
import os
import time
import json
import traceback

from utils.datasets import load_data
from utils.models import LeNet_5, weight_reset
from torchvision.models import resnet18
from utils.app import Clustering_Server, Server
from flwr.server.client_manager import SimpleClientManager

from strategies.TensorboardStrategy import TensorboardStrategy
from strategies.ClusterEmbeddings import ClusterEmbeddings
from strategies.IFCA import IFCA
from strategies.AEPreTraining import AEPreTraining

torch.manual_seed(0)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
fraction_eval=1


def fit_config(server_round: int):
	"""Return training configuration dict for each round."""
	conf = {
		"batch_size": 64,
		"current_round": server_round,
		"local_epochs": args["local_epochs"]
	}

	return conf


if __name__ == "__main__":
	error_handler = logging.FileHandler("error.log")
	error_logger = logging.getLogger("error_logger")
	error_logger.setLevel(level=logging.ERROR)
	error_logger.addHandler(error_handler)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--strategy", type=str, default="testencoding", help="Set of strategies: fedavg, testencoding"
	)
	parser.add_argument(
		"--model", type=str, default="lenet5", help="Model to train: lenet5, resnet18"
	)
	parser.add_argument(
		"--server_address", type=str, required=False, default="127.0.0.1:8080", help="gRPC server address"
	)
	parser.add_argument(
		"--num_rounds", type=int, required=False, default=20, help="number of FL rounds"
	)
	parser.add_argument(
		"--fraction_fit", type=float, required=False, default=1, help="Fraction of clients selected on each rounds"
	)
	parser.add_argument(
		"--min_fit_clients", type=int, required=False, default=2, help="Minimum number of clients selected on each rounds"
	)
	parser.add_argument(
		"--min_available_clients", type=int, required=False, default=2, help="Minimum number of clients selected on each rounds"
	)
	parser.add_argument(
		"--local_epochs", type=int, required=False, default=1, help="Local epochs"
	)
	parser.add_argument(
		"--n_clusters", type=int, required=False, default=3, help="Number of clusters"
	)
	parser.add_argument(
		"--dataset", type=str, required=False, default="mnist", help="mnist, cifar10"
	)
	parser.add_argument(
		"--compression", type=str, required=False, default='Undifined', help="Triplet, AE, StyleExtraction"
	)
	parser.add_argument(
		"--total_num_clients", type=int, required=False, default=100, help="Total number of clients/partitions"
	)
	parser.add_argument(
		"--transforms", type=str, required=False, default='None', help="List of transforms separated by a , (e.g., None,label_flip_1)"
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
	else:
		args["transforms"] = args["transforms"].split(',')

	# Input size
	if args["dataset"] == "mnist":
		n_channels = 1
		im_size = 28
		input_size = n_channels*im_size*im_size
		n_classes = 10
	elif args["dataset"] == "cifar10":
		n_channels = 3
		im_size = 32
		input_size = n_channels*im_size*im_size
		n_classes = 10
	elif args["dataset"] == "pacs":
		n_channels = 3
		im_size = 64
		input_size = n_channels*im_size*im_size
		n_classes = 7
	elif args["dataset"] == "femnist":
		n_channels = 1
		im_size = 28
		input_size = n_channels*im_size*im_size
		n_classes = 62
	
	
	# Global Model
	if args['strategy'] == "testencoding" or args['strategy'] == "ifca":
		if args["model"] == "lenet5":
			n_base_layers = 4
			model = LeNet_5(input_h=im_size, in_channels=n_channels, num_classes=n_classes).to('cpu')
		elif args["model"] == "resnet18":
			# n_base_layers = 90
			# n_base_layers = 120 # max base
			n_base_layers = 0
			model = resnet18().to('cpu')
			model.fc = torch.nn.Linear(model.fc.in_features, n_classes).to('cpu')
		else:
			try:
				raise ValueError('Invalid model name')
			except ValueError as err:
				error_logger.info('Invalid model name')
				raise

	if args['strategy'] == "testencoding":
		model_init = model.state_dict()
		del model
	elif args['strategy'] == "ifca":
		model_init = []
		for _ in range(args["n_clusters"]):
			model.apply(weight_reset)
			model_init.append(copy.deepcopy(model.state_dict()))
		del model

	# SummaryWriter
	if args['strategy'] == 'testencoding':
		writer = SummaryWriter(log_dir=f"./fl_logs/{args['strategy']}-{args['compression']}", filename_suffix=f"{args['strategy']}")
	else:
		writer = SummaryWriter(log_dir=f"./fl_logs/{args['strategy']}", filename_suffix=f"{args['strategy']}")

	writer.add_scalar("hp/batch_size", batch_size)
	writer.add_scalar("hp/num_rounds", args['num_rounds'])
	writer.add_scalar("hp/min_fit_clients", args['min_fit_clients'])
	writer.add_scalar("hp/fraction_fit", args['fraction_fit'])
	writer.add_scalar("hp/local_epochs", args['local_epochs'])
	writer.add_text("hp/strategy", args['strategy'])
	writer.add_text("hp/model", args['model'])


	# Optimization strategy
	if args['strategy'] == "fedavg":
		strategy = TensorboardStrategy(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
			total_num_clients=args["total_num_clients"],
			transforms=args["transforms"]
		)
	elif args['strategy'] == "ae-pretraining":
		strategy = AEPreTraining(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
			total_num_clients=args["total_num_clients"],
			transforms=args["transforms"],
			dataset=args["dataset"]
		)
	elif args['strategy'] == "testencoding":
		strategy = ClusterEmbeddings(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
			n_clusters=args["n_clusters"],
			model_init=model_init,
			total_num_clients=args["total_num_clients"],
			transforms=args["transforms"],
			n_base_layers=n_base_layers
		)
	elif args['strategy'] == "ifca":
		strategy = IFCA(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
			n_clusters=args["n_clusters"],
			model_init=model_init,
			total_num_clients=args["total_num_clients"],
			transforms=args["transforms"],
			n_base_layers=n_base_layers
		)

	# Federation config
	server_config = fl.server.ServerConfig(
		num_rounds=args["num_rounds"]
	)
	
	if args["strategy"] == "testencoding":
		server = Clustering_Server(strategy=strategy)
	else:
		server = Server(client_manager=SimpleClientManager(), strategy=strategy)

	try:
		fl.server.start_server(
			server_address=args["server_address"],
			config=server_config,
			strategy=strategy,
			server=server
		)
	except Exception as e:
		error_logger.error(traceback.format_exc())
		raise