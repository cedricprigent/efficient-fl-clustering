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

from utils.datasets import load_data
from utils.models import Net, LeNet_5_CIFAR, LogisticRegression, weight_reset
from utils.function import test_standard_classifier, test_regression
from utils.app import Clustering_Server, Server
from flwr.server.client_manager import SimpleClientManager

from strategies.TensorboardStrategy import TensorboardStrategy
from strategies.FedMedian import FedMedian
from strategies.Krum import Krum
from strategies.TestEncoding import TestEncoding

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
	logging.basicConfig(filename="log_traces.log", level=logging.INFO)
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--strategy", type=str, default="testencoding", help="Set of strategies: fedavg, testencoding, fedmedian, krum"
	)
	parser.add_argument(
		"--model", type=str, default="cnn", help="Model to train: cnn, regression"
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
		input_size = 28*28
	elif args["dataset"] == "cifar10":
		input_size = 32*32*3
	
	
	# Global Model
	if args['strategy'] == "testencoding" or args['strategy'] == "ifca":
		if args["model"] == "regression":
			model = LogisticRegression(input_size=input_size, num_classes=10).to('cpu')
		elif args["model"] == "cnn":
			if args["dataset"] == "mnist":
				model = Net().to('cpu')
			elif args["dataset"] == "cifar10":
				model = LeNet_5_CIFAR().to('cpu')
		else:
			try:
				raise ValueError('Invalid model name')
			except ValueError as err:
				logging.info('Invalid model name')
				raise

	if args['strategy'] == "testencoding":
		model_init = model.state_dict()
		del model
	elif args['strategy'] == "ifca":
		model_init = []
		for _ in range(n_clusters):
			model.apply(weight_reset)
			model_init.append(model.state_dict())
		del model

	# SummaryWriter
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
		)
	elif args['strategy'] == "testencoding":
		strategy = TestEncoding(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
			n_clusters=args["n_clusters"],
			model_init=model_init
		)
	elif args['strategy'] == "fedmedian":
		strategy = FedMedian(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
		)
	elif args['strategy'] == "krum":
		strategy = Krum(
			min_fit_clients=args["min_fit_clients"],
			min_available_clients=args["min_available_clients"],
			fraction_fit=args["fraction_fit"],
			fraction_evaluate=fraction_eval,
			writer=writer,
			on_fit_config_fn=fit_config,
		)

	# Federation config
	server_config = fl.server.ServerConfig(
		num_rounds=args["num_rounds"]
	)
	
	if args["strategy"] == "testencoding":
		server = Clustering_Server(strategy=strategy)
	else:
		server = Server(client_manager=SimpleClientManager(), strategy=strategy)

	fl.server.start_server(
		server_address=args["server_address"],
		config=server_config,
		strategy=strategy,
		server=server
	)