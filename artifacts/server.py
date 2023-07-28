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
from utils.datasets import load_data
from utils.models import Net, LogisticRegression
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
dataset = 'mnist'


# Centralized eval function
def get_eval_fn(model):

	# Load test data
	_, testloader, num_examples = load_data(dataset, batch_size)

	# Evaluate funcion
	def evaluate(server_round, weights, conf):
		model.set_weights(weights)  # Update model with the latest parameters

		if args.model == 'cnn':
			loss, accuracy = test_standard_classifier(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy}
		elif args.model == 'regression':
			loss, accuracy = test_regression(model, testloader, device=DEVICE)
			return loss, {"accuracy": accuracy}

	return evaluate


def fig_config(server_round: int):
	"""Return training configuration dict for each round."""
	config = {
		"batch_size": 64,
		"current_round": server_round,
		"local_epochs": args.local_epochs
	}

	return config


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

	args = parser.parse_args()
	# Global Model
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

	# SummaryWriter
	writer = SummaryWriter(log_dir=f"./fl_logs/{args.strategy}", filename_suffix=f'{args.strategy}')

	writer.add_scalar("hp/batch_size", batch_size)
	writer.add_scalar("hp/num_rounds", args.num_rounds)
	writer.add_scalar("hp/min_fit_clients", args.min_fit_clients)
	writer.add_scalar("hp/fraction_fit", args.fraction_fit)
	writer.add_scalar("hp/local_epochs", args.local_epochs)
	writer.add_text("hp/strategy", args.strategy)
	writer.add_text("hp/model", args.model)


	# Optimization strategy
	if args.strategy == "fedavg":
		strategy = TensorboardStrategy(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			fraction_evaluate=fraction_eval,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "testencoding":
		strategy = TestEncoding(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			fraction_evaluate=fraction_eval,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
			n_clusters=args.n_clusters,
			model=model
		)
	elif args.strategy == "fedmedian":
		strategy = FedMedian(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			fraction_evaluate=fraction_eval,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)
	elif args.strategy == "krum":
		strategy = Krum(
			min_fit_clients=args.min_fit_clients,
			min_available_clients=args.min_available_clients,
			fraction_fit=args.fraction_fit,
			fraction_evaluate=fraction_eval,
			eval_fn=get_eval_fn(model),
			writer=writer,
			on_fit_config_fn=fig_config,
		)

	# Federation config
	config = fl.server.ServerConfig(
		num_rounds=args.num_rounds
	)
	
	if args.strategy == "testencoding":
		server = Clustering_Server(strategy=strategy)
	else:
		server = Server(client_manager=SimpleClientManager(), strategy=strategy)

	fl.server.start_server(
		server_address=args.server_address,
		config=config,
		strategy=strategy,
		server=server
	)