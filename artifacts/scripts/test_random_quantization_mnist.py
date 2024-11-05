import os
import sys
import csv
import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(f"{path}")

from strategies.ClusterEmbeddings import ClusterEmbeddings
from utils.models import AutoEncoder, Conv_AE, LeNet_5
from utils.client import EncodingClient
from utils.clustering_fn import make_clusters, print_clusters, compute_clustering_acc
from utils.partition import Partition
from client import load_params

import random
import torch

random.seed(0)
DEVICE = 'cuda'


def test_randomized_quantization_mnist(args):

    if args['dataset'] == 'mnist':
        n_partitions = 100
        transforms=["none", "rotate90", "rotate180", "rotate270"]
        model = LeNet_5(input_h=28, in_channels=1, num_classes=10).to('cpu')
        n_clusters = 4
        min_cluster_size = 2

    strategy = ClusterEmbeddings(
            min_fit_clients=1,
            min_available_clients=1,
            fraction_fit=1,
            fraction_evaluate=1,
            writer=None,
            on_fit_config_fn=None,
            n_clusters=n_clusters,
            model_init=model.state_dict(),
            total_num_clients=n_partitions,
            transforms=transforms,
            n_base_layers=None,
            clustering_strategy='agglomerative',
            min_cluster_size=min_cluster_size)

    if args['dataset'] == 'mnist':
        partitions_conf = strategy.assign_all_partitions_for_clustering(total_num_clients=100, sample_size=1)[0]
        partitions_conf.update(args)

        n_channels = 1
        im_size = 28
        n_classes = 10
        if args["compression"] == 'AE':
            z_dim = 20
            hidden_dim = 50
            ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            state_dict = load_params(f"{args['path_to_encoder_weights']}/mnist/federated/params.pt", ae)
            ae.load_state_dict(state_dict)
            encoder = ae.encoder
        elif args["compression"] == 'FashionMNIST':
            z_dim = 20
            hidden_dim = 50
            ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/fashion-mnist/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder

        args['z_dim'] = z_dim
        args['n_classes'] = n_classes

    clustering_accuracy = []
    n_cluster_centers = []
    rand_values = [i/100 for i in range(0,50,5)]

    for randomization in rand_values:
        args['randomized_quantization'] = randomization
        client = EncodingClient(
            model=None, 
            trainloader=None, 
            valloader=None, 
            sim=True, 
            embedding_model=encoder,
            args=args
        )

        low_dims, _, _= client.fit(parameters=None, config=partitions_conf)
        low_dims = low_dims[0]

        cluster_labels, cluster_centers, _ = make_clusters(low_dims, 
                                                            n_clusters=n_clusters, 
                                                            n_clients=len(low_dims), 
                                                            min_cluster_size=min_cluster_size, 
                                                            clustering_strategy='agglomerative')
        cluster_truth = strategy.transform_assignments

        clustering_acc = compute_clustering_acc(cluster_labels, cluster_truth, n_clusters=n_clusters)
        clustering_accuracy.append(clustering_acc)
        n_cluster_centers.append(len(cluster_centers))

    rand_index = clustering_accuracy

    for i, randomization in enumerate(rand_values):
        print(f'Randomization: {randomization} - Rand Index: {rand_index[i]}')

    # Write logs to csv
    with open('test_randomized_quantization_mnist.log', 'a') as file:
        filewriter = csv.writer(file, delimiter=',')
        for i, randomization in enumerate(rand_values):
            filewriter.writerow([args['compression'], randomization, rand_index[i], n_cluster_centers[i]])



if __name__ == '__main__':
    args = {}
    args['dataset'] = 'mnist'
    args['path_to_encoder_weights'] = './pre-trained'
    args["task"] = "compute_low_dim"

    repeat = 3

    for i in range(repeat):
        args['compression'] = 'AE'
        test_randomized_quantization_mnist(args)
        
        args['compression'] = 'FashionMNIST'
        test_randomized_quantization_mnist(args)