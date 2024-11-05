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


def test_randomized_quantization_cifar10(args):

    if args['dataset'] == 'cifar10':
        n_partitions = 100
        transforms=["none", "label_flip_1", "label_flip_2", "label_flip_3"]
        model = LeNet_5(input_h=32, in_channels=3, num_classes=10).to('cpu')
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

    if args['dataset'] == 'cifar10':
        partitions_conf = strategy.assign_all_partitions_for_clustering(total_num_clients=100, sample_size=1)[0]
        partitions_conf.update(args)

        n_channels = 1
        im_size = 32
        n_classes = 10
        if args["compression"] == 'AE':
            z_dim = 768
            ae = Conv_AE().to(DEVICE)
            state_dict = load_params(f"{args['path_to_encoder_weights']}/cifar-10/federated/params.pt", ae)
            ae.load_state_dict(state_dict)
            encoder = ae.encoder
        elif args["compression"] == 'Cifar100':
            z_dim = 768
            ae = Conv_AE().to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/cifar100/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder

        args['z_dim'] = z_dim
        args['n_classes'] = n_classes

    clustering_accuracy = []
    n_cluster_centers = []
    rand_values = [i/100 for i in range(0,45,5)]


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
    with open('test_randomized_quantization_cifar10.log', 'a') as file:
        filewriter = csv.writer(file, delimiter=',')
        for i, randomization in enumerate(rand_values):
            filewriter.writerow([args['compression'], randomization, rand_index[i], n_cluster_centers[i]])


if __name__ == '__main__':
    args = {}
    args['dataset'] = 'cifar10'
    args['path_to_encoder_weights'] = './pre-trained'
    args["task"] = "compute_low_dim"

    repeat = 3

    for i in range(repeat):
        args['compression'] = 'AE'
        test_randomized_quantization_cifar10(args)
        
        args['compression'] = 'Cifar100'
        test_randomized_quantization_cifar10(args)