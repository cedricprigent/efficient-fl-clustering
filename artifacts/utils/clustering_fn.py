import torch
import numpy as np
from sklearn.cluster import KMeans
import logging

z_dim = 100

def select_samples(dataloader, label, device):
    chosen_indices = []
    samples = []
    for _, (x, y) in enumerate(dataloader):
        for i in range(len(x)):
            if y[i].detach().cpu().numpy() == label:
                samples.append(x[i].to(device))
            
    return samples


def split_by_class(dataloader, device):
    samples = [[] for _ in range(10)]
    for _, (x, y) in enumerate(dataloader):
        for i in range(len(x)):
            samples[y[i].detach().cpu().numpy()].append(x[i].to(device))

    return samples


def compute_low_dims(net, batch):
    try:
        return net(torch.stack(batch))
    except:
        return torch.rand(1, z_dim)


def compute_avg(low_dim):
    return low_dim.mean(0).cpu().detach().numpy()


def compute_low_dims_per_class(net, dataloader, device):
    ld = np.array([])
    class_batches = split_by_class(dataloader, device=device)
    for class_id in range(10):
        ld = np.append(ld, compute_avg(compute_low_dims(net, class_batches[class_id])))
    return ld


def make_clusters(low_dims, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters).fit(low_dims)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return labels, centers


def print_clusters(labels, cluster_truth, n_clusters):
    logging.basicConfig(filename="log_traces.log", level=logging.INFO)
    for i in range(n_clusters):
        print(f'CLUSTER {i}:')
        logging.info(f'CLUSTER {i}:')
        for j in range(len(labels)):
            if labels[j] == i:
                print(f"Client {j} - {cluster_truth[j]}")
                logging.info(f"Client {j} - {cluster_truth[j]}")
        print("####")
        logging.info("####")