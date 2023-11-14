import os
import sys
import pathlib
path = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(f"{path}")

from typing import Dict, List, Tuple
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from utils.partition import Partition, FolderPartition

def dirichlet_partitions(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    transform=None,
) -> Tuple[List[Dataset], Dict]:
    np.random.seed(0)
    
    NUM_CLASS = len(dataset.classes)
    MIN_SIZE = 0
    # samples = [[] for _ in range(num_clients)]
    X = [[] for _ in range(num_clients)]
    Y = [[] for _ in range(num_clients)]
    if not isinstance(dataset.targets, np.ndarray):
        dataset.targets = np.array(dataset.targets, dtype=np.int64)
    idx = [np.where(dataset.targets == i)[0] for i in range(NUM_CLASS)]

    while MIN_SIZE < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(NUM_CLASS):
            np.random.shuffle(idx[k])
            distributions = np.random.dirichlet(np.repeat(alpha, num_clients))
            distributions = np.array(
                [
                    p * (len(idx_j) < len(dataset) / num_clients)
                    for p, idx_j in zip(distributions, idx_batch)
                ]
            )
            distributions = distributions / distributions.sum()
            distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
            ]
            MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            np.random.shuffle(idx_batch[k])
            # samples[i] = [dataset.samples[k] for k in idx_batch[i]]
            X[i] = [dataset.samples[k][0] for k in idx_batch[i]]
            Y[i] = [dataset.samples[k][1] for k in idx_batch[i]]

    datasets = [
        FolderPartition(
            data=X[j],
            targets=Y[j],
            # samples=samples[j],
            transform=transform
        )
        for j in range(num_clients)
    ]
    return datasets


if __name__ == '__main__':
    os.makedirs('/tmp/app/data/train', exist_ok=True)
    os.makedirs('/tmp/app/data/test', exist_ok=True)

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_partitions", type=int, required=False, default=25, help="number of partitions per domain"
    )
    parser.add_argument(
        "--alpha", type=float, required=False, default=10
    )
    args = parser.parse_args()
    alpha = args.alpha
    n_partitions = args.n_partitions

    # dataset path
    base_path = '/tmp/app/data/PACS/pacs_data/pacs_data/'
    photos_path = f'{base_path}photo/'
    art_path = f'{base_path}art_painting/'
    cartoon_path = f'{base_path}cartoon/'
    sketch_path = f'{base_path}sketch/'
    paths = [photos_path, art_path, cartoon_path, sketch_path]

    # dataloaders
    transform = Compose([ToTensor(), Resize(size=(64, 64), antialias=True)])
    datasets = [ImageFolder(p, transform=transform) for p in paths]
    dataloaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets]

    print("#################################################")
    print("# Generating partitions for PACS dataset\t#")
    print(f"# - alpha = {alpha}\t\t\t\t\t#")
    print(f"# - partitions per domain = {n_partitions}\t\t\t#")
    print(f"# - total partitions = {n_partitions*4}\t\t\t#")
    print("#################################################")

    for dataloader_idx in range(4):
        partitions = dirichlet_partitions(dataset=dataloaders[dataloader_idx].dataset, num_clients=n_partitions, alpha=alpha, transform=transform)
        
        train_subsets = []
        test_subsets = []
        for partition in partitions:
            split = torch.utils.data.random_split(partition, [0.8, 0.2])
            train_data_split, test_data_split = [transform(default_loader(partition.data[i])) for i in split[0].indices], [transform(default_loader(partition.data[i])) for i in split[1].indices]
            train_targets_split, test_targets_split = [partition.targets[i] for i in split[0].indices], [partition.targets[i] for i in split[1].indices]

            train = Partition(data=train_data_split,
                            targets=train_targets_split,
                            transform=transform,
                            data_type='tensor')
            test = Partition(data=test_data_split,
                            targets=test_targets_split,
                            transform=transform,
                            data_type='tensor')
            train_subsets.append(train)
            test_subsets.append(test)
        
        for i, (train_subset, test_subset) in enumerate(zip(train_subsets, test_subsets)):
                print(f"subset {i + dataloader_idx * n_partitions}: {len(train_subset)} train data, {len(test_subset)} test data")
        
                torch.save(train_subset, f"/tmp/app/data/train/train_subset-{i + dataloader_idx * n_partitions}.pth")
                torch.save(test_subset, f"/tmp/app/data/test/test_subset-{i + dataloader_idx * n_partitions}.pth")