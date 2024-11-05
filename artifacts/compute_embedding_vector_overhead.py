from collections import OrderedDict
import argparse
import time
import csv
import torch
from torchvision.models import resnet18
from utils.models import LeNet_5, AutoEncoder, Conv_AE
from utils.partition import Partition
from utils.function import train_standard_classifier
from utils.clustering_fn import compute_low_dims_per_class
from utils.client import load_transform
from utils.datasets import load_partition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    log_path = '/tmp/logs.csv'

    if args.dataset == 'MNIST':
        print('Running MNIST scenario')

        # hyperparameters
        n_clients = 10
        n_channels = 1
        im_size = 28
        input_size = n_channels*im_size*im_size
        n_classes = 10
        z_dim = 20
        hidden_dim = 50
        transform_instruction = 'rotate90'

        # AE
        ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(device)
        ae.load_state_dict(torch.load("pre-trained/fashion-mnist/ae.pt", map_location=torch.device(device)))
        encoder = ae.encoder

        # Model
        model = LeNet_5(input_h=im_size, in_channels=n_channels, num_classes=n_classes).to(device)


    if args.dataset == 'CIFAR10':
        print('Running CIFAR10 scenario')

        # hyperparameters
        n_clients = 10
        n_channels = 3
        im_size = 32
        input_size = n_channels*im_size*im_size
        n_classes = 10
        z_dim = 768
        transform_instruction = 'label_flip_1'

        # AE
        ae = Conv_AE().to(device)
        ae.load_state_dict(torch.load("pre-trained/cifar100/ae.pt", map_location=torch.device(device)))
        encoder = ae.encoder

        # Model
        model = resnet18().to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes).to(device)


    if args.dataset == 'FEMNIST':
        print('Running FEMNIST scenario')

        # hyperparameters
        n_clients = 10
        n_channels = 1
        im_size = 28
        input_size = n_channels*im_size*im_size
        n_classes = 62
        transform_instruction = 'femnist'

        # AE
        z_dim = 20 
        hidden_dim = 50
        ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(device)
        ae.load_state_dict(torch.load("pre-trained/fashion-mnist/ae.pt", map_location=torch.device(device)))
        encoder = ae.encoder

        # Model
        model = LeNet_5(input_h=im_size, in_channels=n_channels, num_classes=n_classes).to(device)
        
        # saving_path
        saving_path = '/tmp/pacs.logs'


    if args.dataset == 'PACS':
        print('Running PACS scenario')

        # hyperparameters
        n_clients = 10
        n_channels = 3
        im_size = 64
        input_size = n_channels*im_size*im_size
        n_classes = 7
        z_dim = 3072
        transform_instruction = 'pacs'

        # AE
        ae = Conv_AE().to(device)
        ae.load_state_dict(torch.load("pre-trained/cifar100/ae.pt", map_location=torch.device(device)))
        encoder = ae.encoder

        # Model
        model = resnet18().to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes).to(device)

        # saving_path
        saving_path = '/tmp/pacs.logs'


    trainloaders = []    
    for i in range(n_clients):
        transform, target_transform = load_transform(transform_instruction)
        trainloader, _, _ = load_partition(i, 32, transform=transform, target_transform=target_transform)
        trainloaders.append(trainloader)


    config = {"local_epochs": 1}
    start = time.time()
    for i in range(n_clients):
        train_standard_classifier(model=model, train_dataloader=trainloaders[i], config=config, device=device)
    end = time.time()
    training_time = end - start


    start = time.time()
    for i in range(n_clients):
        sample_flat_dim = trainloaders[i].dataset[0][0].flatten().size()[0]
        compute_low_dims_per_class(net=encoder, dataloader=trainloaders[i], output_size=z_dim, sample_size=sample_flat_dim, 
                                    n_classes=n_classes, compression='AE', randomized_quantization=0.1, device=device)
    end = time.time()
    dim_reduction_time = end - start

    print("One epoch training time: ", training_time)
    print("Embedding vector generation time: ", dim_reduction_time)

    with open(log_path, 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        filewriter.writerow([args.dataset, training_time, dim_reduction_time])