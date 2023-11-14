import argparse
import json
import traceback
import torch
import torchvision.transforms as transforms
import logging
import flwr as fl

from utils.datasets import load_partition
from utils.models import LeNet_5, ResNet9, AutoEncoder, EncoderNet, EncoderNet_Cifar, Conv_AE
from torchvision.models import resnet18
from utils.partition import Partition, FolderPartition
from utils.client import StandardClient, EncodingClient, IFCAClient, load_partition

torch.manual_seed(0)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64

if __name__ == "__main__":
    error_handler = logging.FileHandler("error.log")
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(level=logging.ERROR)
    error_logger.addHandler(error_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="cnn", help="Model to train: cnn, resnet9"
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
        "--compression", type=str, required=False, default='AE', help="Triplet, AE, StyleExtraction"
    )
    parser.add_argument(
        "--dataset", type=str, required=False, default="mnist", help="mnist, cifar10"
    )
    parser.add_argument(
        "--randomized_quantization", type=float, required=False, default=0.1, help="Randomization rate for quantization"
    )
    parser.add_argument(
        "--sim", action='store_true', default=True
    )
    parser.add_argument(
        "--no-sim", action='store_false', dest='sim'
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
    encoder = None
    if args["dataset"] == "mnist":
        n_channels = 1
        im_size = 28
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet
        n_classes = 10
        if args["compression"] == 'AE':
            z_dim = 50
            hidden_dim = 100
            ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/mnist/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder
        elif args["compression"] == 'Triplet':
            encoder = encoder_net().to(DEVICE)
            encoder.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/mnist/enc_save_orig.pth", map_location=torch.device(DEVICE)))
            z_dim = 2
        elif args["compression"] == 'StyleExtraction':
            z_dim = 2
        else:
            raise NotImplementedError

    
    elif args["dataset"] == "cifar10":
        n_channels = 3
        im_size = 32
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet_Cifar
        n_classes = 10
        if args["compression"] == 'AE':
            z_dim = 768
            # ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            ae = Conv_AE().to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/cifar-10/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder
        elif args["compression"] == 'feature_embedding':
            z_dim = 10
            encoder = LeNet_5(input_h=32, in_channels=3, num_classes=10).to(DEVICE)
            encoder.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/cifar-10/le_net_feature_embedding.pt", map_location=torch.device(DEVICE)))
            encoder = torch.nn.Sequential(encoder, torch.nn.Flatten())
        elif args["compression"] == 'Triplet':
            encoder = encoder_net().to(DEVICE)
            encoder.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/cifar-10/enc_save_orig.pth", map_location=torch.device(DEVICE)))
            z_dim = 20
        elif args["compression"] == 'StyleExtraction':
            z_dim = 20
        else:
            raise NotImplementedError

    elif args["dataset"] == "pacs":
        n_channels = 3
        im_size = 64
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet_Cifar
        n_classes = 7
        if args["compression"] == 'AE':
            z_dim = 3072
            # ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            ae = Conv_AE().to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/pacs/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder
        elif args["compression"] == 'StyleExtraction':
            z_dim = 20
        else:
            raise NotImplementedError


    elif args["dataset"] == "femnist":
        n_channels = 1
        im_size = 28
        input_size = n_channels*im_size*im_size
        encoder_net = EncoderNet
        n_classes = 62
        if args["compression"] == 'AE':
            z_dim = 50 
            hidden_dim = 100
            ae = AutoEncoder(n_channels=n_channels, im_size=im_size, z_dim=z_dim, hidden_dim=hidden_dim).to(DEVICE)
            ae.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/mnist/ae.pt", map_location=torch.device(DEVICE)))
            encoder = ae.encoder
        elif args["compression"] == 'Triplet':
            encoder = encoder_net().to(DEVICE)
            encoder.load_state_dict(torch.load(f"{args['path_to_encoder_weights']}/mnist/enc_save_orig.pth", map_location=torch.device(DEVICE)))
            z_dim = 2
        elif args["compression"] == 'StyleExtraction':
            z_dim = 2
        else:
            raise NotImplementedError


    if args["model"] == "cnn":
        model = LeNet_5(input_h=im_size, in_channels=n_channels, num_classes=n_classes).to(DEVICE)
    elif args["model"] == "resnet9":
        # model = ResNet9(in_channels=n_channels).to(DEVICE)
        model = resnet18().to(DEVICE)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes).to(DEVICE)
    else:
        try:
            raise ValueError('Invalid model name')
        except ValueError as err:
            error_logger.info('Invalid model name')
            raise


    if args["sim"]:
        trainloader = None
        testloader = None
    else:
        transform, target_transform = load_transform(args["transform"])
        trainloader, testloader, _ = load_partition(args["num"], batch_size, transform=transform, target_transform=target_transform)

    if args["client"] == "EncodingClient":
        args["z_dim"] = z_dim
        args["n_classes"] = n_classes
        client=EncodingClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader,
            embedding_model=encoder,
            args=args
        )
    elif args["client"] == "IFCAClient":
        client=IFCAClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader,
            args=args
        )
    else:
        client=StandardClient(
            model=model,
            trainloader=trainloader,
            valloader=testloader,
            args=args
        )

    try:
        fl.client.start_numpy_client(
            server_address=args["server_address"],
            client=client
        )
    except Exception as e:
        error_logger.error(traceback.format_exc())
        raise