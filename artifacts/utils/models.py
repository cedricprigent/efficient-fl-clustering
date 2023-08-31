from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os

# Hard coding the value for testing purpose
flat_shape = [28*28]
z_dim = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Classifier_small(nn.Module):
    def __init__(self, dim_y):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flat_shape[0], out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=dim_y),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        c_out = self.classifier(x)
        return c_out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class Classifier(nn.Module):
    def __init__(self, dim_y):
        """
        McMahan et al., 2016; 1,663,370 parameters
        """
        super(Classifier, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(32 * 2) * (7 * 7), out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)



class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.input_size = input_size
    
    def forward(self, inputs, device=DEVICE):
        x = inputs.to(device)
        x = x.view(-1, self.input_size)
        out = self.linear(x)
        return out

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

class LeNet_5(nn.Module):

    def __init__(self, input_h=28, in_channels=1, num_classes=10):
        super(LeNet_5, self).__init__()
        
        # conv_maxpool_output = (input_h - kernel_size + 1) / 2 
        features_output_h = int((((input_h - 4)/2) - 4)/2)
        features_output_size = features_output_h * features_output_h

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 16*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16*in_channels, 32*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(features_output_size*32*in_channels, 120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels, 84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return probas

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    """https://github.com/Moddy2024/ResNet-9/blob/main/resnet-9.ipynb"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return F.softmax(out, dim=1)

    def set_weights(self, weights):
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)



class AutoEncoder(torch.nn.Module):
    def __init__(self, n_channels, im_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_channels * im_size * im_size, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, z_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, n_channels * im_size * im_size),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = torch.reshape(self.decoder(encoded), (-1, n_channels, im_size, im_size))
        return decoded


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        # print(f'after convnet' + str(output.size()))
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_encoding(self, x):
        return self.forward(x)


class EncoderNet_Cifar(nn.Module):
    def __init__(self):
        super(EncoderNet_Cifar, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 64, 6), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 6), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, 20)
                                )

    def forward(self, x):
        output = self.convnet(x)
        # print(f'after convnet' + str(output.size()))
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_encoding(self, x):
        return self.forward(x)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()