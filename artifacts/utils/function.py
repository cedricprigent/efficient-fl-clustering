import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from torchvision.utils import save_image, make_grid
import sys
import random
import logging

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(filename="log_traces.log", level=logging.INFO)

torch.manual_seed(0)

def train_standard_classifier(model, train_dataloader, config, device=DEVICE, args=None):
    """Train the network on the training set."""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    drop_batch = 0
    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)

            # Drop batch if it contains only one sample for batch normalization
            if images.shape[0] == 1:
                drop_batch = 1
                continue

            # 1. Forward pass
            c_out = model(images)
            dim_y = c_out.shape[1]             
            y_onehot = F.one_hot(labels, dim_y).to(device)

            # 2. Calculate loss
            loss = criterion(c_out, labels)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(c_out, dim=1))

            # 3. Zero grad
            optimizer.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optimizer.step()

            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch * len(images),
                    len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader),
                    loss.item() / len(images)))
        print('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/(len(train_dataloader) - drop_batch)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/(len(train_dataloader) - drop_batch)))


def test_standard_classifier(model, test_dataloader, n_test_batches=None, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    classif_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    if n_test_batches is None:
        stop_iteration = -1
        n_batches = len(test_dataloader)
    else:
        stop_iteration = n_test_batches
        n_batches = n_test_batches
    total_samples = 0
    print(f'Stop iteration: {stop_iteration}')
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            if i == stop_iteration:
                break
            total_samples += len(X)
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            c_out = model(X)

            # 2. Loss
            loss = criterion(c_out, y)
            test_loss += loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))

    # test_loss /= len(test_dataloader.dataset)
    test_loss /= total_samples
    # classif_accuracy /= len(test_dataloader)
    classif_accuracy /= n_batches
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, classif_accuracy


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def train_autoencoder(model, trainloader, config, device=DEVICE):
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.MSELoss()
    for epochs in range(config["local_epochs"]):
        train_loss = 0
        for batch, (x,y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            # 1. Forward pass
            recon_batch = model(x)
            
            # 2. Calculate loss
            loss = criterion(recon_batch, x)
            
            train_loss += loss.item()

            # 3. Zero grad
            optim.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optim.step()
        print('train_loss: ', train_loss / len(trainloader))
        print(f'epoch {epochs} completed !')