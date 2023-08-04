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
flat_shape = [28*28]
cond_shape=10
logging.basicConfig(filename="log_traces.log", level=logging.INFO)

torch.manual_seed(0)

def train_standard_classifier(model, train_dataloader, config, device=DEVICE, args=None):
    """Train the network on the training set."""

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)

            # 1. Forward pass
            c_out = model(images)                        
            y_onehot = F.one_hot(labels, cond_shape).to(device)

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
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))


def train_regression(model, train_dataloader, config, device=DEVICE, args=None):
    """Train the network on the training set."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["local_epochs"]):
        train_loss = 0
        classif_accuracy = 0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device) #[64, 1, 28, 28]
            labels = labels.to(device)

            # 1. Forward pass
            log_probs = model(images)
            
            # 2. Calculate loss
            loss = loss_func(log_probs, labels)
            train_loss += loss.item()
            classif_accuracy += accuracy_fn(labels, torch.argmax(log_probs, dim=1))

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
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))
            
        logging.info('====> Epoch: {} Average loss: {:.4f}\tClassifier Accuracy: {:.4f}'.format(
            epoch, train_loss / len(train_dataloader.dataset), classif_accuracy/len(train_dataloader)))


def test_standard_classifier(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            c_out = model(X)

            y_onehot = F.one_hot(y, cond_shape).to(device)

            # 2. Loss
            loss = loss_fn_standard_classifier(c_out, y_onehot)
            test_loss += loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(c_out, dim=1))


    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, classif_accuracy/len(test_dataloader)


def test_regression(model, test_dataloader, device=DEVICE):
    #Sets the module in evaluation mode
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    classif_accuracy = 0
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. Forward pass
            log_probs = model(X)

            # 2. Loss
            loss = loss_fn(log_probs, y)
            test_loss += loss.item()
            classif_accuracy += accuracy_fn(y, torch.argmax(log_probs, dim=1))


    test_loss /= len(test_dataloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss, classif_accuracy/len(test_dataloader)


def loss_fn(recon, x, mu, logvar, c_out, y_onehot, loss=torch.nn.BCELoss, device=DEVICE):
    y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
    classif_loss = loss()(c_out, y_onehot1)
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return classif_loss+BCE+KLD, classif_loss, BCE, KLD


def loss_fn_standard_classifier(c_out, y_onehot, device=DEVICE):
    y_onehot1 = y_onehot.type(torch.FloatTensor).to(device)
    classif_loss = torch.nn.BCELoss()(c_out, y_onehot1)
    return classif_loss


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

def train(model, trainloader):
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=1e-3)
    train_loss = 0
    for epochs in range(10):
        for batch, (x,y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            # 1. Forward pass
            x = x.view(-1, 3, 32, 32)
            recon_batch = model(x)
            
            # 2. Calculate loss
            loss = F.binary_cross_entropy(recon_batch, x, reduction='sum')
            train_loss += loss.item()

            # 3. Zero grad
            optim.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optim.step()
        print(f'epoch {epochs} completed !')


def train_autoencoder(model, trainloader, epochs):
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=1e-3)
    train_loss = 0
    for epochs in range(epochs):
        for batch, (x,y) in enumerate(trainloader):
            x = x.to(device)
            y = y.to(device)
            # 1. Forward pass
            x = x.view(-1, 3, 32, 32)
            recon_batch = model(x)
            
            # 2. Calculate loss
            loss = F.binary_cross_entropy(recon_batch, x, reduction='sum')
            train_loss += loss.item()

            # 3. Zero grad
            optim.zero_grad()

            # 4. Backprop
            loss.backward()

            # 5. Step
            optim.step()