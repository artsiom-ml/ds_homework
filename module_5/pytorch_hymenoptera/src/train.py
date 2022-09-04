import os
import argparse
import joblib
import warnings
import PIL
import time
import logging
import torch
import torch.nn as nn
import torchvision

from preprocessing import data_transforms, data_transforms_aug
from models import resnet18_ft, resnet18, vgg, vgg_ft
from utils import str2bool, get_filename

warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(data_iter, net):
    acc_sum, n = torch.Tensor([0]), 0
    net.eval()
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.shape[0]
    return acc_sum.item() / n


def train(net, train_iter, test_iter, trainer, num_epochs):
    loss = nn.CrossEntropyLoss(reduction='sum')
    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))


def build_model(model: str, pretrained: bool):
    if model == 'resnet18' and pretrained is False:
        return resnet18()
    if model == 'resnet18' and pretrained is True:
        return resnet18_ft()
    if model == 'VGG' and pretrained is False:
        return vgg()
    if model == 'VGG' and pretrained is True:
        return vgg_ft()


def run(data_dir, save_to, model, pretrained=False, aug=False):

    impl_models = ['resnet18', 'VGG']
    if model not in impl_models:
        logging.error("Model not found")
        return

    BATCH_SIZE = 4

    if aug is False:
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    else:
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms_aug['train'])
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms_aug['val'])

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = build_model(model, pretrained)
    net.to(device)

    lr, num_epochs = 0.001, 20
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, trainer, num_epochs)

    filename = get_filename(model, pretrained, aug)
    joblib.dump(
        net,
        os.path.join(save_to, filename)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', required=True, type=str)
    parser.add_argument('--save_to', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--pretrained', default=False, type=str2bool)
    parser.add_argument('--aug', default=False, type=str2bool)
    args = parser.parse_args()

    run(
        data_dir=args.datapath,
        save_to=args.save_to,
        model=args.model,
        pretrained=args.pretrained,
        aug=args.aug,
    )
