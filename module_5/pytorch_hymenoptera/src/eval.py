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


def load_model(model: str, pretrained: bool, aug: bool):
    data_dir = '../checkpoints/'
    filename = get_filename(model, pretrained, aug)
    try:
        net = joblib.load(os.path.join(data_dir, filename))
        return net
    except FileNotFoundError:
        logging.error("File not found")


def run(data_dir, model, pretrained=False, aug=False):

    net = load_model(model, pretrained, aug)
    if net is None:
        return

    BATCH_SIZE = 4

    test_dataset = torchvision.datasets.ImageFolder(data_dir, data_transforms['val'])
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_acc = evaluate_accuracy(test_iter, net)
    print('Model {} pretrained={} aug={} - accuracy {}'.format(model, pretrained, aug, test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--pretrained', default=False, type=str2bool)
    parser.add_argument('--aug', default=False, type=str2bool)
    args = parser.parse_args()

    run(
        data_dir=args.datapath,
        model=args.model,
        pretrained=args.pretrained,
        aug=args.aug,
    )

