from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import scipy.stats as stats
from torchvision import datasets, transforms
from MMNIST.make_ds import MnistBagDataset
import shutil

import random
import torch.backends.cudnn as cudnn
from MMNIST.model import MILModel
from util import get_sampler, get_transforms, get_btransforms
from lr_scheduler import build_scheduler
import os

import visdom
import time

# print(imgs.shape, labels.shape)

noise = np.random.randn(64, 3, 64, 64)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=3, metavar='T',
                    help='bags have positive labels if they contain at least one target')
parser.add_argument('--mean_bag_length', type=int, default=64, metavar='ML',
                    help='average bag length')
parser.add_argument('--num_bags_train', type=int, default=2000, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=1000, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pretrained', type=bool, default='0', help='whether load and freeze')
parser.add_argument('--dataset', type=str, default='MNIST', help='Choose dataset')
parser.add_argument('--test', type=str, default='single', help='Choose test mode')
parser.add_argument('--pooling', type=str, default='rgp', help='Choose pooling component')
parser.add_argument('--bagsize', type=int, default='512', help='Choose test mode')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def build_model(dataset, cuda):
    print('Init Model')
    if cuda:
        device = "cuda"
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    m = MILModel(device, dataset=dataset)
    if cuda:
        m.cuda()
    return m


def get_loader(dataset, target_number=0):
    loader_kwargs = {'num_workers': 0, 'pin_memory': True}
    if dataset == "MNIST":
        bagsize = 64
        if args.bagsize > 64:
            bagsize = args.bagsize
        train_loader = data_utils.DataLoader(
            MnistBagDataset(target_number=target_number, training_bags=4000 * (bagsize // 64),
                            train=True),
            batch_size=1,
            shuffle=True,
            **loader_kwargs)

        test_loader = data_utils.DataLoader(
            MnistBagDataset(target_number=target_number, training_bags=4000 * (bagsize // 64),
                            train=False, mode=args.test),
            batch_size=1,
            shuffle=False,
            **loader_kwargs)

        return train_loader, test_loader

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), weight_decay=args.reg)

    return optimizer


def train(model, optimizer, epo, train_loader):
    model.train()
    train_loss = 0.
    correct = 0
    correctposinsts = 0
    positiveinsts = 1
    d = 0.0
    total = 0
    acc = 0.0
    instsacc = 0.0
    diff = 0.0

    tmpdata = []
    tmplabel = []
    bagsize = args.bagsize
    for batch_idx, (data, label) in enumerate(train_loader):
        if args.dataset == "MNIST":
            if bagsize > 64:
                if batch_idx % (bagsize // 64):
                    tmpdata.append(data[0])
                    tmplabel.append(label[0][1:])
                    continue
                else:
                    tmpdata.append(data[0])
                    tmplabel.append(label[0][1:])
                    data = torch.cat(tmpdata, dim=0)
                    label = torch.cat(tmplabel, dim=0)
                    tmpdata.clear()
                    tmplabel.clear()
            else:
                data = data[0][64 - bagsize:]
                label = label[0][65 - bagsize:]
            bag_label = torch.max(label)

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        optimizer.zero_grad()
        loss, preds, weights = model.calculate_objective(data, bag_label, args.pooling)
        train_loss += loss.data
        # backward pass
        loss.backward()
        optimizer.step()
        if bag_label.long() > 0:
            if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                correct += 1
        elif bag_label.long() == 0 and torch.sum(preds) == 0:
            correct += 1
        else:
            pass
        label = label.cpu()
        bag_label = bag_label.long().cpu()

        if bag_label > 0:
            if args.pooling == "rgp" or args.pooling == "abp" or args.pooling == "gab" or args.pooling == "dsp":
                # check rgp correctness in different streams
                for i in range(bag_label - 1, -1, -1):
                    idx = (label == (i + 1))
                    ridx = ~idx
                    if idx.any():
                        positiveinsts += 1
                        w = weights[i][idx]
                        rw = weights[i][ridx]
                        w0 = torch.max(w)
                        w1 = torch.min(w)
                        d += w0 - w1
                        w = torch.mean(w)
                        rw = torch.mean(rw)
                        if w > rw:
                            correctposinsts += 1

            if args.pooling == "max":
                # check max op correctness in different streams
                for idx in range(bag_label - 1, -1, -1):
                    ct = (label == (idx + 1))
                    if ct.any():
                        positiveinsts += 1
                        # print(f"for {idx+1} positive, and corresponding select is {label[weights[idx]]}")
                        if label[weights[idx]] == (idx + 1):
                            correctposinsts += 1
        # print(label, preds, weights)
        total += 1
    acc = correct / total
    instsacc = correctposinsts / positiveinsts
    phi = 1 - instsacc
    gamma = d / positiveinsts
    print(
        f"Epoch {epo},training info :training acc {acc:.2f},phi {phi:.2f} , gamma {gamma:.4f}")

    return acc, phi, gamma


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    bags = []
    preds = []

    if args.test == "single":
        for batch_idx, (data, label) in enumerate(test_loader):
            if args.dataset == "MNIST":
                bag_label = label[0]

                if args.cuda:
                    data, bag_label = data.cuda(), bag_label.cuda()

                data, bag_label = Variable(data), Variable(bag_label)
                # print(data.shape,bag_label.shape)  # [1,1,28,28],[]
                probs, preds, weights = model.calculate_classification_error(data, "test")
                # print(probs, preds, weights, bag_label)

            if bag_label.long() > 0:
                if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                    correct += 1
            elif bag_label.long() == 0 and torch.sum(preds) == 0:
                correct += 1
            else:
                pass
            total += 1
    else:
        for batch_idx, (data, label) in enumerate(test_loader):
            if args.dataset == "MNIST":
                data = data[0]
                bag_label = label[0][0]
                instance_labels = label[0][1:]

                if args.cuda:
                    data, bag_label = data.cuda(), bag_label.cuda()

                data, bag_label = Variable(data), Variable(bag_label)
                _, preds, _ = model.calculate_classification_error(data, args.pooling)

            if bag_label.long() > 0:
                if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                    correct += 1
            elif bag_label.long() == 0 and torch.sum(preds) == 0:
                correct += 1
            else:
                pass
            total += 1

    acc = correct / total

    print('Test Set acc:{:.4f}'.format(acc))
    return acc


if __name__ == "__main__":
    # print('Start Training freeze')

    viz = visdom.Visdom()
    # m = build_model(args.dataset, args.cuda)
    # m.load_state_dict(torch.load(f"weights{args.pooling}\\MNIST_test_0.9674_train1.00_1.00_0.0269.pth"))
    path = f"weights{args.pooling}{args.bagsize}"
    if not os.path.exists(path):
        os.makedirs(path)
    train_loader, test_loader = get_loader(args.dataset, 3)
    # optimizer = get_optimizer(m)
    accs = [0.]
    # acc = test(m, test_loader)
    # accs.append(acc)
    win = viz.line(np.arange(10))
    viz.line(accs, win=win)
    m = build_model(args.dataset, args.cuda)
    optimizer = get_optimizer(m)
    for epoch in range(args.epochs):
        acc, phi, gamma = train(m, optimizer, epoch, train_loader)
        # test_on_single_imgs()
        accs.append(test(m, test_loader))
        viz.line(accs, win=win)
        torch.save(m.state_dict(),
                   f"weights{args.pooling}{args.bagsize}\\MNIST_test_{accs[-1]:.4f}_train{acc:.2f}_{phi:.2f}_{gamma:.4f}.pth")
