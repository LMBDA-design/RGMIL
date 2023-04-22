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
from MMNIST.datasets import UNBCBagDataset
import shutil

import random
import torch.backends.cudnn as cudnn
from MMNIST.model import MILModel
from util import get_sampler, get_transforms, get_btransforms, PCC, ICC, MAE, MSE
from lr_scheduler import build_scheduler
import os

import visdom
# print(imgs.shape, labels.shape)

noise = np.random.randn(64, 3, 64, 64)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
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
parser.add_argument('--mode', type=str, default='MIL', help='Choose MILModel')
parser.add_argument('--pretrained', type=bool, default='0', help='whether load and freeze')
parser.add_argument('--dataset', type=str, default='UNBC', help='Choose dataset')
parser.add_argument('--model', type=str, default='res18', help='Choose model type')
parser.add_argument('--test', type=str, default='single', help='Choose test mode')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def build_model(mode, model, dataset, cuda):
    print('Init Model')
    if cuda:
        device = "cuda"
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

    if mode == 'MIL':
        m = MILModel(device, dataset=dataset)

    if cuda:
        m.cuda()
    return m


def get_loader(dataset, mode, target_number=0):
    loader_kwargs = {'num_workers': 0, 'pin_memory': True}
    if mode == "MIL":
        if dataset == "MNIST":
            train_loader = data_utils.DataLoader(MnistBagDataset(target_number=target_number,
                                                                 train=True),
                                                 batch_size=1,
                                                 shuffle=True,
                                                 **loader_kwargs)

            test_loader = data_utils.DataLoader(MnistBagDataset(target_number=target_number,
                                                                train=False, mode=args.test),
                                                batch_size=1,
                                                shuffle=False,
                                                **loader_kwargs)
            return train_loader, test_loader
        if dataset == "UNBC":
            train_trans = get_btransforms(train=True)
            train_dataset = UNBCBagDataset(target_number=target_number, train=True, transform=train_trans)
            sampler = get_sampler(train_dataset)
            train_loader = data_utils.DataLoader(
                train_dataset, batch_size=1,
                sampler=sampler,
                # shuffle=True,
                **loader_kwargs)
            test_trans = get_btransforms(train=False)
            test_loader = data_utils.DataLoader(
                UNBCBagDataset(target_number=target_number, train=False, transform=test_trans), batch_size=1,
                shuffle=False,
                **loader_kwargs)
            return train_loader, test_loader

    else:
        if dataset == "MNIST":
            train_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))])),
                                                 batch_size=64,
                                                 shuffle=True)

            test_loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                               train=False,
                                                               download=True,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=64,
                                                shuffle=False)
            return train_loader, test_loader

        if dataset == "UNBC":
            data_transform_train = get_transforms(train=True)
            data_transform_test = get_transforms(train=False)

            data_dir = ".\\datasets\\UNBC\\train"

            train_dataset = datasets.ImageFolder(data_dir, data_transform_train)
            sampler = get_sampler(train_dataset)

            data_dir = ".\\datasets\\UNBC\\val"

            test_dataset = datasets.ImageFolder(data_dir, data_transform_test)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=sampler, shuffle=False,
                                                       num_workers=4)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            return train_loader, test_loader

        if args.dataset == "CIFAR":
            IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
            data_transforms = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10("../datasets", train=True, transform=data_transforms,
                                 download=True), batch_size=64, shuffle=True,
                num_workers=4)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10("../datasets", train=False, transform=data_transforms,
                                 download=True), batch_size=64, shuffle=True,
                num_workers=4)
            return train_loader, test_loader


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), weight_decay=args.reg)
    if args.model == "res-swin":
        optimizer = optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=5.5e-4, weight_decay=0.0005)
    # optimizer = optim.Adam([
    #     {'params': model.feature_extractor_part1.parameters(), 'lr': args.lr * .001},
    #     {'params': model.feature_extractor_part2.parameters(), 'lr': args.lr * .001},
    #     {'params': model.attention.parameters(), }, {'params': model.classifier.parameters()}], lr=args.lr,
    #     betas=(0.9, 0.999), weight_decay=args.reg)

    # print(optimizer.state_dict())
    return optimizer


def train(model, optimizer, scheduler, epo, train_loader):
    model.train()
    num_steps = len(train_loader)
    train_loss = 0.
    correct = 0
    correctposinsts = 0
    positiveinsts = 0
    d = 0.0
    total = 0
    acc = 0.0
    instsacc = 0.0
    diff = 0.0

    if args.mode == "MIL":
        for batch_idx, (data, label) in enumerate(train_loader):
            if args.dataset == "MNIST":
                data = data[0]
                bag_label = label[0][0]
                label = label[0][1:]
            if args.dataset == "UNBC":
                data = data[0]
                label = label[0]
                bag_label = torch.max(label)

            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            # print("train shape",data.shape,bag_label.shape)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, preds, weights = model.calculate_objective(data, bag_label)
            # print(f"fp time {fp - start_time}")
            train_loss += loss.data
            # backward pass
            loss.backward()
            # print(f"loss{loss.detach().cpu().numpy()}")
            # for name, parms in model.named_parameters():
            #     if parms.requires_grad:
            #         print(name,'-->grad_value mu,theta:',torch.mean(parms.grad),torch.var(parms.grad))
            # step
            # print(preds,bag_label)
            optimizer.step()
            # print(preds,bag_label)
            if bag_label.long() > 0:
                if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                    correct += 1
            elif bag_label.long() == 0 and torch.sum(preds) == 0:
                correct += 1
            else:
                pass
            label = label.cpu()
            bag_label = bag_label.long().cpu()
            total += 1
            scheduler.step_update(epo * num_steps + batch_idx)
            # print(f"one batch time {et-start_time},bp {et-fp}")
        acc = 0
        instsacc = 0
        diff = 0
        train_loss /= 0
        print(
            f"Epoch {epo},training info :training acc {acc:.2f},instsacc {instsacc:.2f} , diff {diff:.4f},loss {train_loss:.4f}")
    else:
        for batch_idx, (data, label) in enumerate(train_loader):
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            # 　print(data.shape,label.shape)
            # shape:x[batchsize,bagesize,channel,h,w]:[1,10,1,28,28]
            # shape:[1]
            data, label = Variable(data), Variable(label)
            # viz.images(data,nrow=8)
            # reset gradients
            optimizer.zero_grad()
            # calculate loss and metrics
            loss, preds = model.calculate_objective(data, label)
            # print(preds)
            train_loss += loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            scheduler.step_update(epo * num_steps + batch_idx)

    return acc, instsacc, diff


def test(model, test_loader, save_imgs=False):
    model.eval()
    test_loss = 0.
    test_error = 0.
    correct = 0
    total = 0
    bags = []
    preds = []
    r = 0
    counts = 1
    counter = 0
    correct_peak = 0
    all_preds = torch.FloatTensor()
    if args.mode == "MIL":
        if args.test == "single":
            outs = []
            targets = []
            for batch_idx, (data, label) in enumerate(test_loader):
                if args.dataset == "MNIST":
                    bag_label = label[0]

                    if args.cuda:
                        data, bag_label = data.cuda(), bag_label.cuda()

                    data, bag_label = Variable(data), Variable(bag_label)
                    # print(data.shape,bag_label.shape)  # [1,1,28,28],[]
                    _, preds, _ = model.calculate_classification_error(data, bag_label)

                if args.dataset == "UNBC":
                    data = data[0]
                    if not label.shape[0] == 1:
                        label = label[0].squeeze()
                    bag_label = torch.max(label)

                    if args.cuda:
                        data, bag_label = data.cuda(), bag_label.cuda()

                    data, bag_label = Variable(data), Variable(bag_label)
                    # print(data.shape,bag_label.shape)
                    # print("bag shape ",data.shape,label.shape)
                    bs, ncrops, c, h, w = data.shape
                    predicted_bag_label = []
                    for i in range(ncrops):
                        data_slice = data[:, i, :, :, :]
                        with torch.no_grad():
                            # print("slice",i)
                            bag_slice_predict, _, _ = model.calculate_classification_error(data_slice, bag_label)
                        predicted_bag_label.append(bag_slice_predict)
                        # (predicted_bag_label,bag_label)
                    predicted_bag_label = torch.stack(predicted_bag_label).mean(0)
                    preds = torch.argmax(predicted_bag_label, dim=1)
                    #  print(preds,bag_label)
                if bag_label.long() > 0:
                    if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                        correct += 1
                elif bag_label.long() == 0 and torch.sum(preds) == 0:
                    correct += 1
                else:
                    pass
                total += 1
                # high position
                for i in range(preds.shape[0] - 1, -1, -1):
                    if preds[i] == 1:
                        outs.append(i + 1)
                        break
                    if i == 0:
                        outs.append(0)
                targets.append(bag_label.cpu())

            outs = np.array(outs)
            targets = np.array(targets)
            outs = torch.from_numpy(outs)
            targets = torch.from_numpy(targets)
            icc = ICC(outs, targets)
            pcc = PCC(outs, targets)
            mae = MAE(outs, targets)
            mse = MSE(outs, targets)
        else:
            for batch_idx, (data, label) in enumerate(test_loader):
                if args.dataset == "MNIST":
                    data = data[0]
                    bag_label = label[0][0]
                    instance_labels = label[0][1:]
                if args.dataset == "UNBC":
                    data = data[0]
                    if not label.shape[0] == 1:
                        label = label[0].squeeze()
                    bag_label = torch.max(label)

                if args.cuda:
                    data, bag_label = data.cuda(), bag_label.cuda()

                data, bag_label = Variable(data), Variable(bag_label)
                # print(data.shape,bag_label.shape)
                # print("bag shape ",data.shape,label.shape)
                bs, ncrops, c, h, w = data.shape
                predicted_bag_label = []
                for i in range(ncrops):
                    data_slice = data[:, i, :, :, :]
                    with torch.no_grad():
                        # print("slice",i)
                        bag_slice_predict, _, _ = model.calculate_classification_error(data_slice, bag_label)
                    predicted_bag_label.append(bag_slice_predict)
                # (predicted_bag_label,bag_label)
                predicted_bag_label = torch.stack(predicted_bag_label).mean(0)
                preds = torch.argmax(predicted_bag_label, dim=1)
                # print(preds,bag_label)
                if bag_label.long() > 0:
                    if preds[bag_label.long() - 1] == 1 and torch.sum(preds[bag_label.long():]) == 0:
                        correct += 1
                elif bag_label.long() == 0 and torch.sum(preds) == 0:
                    correct += 1
                else:
                    pass
                total += 1

        # print(f'bags {bags},preds {preds}')
        # print("r value:", r / counts)
    else:
        for batch_idx, (data, label) in enumerate(test_loader):
            bs, ncrops, c, h, w = data.shape
            data = data.view(-1, c, h, w)
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                data, label = Variable(data), Variable(label)
                _, preds = model.calculate_objective(data, label, train=False)
            outputs_avg = preds.view(bs, ncrops, -1).mean(1)  # avg over crops
            # viz.images(data, win=win)
            # time.sleep(10)
            # print(label)
            preds = outputs_avg.detach().cpu()
            all_preds = torch.cat([all_preds, preds], dim=0)

        print(all_preds[:50], all_preds[500:550])
        all_preds = torch.argmax(all_preds, dim=1)
        preds0 = torch.zeros_like(all_preds)
        check = preds0 == all_preds
        print(torch.sum(check))
        test_dataset = test_loader.dataset
        right_path = './class_result/right'
        wrong_path = './class_result/wrong'
        for i, pred in enumerate(all_preds):
            y = test_dataset.imgs[i][1]
            if y > 0:
                y = 1
            if y == pred:
                if save_imgs:
                    if not os.path.exists(right_path):
                        os.makedirs(right_path)
                    shutil.copyfile(test_dataset.imgs[i][0], os.path.join(right_path, str(
                        all_preds[i].cpu().numpy()) + '_' + str(
                        test_dataset.imgs[i][1]) + os.path.basename(test_dataset.imgs[i][0])))
                else:
                    correct += 1
                    total += 1
            else:
                if save_imgs:
                    if not os.path.exists(wrong_path):
                        os.makedirs(wrong_path)
                    shutil.copyfile(test_dataset.imgs[i][0],
                                    os.path.join(wrong_path, str(
                                        all_preds[i].cpu().numpy()) + '_' + str(
                                        test_dataset.imgs[i][1]) + os.path.basename(test_dataset.imgs[i][0])))
                else:
                    total += 1
        if save_imgs:
            correct = len(os.listdir(right_path))
            total = len(os.listdir(right_path)) + len(os.listdir(wrong_path))

    acc = correct / total

    print(f'Test Set acc:{acc:.4f},pcc:{pcc:.4f},icc:{icc:.4f},mse:{mse:.4f},mae:{mae:.4f}')
    return acc, pcc, icc, mse, mae


if __name__ == "__main__":
    # print('Start Training freeze')

    viz = visdom.Visdom()
    for i in range(25):
        m = build_model(args.mode, args.model, args.dataset, args.cuda)
        train_loader, test_loader = get_loader(args.dataset, args.mode, i)
        optimizer = get_optimizer(m)
        lr_scheduler = build_scheduler(optimizer, len(train_loader))
        accs = []
        acc, pcc, icc, mse, mae = test(m, test_loader)
        accs.append(acc)
        win = viz.line(np.arange(10))
        viz.line(accs, win=win)
        print("testing", i)
        for epoch in range(args.epochs):
            acc, instsacc, diff = train(m, optimizer, lr_scheduler, epoch, train_loader)
            # test_on_single_imgs()
            acc, pcc, icc, mse, mae = test(m, test_loader)
            accs.append(acc)
            viz.line(accs, win=win)
            torch.save(m.state_dict(),
                       f"weights4\\{i}_{accs[-1]:.4f}_{pcc:.4f}_{icc:.4f}_{mse:.4f}_{mae:.4f}_train{acc:.2f}_{instsacc:.2f}_{diff:.4f}.pth")
