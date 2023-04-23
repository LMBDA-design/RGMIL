import numpy as np

import torch
import torch.utils.data as data_utils
from MMNIST.datasets import MnistBags
import visdom
import time
import os

loader_kwargs = {'num_workers': 4, 'pin_memory': True}
GLOBAL_SEED = 1
BAG_LENGTH = 64
NUM_BAGS = 10000
MAX_NUM = 9
ds = {
    "MNIST": MnistBags
}


class MnistBagDataset(data_utils.Dataset):
    def __init__(self, target_number=1, training_bags=4000, train=True, mode="single"):
        self.target_number = target_number
        self.train = train
        self.mode = mode
        self.training_bags = training_bags
        self.imgs, self.labels = self._create_bags()
        # print(self.imgs.shape, self.labels.shape)

    def _create_bags(self):
        datas = torch.Tensor()
        labels = torch.Tensor()
        if self.train:
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\train\\{self.target_number}_x_.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\train\\{self.target_number}_y_.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)
            datas = datas[:self.training_bags]
            labels = labels[:self.training_bags]
        elif self.mode == "single":  # for single img tests
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_x.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_y.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            datas = datas.view(-1, 1, 28, 28)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)[:, 1:]
            labels = labels.reshape(-1)
            datas = datas[:10000]
            labels = labels[:10000]
        else:  # for bag tests
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_x_.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_y_.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)
            datas = datas[:1000]
            labels = labels[:1000]
        return datas, labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


class UNBCBagDataset(data_utils.Dataset):
    def __init__(self, target_number=1, train=True, mode="single"):
        self.target_number = target_number
        self.train = train
        self.mode = mode
        self.imgs, self.labels = self._create_bags()
        # print(self.imgs.shape, self.labels.shape)

    def _create_bags(self):
        datas = torch.Tensor()
        labels = torch.Tensor()
        if self.train:
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\train\\{self.target_number}_x_.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\train\\{self.target_number}_y_.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)
        elif self.mode == "single":  # for single img tests
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_x.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_y.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            datas = datas.view(-1, 1, 28, 28)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)[:, 1:]
            labels = labels.reshape(-1)
            datas = datas[:10000]
            labels = labels[:10000]
        else:  # for bag tests
            pathx = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_x_.npy"
            pathy = f"datasets\\MNIST_{BAG_LENGTH}\\test\\{self.target_number}_y_.npy"
            data = torch.Tensor(np.load(pathx))
            datas = torch.cat([datas, data], dim=0)
            label = torch.Tensor(np.load(pathy))
            labels = torch.cat([labels, label], dim=0)
            datas = datas[:1000]
            labels = labels[:1000]
        return datas, labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]


def get_loader(databag, train=True, target_number=1, num_bag=20000, bag_length=64):
    loader = data_utils.DataLoader(databag(target_number=target_number,
                                           mean_bag_length=bag_length,
                                           num_bag=num_bag,
                                           seed=GLOBAL_SEED,
                                           train=train),
                                   batch_size=1,
                                   **loader_kwargs)
    return loader


def makedataset(dataset="MNIST", max_num=3, num_bag=10000, bag_length=64):
    for i in range(max_num):
        print(f"making {i}")
        j = i + 1

        if j == 1:
            train_loader = get_loader(ds[dataset], train=True, target_number=j, num_bag=num_bag * 2,
                                      bag_length=bag_length)
            test_loader = get_loader(ds[dataset], train=False, target_number=j, num_bag=num_bag * 2 // 10,
                                     bag_length=bag_length)
            datas = torch.Tensor()
            labels = torch.Tensor()

            for batch_idx, (data, label) in enumerate(train_loader):
                bag_label = label[0].unsqueeze(1)
                instance_labels = label[1]

                lab = torch.cat([bag_label, instance_labels], axis=1)

                # [b,1],[b,t,1,28,28],[b,t],[b,t+1] (b=1)
                # print(bag_label.shape, data.shape, instance_labels.shape, lab.shape)
                datas = torch.cat([datas, data], axis=0)
                labels = torch.cat([labels, lab], axis=0)

            print(datas.shape, labels.shape)
            label_idx = labels[:, 0] > 0

            datas_1 = datas[label_idx]
            labels_1 = labels[label_idx]
            path = f"datasets\\{dataset}_{bag_length}\\train"
            if not os.path.exists(path):
                os.makedirs(path)

            print("saving ... ")
            np.save(f"{path}\\{j}_x.npy", datas_1)
            np.save(f"{path}\\{j}_y.npy", labels_1)

            label_idx = ~label_idx
            datas_0 = datas[label_idx]
            labels_0 = labels[label_idx]

            np.save(f"{path}\\{j - 1}_x.npy", datas_0)
            np.save(f"{path}\\{j - 1}_y.npy", labels_0)

            datas = torch.Tensor()
            labels = torch.Tensor()
            for batch_idx, (data, label) in enumerate(test_loader):
                bag_label = label[0].unsqueeze(1)
                instance_labels = label[1]

                lab = torch.cat([bag_label, instance_labels], axis=1)

                # [b,1],[b,t,1,28,28],[b,t],[b,t+1] (b=1)
                # print(bag_label.shape, data.shape, instance_labels.shape, lab.shape)
                datas = torch.cat([datas, data], axis=0)
                labels = torch.cat([labels, lab], axis=0)

            # print(datas.shape,labels.shape)
            label_idx = labels[:, 0] > 0

            datas_1 = datas[label_idx]
            labels_1 = labels[label_idx]
            path = f"datasets\\{dataset}_{bag_length}\\test"
            if not os.path.exists(path):
                os.makedirs(path)

            np.save(f"{path}\\{j}_x.npy", datas_1)
            np.save(f"{path}\\{j}_y.npy", labels_1)

            label_idx = ~label_idx
            datas_0 = datas[label_idx]
            labels_0 = labels[label_idx]

            np.save(f"{path}\\{j - 1}_x.npy", datas_0)
            np.save(f"{path}\\{j - 1}_y.npy", labels_0)

        else:
            train_loader = get_loader(ds[dataset], train=True, target_number=j, num_bag=num_bag * 2,
                                      bag_length=bag_length)
            test_loader = get_loader(ds[dataset], train=False, target_number=j, num_bag=num_bag * 2 // 10,
                                     bag_length=bag_length)
            datas = torch.Tensor()
            labels = torch.Tensor()

            for batch_idx, (data, label) in enumerate(train_loader):
                bag_label = label[0].unsqueeze(1)
                instance_labels = label[1]

                lab = torch.cat([bag_label, instance_labels], axis=1)

                # [b,1],[b,t,1,28,28],[b,t],[b,t+1] (b=1)
                # print(bag_label.shape, data.shape, instance_labels.shape, lab.shape)
                datas = torch.cat([datas, data], axis=0)
                labels = torch.cat([labels, lab], axis=0)

            # print(datas.shape,labels.shape)
            label_idx = labels[:, 0] == j
            datas_1 = datas[label_idx]
            labels_1 = labels[label_idx]
            print("train ", datas_1.shape, labels_1.shape)
            np.save(f"datasets\\MNIST_{bag_length}\\train\\{j}_x.npy", datas_1)
            np.save(f"datasets\\MNIST_{bag_length}\\train\\{j}_y.npy", labels_1)

            datas = torch.Tensor()
            labels = torch.Tensor()
            for batch_idx, (data, label) in enumerate(test_loader):
                bag_label = label[0].unsqueeze(1)
                instance_labels = label[1]

                lab = torch.cat([bag_label, instance_labels], axis=1)

                # [b,1],[b,t,1,28,28],[b,t],[b,t+1] (b=1)
                # print(bag_label.shape, data.shape, instance_labels.shape, lab.shape)
                datas = torch.cat([datas, data], axis=0)
                labels = torch.cat([labels, lab], axis=0)

            # print(datas.shape,labels.shape)
            label_idx = labels[:, 0] == j

            datas_1 = datas[label_idx]
            labels_1 = labels[label_idx]
            print("test ", datas_1.shape, labels_1.shape)
            np.save(f"datasets\\MNIST_{bag_length}\\test\\{j}_x.npy", datas_1)
            np.save(f"datasets\\MNIST_{bag_length}\\test\\{j}_y.npy", labels_1)


def checkdataset():
    imgs = np.load(f"datasets\\MNIST_64\\test\\3_x_.npy")
    labels = np.load(f"datasets\\MNIST_64\\test\\3_y_.npy")

    viz = visdom.Visdom()
    print(imgs.shape, labels.shape)
    label_counts = np.bincount(labels[:, 0].astype(int))
    for i, count in enumerate(label_counts):
        print(f'Label {i}: {count} samples')
    
    noise = np.random.randn(64, 1, 28, 28)
    win = viz.images(noise, 8)
    for i in range(imgs.shape[0]):
        viz.images(imgs[i], 8, win=win, )
        print(labels[i])
        time.sleep(10)


def mixup_test(target_num):
    datas = torch.Tensor()
    labels = torch.Tensor()
    for i in range(target_num):
        img = torch.Tensor(np.load(f"datasets\\MNIST_64\\test\\{i}_x.npy"))
        label = torch.Tensor(np.load(f"datasets\\MNIST_64\\test\\{i}_y.npy"))
        datas = torch.cat([datas, img], dim=0)
        labels = torch.cat([labels, label], dim=0)

    for i in range(20):
        idx = torch.randperm(labels.shape[0])
        datas = datas[idx, :, :, :]
        labels = labels[idx]

    np.save(f"datasets\\MNIST_64\\test\\{target_num}_x_.npy", datas)
    np.save(f"datasets\\MNIST_64\\test\\{target_num}_y_.npy", labels)


def mixup_train(target_num):
    datas = torch.Tensor()
    labels = torch.Tensor()
    for i in range(target_num):
        img = torch.Tensor(np.load(f"datasets\\MNIST_64\\train\\{i}_x.npy"))
        label = torch.Tensor(np.load(f"datasets\\MNIST_64\\train\\{i}_y.npy"))
        datas = torch.cat([datas, img], dim=0)
        labels = torch.cat([labels, label], dim=0)

    for i in range(20):
        idx = torch.randperm(labels.shape[0])
        datas = datas[idx, :, :, :]
        labels = labels[idx]

    np.save(f"datasets\\MNIST_64\\train\\{target_num}_x_.npy", datas)
    np.save(f"datasets\\MNIST_64\\train\\{target_num}_y_.npy", labels)


if __name__ == "__main__":
    os.chdir("..")
    makedataset()
    mixup_train(4)
    mixup_test(4)

