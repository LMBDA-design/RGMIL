import time

import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from util import get_transforms
import os
import shutil
import visdom
from util import get_transforms, get_btransforms
from PIL import Image


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))
                                                          ])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            # print(batch_labels.shape, batch_data.shape)
            all_imgs = batch_data
            all_labels = batch_labels

            idx = batch_labels <= self.target_number
            all_imgs = all_imgs[idx]
            all_labels = all_labels[idx]

            self.num_in_train = all_imgs.shape[0]
            self.num_in_test = all_labels.shape[0]

        bags_list = []
        labels_list = []
        Ts = 0
        Fs = 0

        # Num_bags个包，每个包mean_bag_length
        for i in range(self.num_bag):
            bag_length = int(self.mean_bag_length)
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            imgs_in_bag = all_imgs[indices]
            instance_labels = labels_in_bag
            labels_in_bag = labels_in_bag == self.target_number

            # make sure .5 ,.5,balanced data
            if Ts > Fs:
                nontargets_in_bag = labels_in_bag == 0
                bag_imgs = imgs_in_bag[nontargets_in_bag]

                if bag_imgs.shape[0] > 0:
                    # 被抽出了，不足baglength了，补足负样本
                    instance_labels = instance_labels[nontargets_in_bag]
                    count = nontargets_in_bag.shape[0] - torch.sum(nontargets_in_bag)
                    while count > 0:
                        indice = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        while all_labels[indice] == self.target_number:
                            indice = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        img = all_imgs[indice]
                        label = all_labels[indice]
                        bag_imgs = torch.cat([bag_imgs, img], axis=0)
                        instance_labels = torch.cat([instance_labels, label], axis=0)
                        count -= 1

                    bags_list.append(bag_imgs)
                    labels_list.append(instance_labels)
                    Fs += 1
            else:
                bags_list.append(imgs_in_bag)
                if torch.sum(labels_in_bag) > 0:
                    Ts += 1
                else:
                    Fs += 1
                labels_list.append(instance_labels)

        # labels_list 指每个bag内target的对应T/F数组
        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            bag_label = torch.max(self.train_labels_list[index])
            label = [bag_label, self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            bag_label = torch.max(self.test_labels_list[index])
            label = [bag_label, self.test_labels_list[index]]

        # bag:[bag,h,w]; label:list of 2,list[0] is the P/N ，list[1] is the TF array of shape [bag]
        return bag, label


class UNBCBagDataset(data_utils.Dataset):
    def __init__(self, target_number=0, mean_bag_length=64, seed=1, slide=8, mode="single", train=True, transform=None):
        super().__init__()
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.slide = slide
        self.train = train
        self.train_len = 0
        self.test_len = 0
        self.transforms = transform
        self.mode = mode

        self.r = np.random.RandomState(seed)

        if not os.path.exists(os.path.join(f"datasets\\UNBC\\{target_number}", 'train')):
            for i in range(25):
                self.__split_dataset_UNBC_4(
                    "D:\\study\\codes\\Attention\\datasets\\UNBC-McMaster\\Frame_Labels\\Frame_Labels\\PSPI",
                    "D:\\study\\codes\\Attention\\datasets\\UNBC-McMaster\\Images",
                    "datasets\\UNBC" + os.sep + str(i),
                    i
                )

        if os.path.exists(os.path.join("datasets\\UNBC\\", str(target_number))):
            if self.train:
                self.train_imgs = torch.Tensor(np.load(
                    os.path.join("datasets\\UNBC\\" + str(target_number), 'train', "imgs.npy")))
                print(f"train shape {self.train_imgs.shape}")
                self.train_labels = torch.Tensor(np.load(
                    os.path.join("datasets\\UNBC\\" + str(target_number), 'train', "labels.npy")))
                print(f"train shape {self.train_labels.shape}")
                self.train_len = self.train_labels.shape[0]
                self.train_len = (self.train_len - self.mean_bag_length) // self.slide - 2
                print(f"trainlen {self.train_len}")
                self.targets = self.train_labels
            else:
                self.test_imgs = torch.Tensor(
                    np.load(os.path.join("datasets\\UNBC\\" + str(target_number), 'test', "imgs.npy")))
                print(f"test shape {self.test_imgs.shape}")
                self.test_labels = torch.Tensor(np.load(
                    os.path.join("datasets\\UNBC\\" + str(target_number), 'test', "labels.npy")))
                print(f"test shape {self.test_labels.shape}")
                self.test_len = self.test_labels.shape[0]
                if self.mode == "MIL":
                    self.test_len = (self.test_labels.shape[0] - self.mean_bag_length) // self.slide - 2
                else:
                    self.slide = 1
                    self.test_len = (self.test_labels.shape[0] - 1) // self.slide - 2
                print(f"terstlen {self.test_len}")
                self.targets = self.test_labels

    def __split_dataset_UNBC_4(self, label_path, path, dest_path, which_id):
        pain_dict = {'0': '0', '1': '1', '2': '1', '3': '2', '4': '2', '5': '2', '6': '3', '7': '3', '8': '3', '9': '3',
                     '10': '3', '11': '3', '12': '3', '13': '3', '14': '3', '15': '3', '16': '3'}
        if not os.path.exists(os.path.join(dest_path, 'train')):
            os.makedirs(os.path.join(dest_path, 'train'))
        else:
            if self.train:
                self.train_labels = np.load(os.path.join(dest_path, 'train', "labels.npy"))
                self.train_len = self.train_labels.shape[0]
                self.train_len = self.train_len - self.mean_bag_length + 1

        if not os.path.exists(os.path.join(dest_path, 'test')):
            os.makedirs(os.path.join(dest_path, 'test'))
        else:
            if not self.train:
                self.test_labels = np.load(os.path.join(dest_path, 'test', "labels.npy"))
                self.test_len = self.test_labels.shape[0]
                self.test_len = self.test_len - self.mean_bag_length + 1

        train_imgs = []
        train_labels = []
        test_imgs = []
        test_labels = []
        id_list = sorted(os.listdir(label_path))
        for index, ids in enumerate(id_list):
            for seq in sorted(os.listdir(os.path.join(label_path, ids))):
                for txt in sorted(os.listdir(os.path.join(label_path, ids, seq))):
                    img = txt.split('_')[0] + '.png'
                    with open(os.path.join(label_path, ids, seq, txt), 'r') as f:
                        score = float(f.readlines()[0][2:-1])
                    score = float(pain_dict[str(int(score))])
                    if which_id == index:
                        shutil.copyfile(os.path.join(path, ids, seq, img), os.path.join(dest_path, 'test', img))
                        im = Image.open(os.path.join(dest_path, 'test', img)).resize((224, 224))
                        tu = torch.Tensor(np.array(im)).permute(2, 0, 1)
                        test_imgs.append(tu)
                        test_labels.append(score)
                        # tu = cv2.resize(tu, (224, 224))
                        # cv2.imwrite(os.path.join(dest_path, 'val', img), tu)
                    else:
                        shutil.copyfile(os.path.join(path, ids, seq, img), os.path.join(dest_path, 'train', img))
                        im = Image.open(os.path.join(dest_path, 'train', img)).resize((224, 224))
                        tu = torch.Tensor(np.array(im)).permute(2, 0, 1)
                        train_imgs.append(tu)
                        train_labels.append(score)
                        # tu = cv2.imread(os.path.join(dest_path, 'train', score, img))
                        # tu = cv2.resize(tu, (224, 224))
                        # print(os.path.join(dest_path, 'val', score, img))
                        # cv2.imwrite(os.path.join(dest_path, 'train', score, img), tu)

        self.train_imgs = torch.stack(train_imgs)
        self.train_labels = torch.Tensor(train_labels)
        self.test_imgs = torch.stack(test_imgs)
        self.test_labels = torch.Tensor(test_labels)
        self.train_len = self.train_labels.shape[0]
        self.test_len = self.test_labels.shape[0]
        self.train_len = (self.train_len - self.mean_bag_length) // self.slide - 2
        if self.mode == "MIL":
            self.test_len = (self.test_len - self.mean_bag_length) // self.slide - 2
        else:
            self.test_len = (self.test_len - 1) // self.slide - 2

        np.save(os.path.join(dest_path, 'train', "imgs.npy"), self.train_imgs.numpy())
        np.save(os.path.join(dest_path, 'train', "labels.npy"), self.train_labels.numpy())
        np.save(os.path.join(dest_path, 'test', "imgs.npy"), self.test_imgs.numpy())
        np.save(os.path.join(dest_path, 'test', "labels.npy"), self.test_labels.numpy())

    def __getitem__(self, index):
        index = index * self.slide
        if self.train:
            imgs = self.train_imgs[index:index + self.mean_bag_length]
            labels = self.train_labels[index:index + self.mean_bag_length]
            return self.transforms(imgs), labels
        else:
            if self.mode == "MIL":
                imgs = self.test_imgs[index:index + self.mean_bag_length]
                labels = self.test_labels[index:index + self.mean_bag_length]
            else:
                imgs = self.test_imgs[index:index + 1]
                labels = self.test_labels[index:index + 1]
            return self.transforms(imgs), labels

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len


class TmpDataset(data_utils.Dataset):
    def __init__(self, bag, label, transform=None):
        self.imgs = bag
        self.labels = label
        self.transform = transform

    def __getitem__(self, item):
        img = self.imgs[item]
        if not self.transform is None:
            img = self.imgs[item].permute(1, 2, 0)
            img = Image.fromarray(np.uint8(img.numpy()), mode="RGB")
            img = self.transform(img)
        img = torch.Tensor(np.array(img))
        return img, self.labels[item]

    def setData(self, bag, label):
        self.imgs = bag
        self.labels = label

    def __len__(self):
        return self.labels.shape[0]


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=3,
                                                   mean_bag_length=10,
                                                   num_bag=1000,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)
    for batch_idx, (bag, label) in enumerate(train_loader):
        # print(f"bag shape{bag[0].shape} {label[0].shape} {label[1].shape}")
        print(label[0], label[1])

    os.chdir('..')
    trans = get_btransforms(True)
    train_loader = data_utils.DataLoader(UNBCBagDataset(transform=trans), batch_size=1, shuffle=False)
    viz = visdom.Visdom()
    noise = torch.rand(4, 3, 224, 224)
    win = viz.images(noise)
    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        bag = bag[0]
        label = label[0]
        len_bag_list_train.append(int(bag.size()[0]))
        mnist_bags_train += torch.max(label)
        # trans = get_transforms(train=True)
        # transformed = data_utils.DataLoader(TmpDataset(bag, label, transform=trans), batch_size=bag.shape[0])
        # transformed_imgs = next(iter(transformed))[0]
        transformed_imgs = bag
        viz.images(transformed_imgs, win=win)
        time.sleep(10)
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))
