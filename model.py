
import torch
import torch.nn as nn
from torchvision import models
import timm
import torch.nn.functional as F

F_DIM = 512


class Backbone(nn.Module):
    def __init__(self, dataset="MNIST"):
        super(Backbone, self).__init__()

        self.stem = None

        if dataset == "MNIST":
            # input [t,c,h,w]
            self.stem = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 32, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.AdaptiveAvgPool2d((4, 4))
                # [t,32,4,4]
            )

        dim_orig = 0
        if dataset == "MUSK1" or dataset == "MUSK2":
            dim_orig = 167
        elif dataset == "TIGER" or dataset == "FOX" or dataset == "ELEPHANT":
            dim_orig = 231
        elif dataset == "web1":
            dim_orig = 5864
        elif dataset == "web2":
            dim_orig = 6520
        elif dataset == "web3":
            dim_orig = 6307
        elif dataset == "web4":
            dim_orig = 6060
        elif dataset == "web5":
            dim_orig = 6408
        elif dataset == "web6":
            dim_orig = 6418
        elif dataset == "web7":
            dim_orig = 6451
        elif dataset == "web8":
            dim_orig = 6000
        elif dataset == "web9":
            dim_orig = 6280
        elif dataset == "messidor":
            dim_orig = 688
        elif dataset == "ucsb_breast":
            dim_orig = 709
        else:
            dim_orig = 201

        if dataset == "messidor":
            self.stem = nn.Sequential(
                nn.Linear(dim_orig, 2048),
                nn.ReLU(),
                # [t,32,4,4]
            )
        else:
            self.stem = nn.Sequential(
                nn.Linear(dim_orig, F_DIM),
                nn.ReLU(),
                nn.Linear(F_DIM, F_DIM),
                nn.ReLU(),
                # [t,32,4,4]
            )

    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,F_DIM]


class MILModel(nn.Module):
    def __init__(self, device, dataset="MNIST"):
        super(MILModel, self).__init__()

        self.classifiers = None
        self.channels = 100
        self.stem = Backbone(dataset)
        self.dataset = dataset
        self.linear = nn.Parameter(data=torch.FloatTensor(F_DIM, 2)).to(device)

        if dataset == "messidor":
            self.linear = nn.Parameter(data=torch.FloatTensor(2048, 2)).to(device)
        nn.init.kaiming_uniform_(self.linear)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        fs = self.stem(x)  # [t,o]

        # nsp
        bn = nn.LayerNorm(x.shape[0]).to(x.device)
        alpha = torch.mm(fs, self.linear)  # [t,ks]
        alpha = self.softmax(bn(alpha[:, 1] - alpha[:, 0]))
        F = torch.matmul(alpha, fs)  # [o]

        Y_logits = torch.matmul(F, self.linear)  # [ks]
        Y_hat = torch.argmax(Y_logits, dim=0)

        return Y_logits, Y_hat, alpha, F, alpha

    def calculate_objective(self, X, Y):
        Y0 = Y.squeeze().long()
        target = torch.zeros(2).to(X.device)
        target[Y0] = 1

        Y_logits, Y_hat, weights, feature, weight = self.forward(X)

        loss = torch.nn.CrossEntropyLoss()
        all_loss = loss(Y_logits, target)

        return all_loss, Y_hat, weight


if __name__ == "__main__":
    x = torch.randn(64, 167).cuda()
    m = MILModel(device="cuda", dataset="MUSK1").cuda()
    for p, t in m.named_parameters():
        print(p)
    y = torch.Tensor([1]).squeeze().long().cuda()

    m.calculate_objective(x, y)