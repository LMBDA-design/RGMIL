import torch
import torch.nn as nn
from torchvision import models
from MMNIST.TransMIL import TransMIL
import timm


class Backbone(nn.Module):
    def __init__(self, dataset="MNIST", type="res18", out_ch=100):
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
        if dataset == "CIFAR" or dataset == "UNBC":
            if type == "res18":
                self.stem = models.resnet18(pretrained=True)
                self.stem.fc = nn.Identity()
            if type == "swin":
                self.stem = timm.models.swin_base_patch4_window7_224(pretrained=True)
                self.stem.head = nn.Sequential(
                    nn.Linear(1024, out_ch * 16),
                )

    def forward(self, x):
        x = self.stem(x)
        return x.view(x.shape[0], -1)  # [t,o*4*4]


class MILModel(nn.Module):
    def __init__(self, device, dataset="MNIST", type="res18"):
        super(MILModel, self).__init__()

        self.classifiers = None
        self.channels = 100
        self.stem = Backbone(dataset, type, out_ch=self.channels)
        self.dataset = dataset
        self.ks = 3

        # regressors
        self.linearsf = []
        for i in range(self.ks):
            setattr(self, 'l{}'.format(i), nn.Parameter(data=torch.FloatTensor(512, 2), requires_grad=True))
            self.linearsf.append(getattr(self, 'l{}'.format(i)))
        for p in self.linearsf:
            nn.init.kaiming_uniform_(p)

        self.linears = [nn.Parameter(data=torch.FloatTensor(512, 2), requires_grad=False).to(device) for i in
                        range(self.ks)]
        for p in self.linears:
            nn.init.kaiming_uniform_(p)

        # used for abp
        self.attentions = []
        for i in range(self.ks):
            setattr(self, 'a{}'.format(i), nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            ))
            self.attentions.append(getattr(self, 'a{}'.format(i)))

        # used for gab
        self.attention_V = []
        self.attention_U = []
        self.attention_weights = []
        for i in range(self.ks):
            setattr(self, 'v{}'.format(i), nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh()
            ))
            self.attention_V.append(getattr(self, 'v{}'.format(i)))
            setattr(self, 'u{}'.format(i), nn.Sequential(
                nn.Linear(512, 128),
                nn.Tanh()
            ))
            self.attention_U.append(getattr(self, 'u{}'.format(i)))
            setattr(self, 'aweight{}'.format(i), nn.Linear(128, 1))
            self.attention_weights.append(getattr(self, 'aweight{}'.format(i)))

        # used for dsp . q for query ,v for itself
        self.qs = []
        self.maxbranchfcs = []
        for i in range(self.ks):
            setattr(self, 'q{}'.format(i), nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh()
            ))
            self.qs.append(getattr(self, 'q{}'.format(i)))
            setattr(self, 'br{}'.format(i), nn.Parameter(data=torch.FloatTensor(512, 2), requires_grad=True))
            self.maxbranchfcs.append(getattr(self, 'br{}'.format(i)))
        for p in self.maxbranchfcs:
            nn.init.kaiming_uniform_(p)

        # used for tsp
        self.tsp = []
        for i in range(self.ks):
            setattr(self, 't{}'.format(i), TransMIL(n_classes=2))
            self.tsp.append(getattr(self, 't{}'.format(i)))

        self.output = nn.LogSoftmax(dim=1)  # [ks,2]

    def forward(self, x, pooling):

        fs = self.stem(x)  # [t,o]
        if pooling == "test":
            x = fs.squeeze(0)
            Y_logits = torch.stack([torch.matmul(x, self.linears[i]) for i in range(self.ks)],
                                   dim=0)  # [ks,2]
            Y_prob = self.output(Y_logits)  # [ks,2]
            Y_hat = torch.argmax(Y_prob, dim=1)

            return Y_prob, Y_hat, 0, 0
        F = fs
        alpha = torch.stack([torch.mm(fs, m) for m in self.linears], dim=0)  # [ks,t,2]

        # rgp
        if pooling == "rgp":
            bn = nn.LayerNorm(x.shape[0]).to(x.device)
            alpha = alpha[:, :, 1] - alpha[:, :, 0]  # [ks,t]
            alpha = bn(alpha)  # [ks,t]
            alpha = torch.softmax(alpha, dim=1)  # [ks,t]
            F = torch.matmul(alpha, fs)  # [ks,o]

        # max pooling
        if pooling == "max":
            alpha = torch.argmax(alpha[:, :, 1], dim=1)  # [ks] for max 1 activate
            F = torch.stack([fs[alpha[i]] for i in range(self.ks)], dim=0)  # [ks,o]

        # attention-based
        if pooling == "abp":
            alpha = torch.stack([a(fs).squeeze(1) for a in self.attentions], dim=0)  # [ks,t]
            alpha = torch.softmax(alpha, dim=1)  # softmax over t
            F = torch.mm(alpha, fs)  # [ks,o]

        # gated ab
        if pooling == "gab":
            A_V = torch.stack([V(fs) for V in self.attention_V], dim=0)  # [ks,t,128]
            A_U = torch.stack([U(fs) for U in self.attention_U], dim=0)  # [ks,t,128]
            A = A_V * A_U  # element wise multiplication # [ks,t,128]
            B = torch.stack([m(A[i]).squeeze(1) for i, m in enumerate(self.attention_weights)], dim=0)  # [ks,t]
            alpha = torch.softmax(B, dim=1)  # softmax over t  [ks,t]
            F = torch.matmul(alpha, fs)  # [ks,o]

        # one branch used for dsp,another branch aggregated outside it
        maxscore = 0.
        if pooling == "dsp":
            V = fs  # default no change [t,o]
            Q = torch.stack([q(fs).view(x.shape[0], -1) for q in self.qs], dim=0)  # [ks,t,q], unsorted
            c = torch.stack([torch.matmul(fs, self.maxbranchfcs[i]) for i in range(self.ks)], dim=0)  # [ks,t,2],
            _, indices = torch.max(c[:, :, 1], dim=1)  # [ks]
            maxscore = torch.stack([c[i, indices[i]] for i in range(self.ks)], dim=0)  # [ks,2]
            m_feats = torch.stack([fs[idx] for idx in indices],
                                  dim=0)  # select critical instances, m_feats in shape [ks,o]

            q_max = torch.stack([q(m_feats[i]) for i, q in enumerate(self.qs)],
                                dim=0)  # compute queries of critical instances, q_max in shape [ks,q]
            A = torch.bmm(Q, q_max.unsqueeze(
                -1)).squeeze(-1)  # inner product A in shape [ks,t], contains unnormalized attention scores

            alpha = nn.functional.softmax(A, dim=1)  # normalize attention scores, A in shape [ks,t]
            F = torch.mm(alpha, V)  # compute bag representation, B in shape [ks,o]

        # t-p-t module
        if pooling == "tsp":
            if len(fs.shape) < 3:
                fs = fs.unsqueeze(0)  # [1,t,o]
            Y_logits = torch.cat([TPT(fs) for i, TPT in enumerate(self.tsp)], dim=0)  # [ks,2]

        # classify on F
        if not (pooling == "tsp"):
            Y_logits = torch.stack([torch.matmul(F[i], self.linears[i]) for i in range(self.ks)],
                                   dim=0)  # [ks,2]

        # dual stream
        if pooling == "dsp":
            Y_logits = (Y_logits + maxscore) / 2

        Y_prob = self.output(Y_logits)  # [ks,2]
        Y_hat = torch.argmax(Y_prob, dim=1)

        return Y_prob, Y_hat, alpha, F

    def calculate_objective(self, X, Y, pooling):
        Y0 = Y.squeeze().long()
        if Y0 > 0:
            Y = torch.zeros(self.ks).to(X.device).to(torch.int64)
            Y[Y0 - 1] = 1
        else:
            Y = torch.zeros(self.ks).to(X.device).to(torch.int64)

        Y_prob, Y_hat, weights, F = self.forward(X, pooling)
        loss = torch.nn.NLLLoss()
        l = 0.
        if Y0 == 0:
            for i in range(self.ks):
                bce_loss = loss(Y_prob[i], Y[i])
                l += bce_loss
        else:
            for i in range(Y0 - 1, self.ks):
                bce_loss = loss(Y_prob[i], Y[i])
                l += bce_loss
        all_loss = l

        return all_loss, Y_hat, weights

    def calculate_classification_error(self, X, pooling):
        Y_prob, Y_hat, w, F = self.forward(X, pooling)
        return Y_prob, Y_hat, w


if __name__ == "__main__":
    x = torch.randn(64, 1, 28, 28).cuda()
    m = MILModel(device="cuda", dataset="MNIST").cuda()
    y = torch.Tensor([2]).long().cuda()

    m.calculate_objective(x, y, "tsp")
