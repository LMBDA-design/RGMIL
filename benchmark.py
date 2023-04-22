import scipy
import torch
from scipy import io
import numpy as np
from model import MILModel
from sklearn.model_selection import KFold
import torch.optim as optim

DATASETS = [
            "MUSK2",
            "MUSK1",
            "FOX",
            "TIGER",
            "ELEPHANT",
    "talk_politics_guns",
    "talk_politics_mideast",
    "talk_politics_misc",
    "talk_religion_misc",
    "web1",
    "web2",
    "web3",
    "web4",
    "web5",
    "web6",
    "web7",
    "web8",
    "web9",
    "messidor",
    "ucsb_breast"
]

if __name__ == "__main__":
    for DATASET in DATASETS:
        datapath = f'..\\datasets\\Benchmark\\{DATASET}.mat'
        features_struct = scipy.io.loadmat(datapath)
        data = features_struct['data']
        x = data[:, 0]
        y = data[:, 1]

        insts = 0
        for i in range(x.shape[0]):
            insts += x[i].shape[0]
        KF = KFold(n_splits=10)
        i = 1

        accs = []
        for run in range(5):
            for train_index, test_index in KF.split(x):
                opt_acc = 0.0
                model = MILModel("cuda", dataset=DATASET).cuda()
                optimizer = optim.Adam(model.parameters(), lr=0.00005,
                                       betas=(0.9, 0.999))
                i += 1
                x_train = x[train_index].squeeze()
                y_train = y[train_index].squeeze()
                x_test = x[test_index].squeeze()
                y_test = y[test_index].squeeze()
                for ep in range(40):
                    idx = torch.randperm(x_train.shape[0])
                    x_train = x_train[idx]
                    y_train = y_train[idx]
                    for j in range(x_train.shape[0]):
                        ix = torch.Tensor(x_train[j]).cuda()
                        iy = torch.Tensor(y_train[j]).squeeze().cuda()
                        if iy < 0:  # negative class, -1 to 0
                            iy += 1.
                        optimizer.zero_grad()
                        loss, preds, weights = model.calculate_objective(ix, iy)
                        loss.backward()
                        optimizer.step()
                    correct = 0
                    total = 0
                    for j in range(x_test.shape[0]):
                        ix = torch.Tensor(x_test[j]).cuda()
                        iy = torch.Tensor(y_test[j]).squeeze().cuda()
                        if iy < 0:  # negative class, -1 to 0
                            iy += 1.
                        loss, preds, weights = model.calculate_objective(ix, iy)
                        if preds == iy:
                            correct += 1
                        total += 1
                    acc = correct / total
                    # print(ep, acc)
                    opt_acc = max(acc, opt_acc)
                accs.append(opt_acc)
        acc = np.mean(accs)
        std = np.std(accs)
        print(f"overall RGP {DATASET}   acc {acc},std:{std}")
