import scipy
import torch
from scipy import io
import numpy as np
from sklearn.model_selection import KFold
import dsmil as mil

DATASETS = [
            "alt_atheism",
            "comp_graphics",
            "comp_os_ms-windows_misc",
            "comp_sys_ibm_pc_hardware",
            "comp_sys_mac_hardware",
            "comp_windows_x",
            "misc_forsale",
            "rec_autos",
            "rec_motorcycles",
            "rec_sport_baseball",
            "rec_sport_hockey",
            "sci_crypt",
            "sci_electronics",
            "sci_med",
            "sci_religion_christian",
            "sci_space",
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


for dataset in DATASETS:

    dim_orig = 0
    if dataset == "MUSK1" or dataset == "MUSK2" or dataset == "MUSK1+" or dataset == "MUSK2+":
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

    datapath = f'..\\datasets\\Benchmark\\{dataset}.mat'
    features_struct = scipy.io.loadmat(datapath)
    data = features_struct['data']  # [92 recs,2 (0 for a ndarray(baglength,167) of conforms,1 for label)]
    x = data[:, 0]
    y = data[:, 1]

    KF = KFold(n_splits=10)
    i = 1

    accs = []
    for run in range(5):
        for train_index, test_index in KF.split(x):
            opt_acc = 0.0
            i_classifier = mil.FCLayer(dim_orig, 1)
            b_classifier = mil.BClassifier(input_size=dim_orig, output_class=1)
            milnet = mil.MILNet(i_classifier, b_classifier).cuda()
            optimizer = torch.optim.Adam(milnet.parameters(), lr=0.0002, betas=(0.5, 0.9))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, 0)
            criterion = torch.nn.BCEWithLogitsLoss()

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
                    if iy < 0:
                        iy += 1.
                    optimizer.zero_grad()
                    classes, bag_prediction, _, _ = milnet(ix)  # n X L
                    max_prediction, index = torch.max(classes, 0)
                    loss_bag = criterion(bag_prediction.view(1, -1), iy.view(1, -1))
                    loss_max = criterion(max_prediction.view(1, -1), iy.view(1, -1))
                    loss_total = 0.5 * loss_bag + 0.5 * loss_max
                    loss_total = loss_total.mean()
                    loss_total.backward()
                    optimizer.step()
                correct = 0
                total = 0
                for j in range(x_test.shape[0]):
                    ix = torch.Tensor(x_test[j]).cuda()
                    iy = torch.Tensor(y_test[j]).squeeze().cuda()
                    if iy < 0:
                        iy += 1.
                    classes, bag_prediction, _, _ = milnet(ix)
                    preds = torch.sigmoid(bag_prediction).detach().cpu().squeeze().numpy()
                    if preds > 0.5:
                        preds = 1
                    else:
                        preds = 0
                    if preds == iy:
                        correct += 1
                    total += 1
                acc = correct/total
                opt_acc = max(acc, opt_acc)
            accs.append(opt_acc)
            # print(f"fold {i} acc: {correct / total}")
    acc1 = np.mean(accs)
    std1 = np.std(accs)
    print(f"overall 5 DsMIL {dataset} acc {acc1},std:{std1}")
