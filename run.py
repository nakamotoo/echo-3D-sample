# coding: UTF-8
import os, os.path
import numpy as np
import sys
import matplotlib.pyplot as plt
import sys
import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import echonet
import torch.optim as optim
import torch.nn as nn

print("\nNEW!!", sys.argv[1], sys.argv[2])

modelname = sys.argv[1]
DestinationForWeights = "weights/r2plus1d_18_32_2_pretrained.pt"

model = torchvision.models.video.__dict__[modelname](pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.fc.bias.data[0] = 55.6

if torch.cuda.is_available() and sys.argv[2] == "pretrained":
    print("cuda is available, pretrianed weights")
    print("loading weight from ", DestinationForWeights)
    device = torch.device('cuda:1')
    model = torch.nn.DataParallel(model, device_ids=[1, 2]).cuda()
    model.to(device)
    checkpoint = torch.load(DestinationForWeights)
    model.load_state_dict(checkpoint['state_dict'])
    lr = 1e-5
elif torch.cuda.is_available() and sys.argv[2] == "random":
    print("cuda is available, random weights")
    device = torch.device('cuda:1')
    model = torch.nn.DataParallel(model, device_ids=[1, 2]).cuda()
    model.to(device)
    lr = 1e-5

criterion = nn.BCEWithLogitsLoss(reduction="none")

optimizer = optim.Adam(model.parameters(), lr=lr)
mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo_Aug(split='train'))

# ハイパーパラメータ
length = 32
period = 1
n_clips = 3

output = os.path.join("output",  "{}_{}_{}_{}_{}".format(modelname,length, period,n_clips, sys.argv[2]))
os.makedirs(output, exist_ok=True)

train_dataset = echonet.datasets.Echo_Aug(split="train", clips = n_clips, mean=mean, std=std, length=length, period=period)
val_dataset= echonet.datasets.Echo_Aug(split="valid", clips = n_clips, mean=mean, std=std, length=length, period=period)
test_dataset = echonet.datasets.Echo_Aug(split="test", clips = n_clips, mean=mean, std=std, length=length, period=period)

n_batch_train = 8
n_batch_val = 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=n_batch_train, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=n_batch_val, shuffle=True)

train_loss = []
val_loss = []

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
early_stopping = echonet.utils.EarlyStopping(patience=10, verbose=True, path=os.path.join(output, "best.pt"))

# reduce処理ありのauc
with open(os.path.join(output, "log.csv"), "a") as f:
    for epoch in range(100):
        t_true_train = []
        t_pred_train = []
        n_train = 0
        total_loss_train = 0

        model.train()

        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            n_clips = x.shape[1]
            t_true_train.extend(y.tolist())
            n_batch = x.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            x = x.reshape(-1, 3, length, 112, 112).to(device)
            y = y.reshape(-1, 1).type(torch.DoubleTensor).to(device)
            # forward + backward + optimize

            outputs = model(x)

            outputs_reduce = torch.from_numpy(np.zeros((n_batch, 1))).clone().to(device)
            for i in range(n_batch):
                mean = outputs[i*n_clips: i*n_clips+n_clips].mean()
                outputs_reduce[i] = mean

            loss = criterion(outputs_reduce, y)
            n_train += len(loss)
            total_loss_train += loss.sum().item()
            loss.mean().backward()
            optimizer.step()

            outputs_sigmoid = torch.sigmoid(outputs_reduce)
            t_pred_train.extend(outputs_sigmoid.reshape(-1).tolist())


        model.eval()
        t_true_val = []
        t_pred_val = []
        n_val = 0
        total_loss_val = 0

        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            n_clips = x.shape[1]
            t_true_val.extend(y.tolist())
            n_batch = x.shape[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            x = x.reshape(-1, 3, length, 112, 112).to(device)
            y = y.reshape(-1, 1).type(torch.DoubleTensor).to(device)
            # forward + backward + optimize

            outputs = model(x)

            outputs_reduce = torch.from_numpy(np.zeros((n_batch, 1))).clone().to(device)
            for i in range(n_batch):
                mean = outputs[i*n_clips: i*n_clips+n_clips].mean()
                outputs_reduce[i] = mean

            loss = criterion(outputs_reduce, y)
            n_val += len(loss)
            total_loss_val += loss.sum().item()

            outputs_sigmoid = torch.sigmoid(outputs_reduce)
            t_pred_val.extend(outputs_sigmoid.reshape(-1).tolist())

        train_loss.append(total_loss_train / n_train)
        val_loss.append(total_loss_val / n_val)
        scheduler.step(total_loss_val / n_val)
        early_stopping(total_loss_val / n_val, model)

        print('EPOCH: {}, Train[{:.3f}, AUC: {:.3f}], Val[{:.3f}, AUC: {:.3f}]'.format(
            epoch,
            total_loss_train / n_train,
            roc_auc_score(t_true_train, t_pred_train),
            total_loss_val / n_val,
            roc_auc_score(t_true_val, t_pred_val)
        ))

        torch.save(model.state_dict(), os.path.join(output, "latest.pt"))
        f.write("{},{},{},{},{}\n".format(
            epoch,
            total_loss_train / n_train,
            roc_auc_score(t_true_train, t_pred_train),
            total_loss_val / n_val,
            roc_auc_score(t_true_val, t_pred_val)))
        f.flush()


        if early_stopping.early_stop:
            print("Early stopped")
            break

    print('Finished Training')

# test - with reducing

checkpoint = torch.load(os.path.join(output, "best.pt"))
model.load_state_dict(checkpoint)
model.eval()

pred_test_all = []
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

for i in range(10):
    t_true_test = []
    t_pred_test = []
    n_test = 0
    total_loss_test = 0

    for i, (x, y) in enumerate(tqdm(test_dataloader)):
            n_clips = x.shape[1]
            t_true_test.extend(y.tolist())
            n_batch = x.shape[0]

            # zero the parameter gradients
            optimizer.zero_grad()
            x = x.reshape(-1, 3, length, 112, 112).to(device)
            y = y.reshape(-1, 1).type(torch.DoubleTensor).to(device)
            # forward + backward + optimize

            outputs = model(x)
            outputs_reduce = torch.from_numpy(np.zeros((n_batch, 1))).clone().to(device)
            for i in range(n_batch):
                mean = outputs[i*n_clips: i*n_clips+n_clips].mean()
                outputs_reduce[i] = mean

            loss = criterion(outputs_reduce, y)
            n_test += len(loss)
            total_loss_test += loss.sum().item()

            outputs_sigmoid = torch.sigmoid(outputs_reduce)
            t_pred_test.extend(outputs_sigmoid.reshape(-1).tolist())

    print('Test{:.3f}, AUC: {:.3f}'.format(
            total_loss_test / n_test,
            roc_auc_score(t_true_test, t_pred_test))
        )
    pred_test_all.append(t_pred_test)

pred_test_all = np.array(pred_test_all).T
pred_test_mean = pred_test_all.mean(axis = 1)
roc_auc_score(t_true_test, pred_test_mean)

plt.plot(train_loss, linewidth=3, label="train")
plt.plot(val_loss, linewidth=3, label="validation")
plt.title("Learning curve")
plt.ylabel("error")
plt.xlabel("epoch")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output, "fig1.png"))

plt.clf()

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(t_true_test, pred_test_mean)
plt.plot(fpr, tpr)
plt.title("TEST AUC : {}".format(str(roc_auc_score(t_true_test, pred_test_mean))))
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig(os.path.join(output, "fig2.png"))

np.save(os.path.join(output,"train_loss"), train_loss)
np.save(os.path.join(output,"val_loss"), val_loss)
