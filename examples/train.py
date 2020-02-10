import sys
sys.path.append("./models")
sys.path.append("..")

import argparse

import breizhcrops
from breizhcrops.models import LSTM, TransformerEncoder, TempCNN, MSResNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import torch
import pandas as pd
import os
import sklearn.metrics

def train(args):

    padded_value = -1
    sequencelength = 45

    bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa']

    selected_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']

    selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def transform(x):
        x = x[x[:, 0] != padded_value, :] # remove padded values

        # choose selected bands
        x = x[:,selected_band_idxs] * 1e-4 # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor).to(device)

    def target_transform(y):
        y = frh01.mapping.loc[y].id
        return torch.tensor(y, dtype=torch.long, device=device)

    datapath = "../data" # "/data2/Breizhcrops"

    frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)

    gdf = frh04.geodataframe()

    frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)
    frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)
    frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)


    frh01frh02 = torch.utils.data.ConcatDataset([frh01,frh02])
    traindataloader = DataLoader(frh01frh02, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)
    valdataloader = DataLoader(frh03, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)
    testdataloader = DataLoader(frh04, batch_size=args.batchsize, shuffle=False, num_workers=args.workers)

    num_classes = len(frh01.classes)
    ndims = len(selected_bands)

    logdir=args.logdir

    device = torch.device(args.device)

    if args.model == "LSTM":
        model = LSTM(input_dim=ndims, num_classes=num_classes).to(device)
    elif args.model == "MSResNet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes).to(device)
    elif args.model == "TransformerEncoder":
        model = TransformerEncoder(input_dim=ndims, num_classes=num_classes, len_max_seq=sequencelength).to(device)
    elif args.model == "TempCNN":
        model = TempCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength).to(device)
    else:
        raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")
    print(f"Initialized {model.modelname}")

    optimizer = Adam(model.parameters(),lr=1e-2, weight_decay=1e-6)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = list()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, traindataloader)
        val_loss, y_true, y_pred = test_epoch(model, criterion, valdataloader)
        scores = metrics(y_true.cpu(), y_pred.cpu())
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k,v) in scores.items()])
        val_loss = val_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"epoch {epoch}: trainloss {train_loss:.2f}, valloss {val_loss:.2f} " + scores_msg)

        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["valloss"] = val_loss
        log.append(scores)

    log = pd.DataFrame(scores).set_index("epoch")
    log.to_csv(os.path.join(logdir,"trainlog.csv"))

    test_loss, y_true, y_pred = test_epoch(model, criterion, testdataloader)
    print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()))

def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )

def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true = batch
            loss = criterion(model.forward(x), y_true)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)

def test_epoch(model, criterion, dataloader):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader)) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x)
                y_pred = logprobabilities.argmax(-1)
                loss = criterion(logprobabilities, y_true)
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str, default="LSTM", help='select model "LSTM"')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=256, help='batch size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir for progress and models')
    args, _ = parser.parse_known_args()

    return args

if __name__=="__main__":

    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)