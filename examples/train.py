import sys

#sys.path.append("./models")
sys.path.append("..")

import argparse

import breizhcrops
from breizhcrops.models import LSTM, TransformerModel, TempCNN, MSResNet, InceptionTime, StarRNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import torch
import pandas as pd
import os
import sklearn.metrics


def train(args):
    traindataloader, testdataloader, meta = get_dataloader(args.datapath, args.mode, args.batchsize, args.workers)

    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    print(f"Logging results to {args.logdir}")
    logdir = args.logdir

    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.modelname += f"_learning-rate={args.learning_rate}_weight-decay={args.weight_decay}"
    print(f"Initialized {model.modelname}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = list()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)
        test_loss, y_true, y_pred = test_epoch(model, criterion, testdataloader, device)
        scores = metrics(y_true.cpu(), y_pred.cpu())
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
        test_loss = test_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)

        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = test_loss
        log.append(scores)

    log = pd.DataFrame(log).set_index("epoch")
    log.to_csv(os.path.join(logdir, "trainlog.csv"))

    test_loss, y_true, y_pred = test_epoch(model, criterion, testdataloader, device)
    print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()))


def get_dataloader(datapath, mode, batchsize, workers):
    print(f"Setting up datasets in {os.path.abspath(datapath)}")
    datapath = os.path.abspath(datapath)

    padded_value = -1
    sequencelength = 45

    bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
             'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa']
    selected_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def transform(x):
        x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor)

    def target_transform(y):
        y = frh01.mapping.loc[y].id
        return torch.tensor(y, dtype=torch.long)

    frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)
    frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)
    frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, transform=transform,
                                    target_transform=target_transform, padding_value=padded_value)
    if mode == "evaluation":
        frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath, transform=transform,
                                        target_transform=target_transform, padding_value=padded_value)
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
        testdataset = frh04
    elif mode == "validation":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02])
        testdataset = frh03
    else:
        raise ValueError("only --mode 'validation' or 'evaluation' allowed")

    traindataloader = DataLoader(traindatasets, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)

    meta = dict(
        num_classes=len(selected_bands),
        ndims=len(frh01.classes),
        sequencelength=sequencelength
    )

    return traindataloader, testdataloader, meta


def get_model(model, ndims, num_classes, sequencelength, device, **hyperparameter):
    if model == "LSTM":
        model = LSTM(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif model == "StarRNN":
        model = StarRNN(input_dim=ndims,
                        num_classes=num_classes,
                        bidirectional=False,
                        use_batchnorm=False,
                        use_layernorm=True,
                        device=device,
                        **hyperparameter).to(device)
    elif model == "InceptionTime":
        model = InceptionTime(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device)
    elif model == "MSResNet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif model == "TransformerEncoder":
        model = TransformerModel(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength,
                            activation="relu",
                            **hyperparameter).to(device)
    elif model == "TempCNN":
        model = TempCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(
            device)
    else:
        raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")

    return model


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


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true = batch
            loss = criterion(model.forward(x.to(device)), y_true.to(device))
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)


def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true = batch
                logprobabilities = model.forward(x.to(device))
                y_pred = logprobabilities.argmax(-1)
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models on the'
                                                 'BreizhCrops dataset. This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument(
        'model', type=str, default="LSTM", help='select model architecture. Available models are '
                                                '"LSTM","TempCNN","MSRestNet","TransformerEncoder"')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=256, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-e', '--epochs', type=int, default=150, help='number of training epochs (training on entire dataset)')
    parser.add_argument(
        '-m', '--mode', type=str, default="validation", help='training mode. Either "validation" '
                                                             '(train on FRH01+FRH02 test on FRH03) or '
                                                             '"evaluation" (train on FRH01+FRH02+FRH03 test on FRH04)')
    parser.add_argument(
        '-D', '--datapath', type=str, default="../data", help='directory to download and store the dataset')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-H', '--hyperparameter', type=str, default=None, help='model specific hyperparameter as single string, '
                                                               'separated by comma of format param1=value1,param2=value2')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-6, help='optimizer weight_decay (default 1e-6)')
    parser.add_argument(
        '--learning-rate', type=float, default=1e-2, help='optimizer learning rate (default 1e-2)')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
    args, _ = parser.parse_known_args()

    hyperparameter_dict = dict()
    if args.hyperparameter is not None:
        for hyperparameter_string in args.hyperparameter.split(","):
            param, value = hyperparameter_string.split("=")
            hyperparameter_dict[param] = float(value) if '.' in value else int(value)
    args.hyperparameter = hyperparameter_dict

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    args = parse_args()

    train(args)
