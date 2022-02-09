import sys

sys.path.append("./models")
sys.path.append("..")

import argparse

import breizhcrops
from breizhcrops.models import LSTM, TempCNN, MSResNet, TransformerModel, InceptionTime, StarRNN, OmniScaleCNN, PETransformerModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch
import pandas as pd
import os
import sklearn.metrics


def train(args):
    traindataloader, testdataloader, meta = get_dataloader(args.datapath, args.mode, args.batchsize, args.workers, args.preload_ram, args.level)

    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, sequencelength, device, **args.hyperparameter)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.modelname += f"_learning-rate={args.learning_rate}_weight-decay={args.weight_decay}"
    print(f"Initialized {model.modelname}")

    logdir = os.path.join(args.logdir, model.modelname)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    log = list()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)
        test_loss, y_true, y_pred, *_ = test_epoch(model, criterion, testdataloader, device)
        scores = metrics(y_true.cpu(), y_pred.cpu())
        scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
        test_loss = test_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
        print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)

        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = test_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(logdir, "trainlog.csv"))

def get_dataloader(datapath, mode, batchsize, workers, preload_ram=False, level="L1C"):
    print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")
    datapath = os.path.abspath(datapath)

    if mode == "unittest":
        belle_ile = breizhcrops.BreizhCrops(region="belle-ile", root=datapath)
    else:
        frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath,
                                        preload_ram=preload_ram, level=level)
        frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,
                                        preload_ram=preload_ram, level=level)
        frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath,
                                        preload_ram=preload_ram, level=level)

    if "evaluation" in mode:
            frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath,
                                            preload_ram=preload_ram, level=level)

    if mode == "evaluation" or mode == "evaluation1":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
        testdataset = frh04
    elif mode == "evaluation2":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh04])
        testdataset = frh03
    elif mode == "evaluation3":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh03, frh04])
        testdataset = frh02
    elif mode == "evaluation4":
        traindatasets = torch.utils.data.ConcatDataset([frh02, frh03, frh04])
        testdataset = frh01
    elif mode == "validation":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02])
        testdataset = frh03
    elif mode == "unittest":
        traindatasets = belle_ile
        testdataset = belle_ile
    else:
        raise ValueError("only --mode 'validation' or 'evaluation' allowed")

    traindataloader = DataLoader(traindatasets, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)

    meta = dict(
        ndims=13 if level == "L1C" else 10,
        num_classes=len(belle_ile.classes) if mode == "unittest" else len(frh01.classes),
        sequencelength=45
    )

    return traindataloader, testdataloader, meta


def get_model(modelname, ndims, num_classes, sequencelength, device, **hyperparameter):
    modelname = modelname.lower() #make case invariant
    if modelname == "omniscalecnn":
        model = OmniScaleCNN(input_dim=ndims, num_classes=num_classes, sequencelength=sequencelength, **hyperparameter).to(device)
    elif modelname == "lstm":
        model = LSTM(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname == "starrnn":
        model = StarRNN(input_dim=ndims,
                        num_classes=num_classes,
                        bidirectional=False,
                        use_batchnorm=False,
                        use_layernorm=True,
                        device=device,
                        **hyperparameter).to(device)
    elif modelname == "inceptiontime":
        model = InceptionTime(input_dim=ndims, num_classes=num_classes, device=device,
                              **hyperparameter).to(device)
    elif modelname == "msresnet":
        model = MSResNet(input_dim=ndims, num_classes=num_classes, **hyperparameter).to(device)
    elif modelname in ["transformerencoder","transformer"]:
        model = TransformerModel(input_dim=ndims, num_classes=num_classes,
                            activation="relu",
                            **hyperparameter).to(device)
    elif modelname in ["petransformer"]:
        model = PETransformerModel(input_dim=ndims, num_classes=num_classes,
                                 activation="relu",
                                 **hyperparameter).to(device)
    elif modelname == "tempcnn":
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
            x, y_true, _ = batch
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
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, field_id = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models on the'
                                                 'BreizhCrops dataset. This script trains a model on training dataset'
                                                 'partition, evaluates performance on a validation or evaluation partition'
                                                 'and stores progress and model paths in --logdir')
    parser.add_argument(
        'model', type=str, default="LSTM", help='select model architecture. Available models are '
                                                '"LSTM","TempCNN","MSRestNet","TransformerEncoder"')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=1024, help='batch size (number of time series processed simultaneously)')
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
        '--level', type=str, default="L1C", help='Sentinel 2 processing level (L1C, L2A)')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-6, help='optimizer weight_decay (default 1e-6)')
    parser.add_argument(
        '--learning-rate', type=float, default=1e-2, help='optimizer learning rate (default 1e-2)')
    parser.add_argument(
        '--preload-ram', action='store_true', help='load dataset into RAM upon initialization')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
    args = parser.parse_args()

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
