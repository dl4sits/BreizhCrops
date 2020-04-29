from train import get_model, get_dataloader, train_epoch, test_epoch
import torch
import argparse
import sklearn.metrics
import os
import numpy as np

def main(args):
    traindataloader, testdataloader, meta = get_dataloader(args.datapath, "evaluation", args.batchsize, args.workers)

    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]

    print(f"Logging results to {args.logdir}")
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    epochs, learning_rate, weight_decay = select_hyperparameter(args.model)

    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, sequencelength, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.modelname += f"_learning-rate={learning_rate}_weight-decay={weight_decay}"
    print(f"Initialized {model.modelname}")
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(epochs):
        print(f"train epoch {epoch}")
        train_epoch(model, optimizer, criterion, traindataloader, device)
    losses, y_true, y_pred, y_score, field_ids = test_epoch(model, criterion, dataloader=testdataloader, device=device)

    print(f"saving results to {logdir}")
    print(sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu()),
          file=open(os.path.join(logdir, "classification_report.txt"), "w"))
    np.save(os.path.join(logdir,"y_pred.npy"), y_pred.cpu().numpy())
    np.save(os.path.join(logdir, "y_true.npy"), y_true.cpu().numpy())
    np.save(os.path.join(logdir, "y_score.npy"), y_score.cpu().numpy())
    np.save(os.path.join(logdir, "field_ids.npy"), field_ids.numpy())
    model.save(os.path.join(logdir, model.modelname + ".pth"))

def select_hyperparameter(model):
    """
    a function to select training-specific hyperparameter. the model-specific hyperparameter should be set
    in the defaults of the respective model parameters.
    """
    if model == "LSTM":
        epochs, learning_rate, weight_decay = 17, 0.009880117756170353, 5.256755602421856e-07
    elif model == "StarRNN":
        epochs, learning_rate, weight_decay = 17, 0.008960989762612663, 2.2171861339535254e-06
    elif model == "InceptionTime":
        epochs, learning_rate, weight_decay = 23, 0.0005930998594456241, 1.8660112778851542e-05
    elif model == "MSResNet":
        epochs, learning_rate, weight_decay = 23, 0.0006271686393146093, 4.750234747127917e-06
    elif model == "TransformerEncoder":
        epochs, learning_rate, weight_decay = 26, 0.0013144015360979785, 5.523908582054716e-08
    elif model == "TempCNN":
        epochs, learning_rate, weight_decay = 11, 0.00023892874563871753, 5.181869707846283e-05
    elif model == "OmniScaleCNN":
        epochs, learning_rate, weight_decay = 19, 0.001057192239267413, 2.2522895556530792e-07
    return epochs, learning_rate, weight_decay

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
        '-D', '--datapath', type=str, default="../data", help='directory to download and store the dataset')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')
    parser.add_argument(
        '-l', '--logdir', type=str, default="/tmp", help='logdir to store progress and models (defaults to /tmp)')
    parser.add_argument(
        '--preload-ram', action='store_true', help='load dataset into RAM upon initialization')

    args, _ = parser.parse_known_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)