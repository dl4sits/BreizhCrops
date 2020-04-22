import sys
sys.path.append("./models")
sys.path.append("..")
import argparse
import numpy as np
from train import train
import torch

def dict2str(hyperparameter_dict):
    return ",".join([f"{k}={v}" for k,v in hyperparameter_dict.items()])


def tune(args):
    args.mode = "validation"

    while True:
        if args.model == "LSTM":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            hyperparameter_dict = dict(
                hidden_dims=int(np.random.choice([32, 64, 128])),
                num_layers=int(np.random.choice([1, 2, 3, 4])),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "MSResNet":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            hyperparameter_dict = dict(
                hidden_dims=np.random.choice([32, 64, 128])
            )

        elif args.model == "TransformerEncoder":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            d_model = int(np.random.choice([32, 64, 128, 256, 512], 1))
            n_head = int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1))

            hyperparameter_dict = dict(
                d_model=d_model,
                n_head=n_head,
                n_layers=int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1)),
                d_inner=d_model * 4,
                dropout=np.random.uniform(0, 0.8),
                d_k=d_model//n_head,
                d_word_vec=d_model,
                d_v=d_model//n_head
            )

        elif args.model == "TempCNN":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            hyperparameter_dict = dict(
                kernel_size=np.random.choice([3, 5, 7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "StarRNN":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            hyperparameter_dict = dict(
                hidden_dims=np.random.choice([32, 64, 128]),
                num_layers=np.random.choice([1, 2, 3, 4]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "InceptionTime":
            args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
            args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

            hyperparameter_dict = dict(
                num_layers=np.random.choice([1, 2, 3, 4]),
                hidden_dims=np.random.choice([32, 64, 128])
            )

        else:
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder',"
                             "'TempCNN','StarRNN' or 'InceptionTime'")

        args.hyperparameter = hyperparameter_dict
        hyperparameter_string = dict2str(hyperparameter_dict)

        if args.model == "LSTM":
            args.store = f"/tmp/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "MSResNet":
            args.store = f"/tmp/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TransformerEncoder":
            args.store = f"/tmp/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TempCNN":
            args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "StarRNN":
            args.store = f"/tmp/StarRNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/StarRNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "InceptionTime":
            args.store = f"/tmp/InceptionTime-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"{args.logdir}/InceptionTime-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='select model architecture')
    parser.add_argument('-D', '--datapath', type=str, default="../data", help='path to dataset')
    parser.add_argument('-l', '--logdir', type=str, default="/tmp", help='logdir')
    parser.add_argument(
        '-b', '--batchsize', type=int, default=256, help='batch size (number of time series processed simultaneously)')
    parser.add_argument(
        '-e', '--epochs', type=int, default=10, help='number of training epochs (training on entire dataset)')
    parser.add_argument(
        '-w', '--workers', type=int, default=0, help='number of CPU workers to load the next batch')
    parser.add_argument(
        '-d', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". '
                                                       'default will check by torch.cuda.is_available() ')

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

if __name__ == '__main__':
    args = parse_args()
    tune(args)