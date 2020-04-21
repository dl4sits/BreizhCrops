import sys
sys.path.append("./models")
sys.path.append("..")
import argparse
from argparse import Namespace
import numpy as np
from train import train


def dict2str(hyperparameter_dict):
    return ",".join([f"{k}={v}" for k,v in hyperparameter_dict.items()])


def tune(modelpara):
    args = Namespace(
        mode="validation",
        model=modelpara.model,
        epochs=10,
        datapath="../data",
        batchsize=256,
        workers=16,
        device="cuda",
        logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs"
    )

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
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder','TempCNN','StarRNN' or 'InceptionTime'")

        args.hyperparameter = hyperparameter_dict
        hyperparameter_string = dict2str(hyperparameter_dict)

        if args.model == "LSTM":
            args.store = f"/tmp/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "MSResNet":
            args.store = f"/tmp/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TransformerEncoder":
            args.store = f"/tmp/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TempCNN":
            args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "StarRNN":
            args.store = f"/tmp/StarRNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/StarRNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "InceptionTime":
            args.store = f"/tmp/InceptionTime-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/InceptionTime-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='select model architecture')
    args = parser.parse_args()
    return args


tune(parse_args())