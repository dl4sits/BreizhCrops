import sys

sys.path.append("./models")
sys.path.append("..")

import argparse

from argparse import Namespace
import numpy as np

import torch

from train import train  # import the train() function from the train.py script


# print(torch.__version__)

def dict2str(hyperparameter_dict):
    return ",".join([f"{k}={v}" for k,v in hyperparameter_dict.items()])


def tune(modelpara):
    # default parameters
    args = Namespace(
        mode="validation",
        model=modelpara.model,
        epochs=10,
        datapath="../data",
        batchsize=256,
        workers=16,
        device="cuda",
        logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs" # /home/ga63cuh/Documents/Logs
    )

    while True:
        if args.model == "LSTM":
            # args.learning_rate = np.random.uniform(1e-2, 1e-4) # 1e-(np.random.uniform())
            # args.weight_decay = np.random.uniform(1e-2, 1e-8)
            r1 = np.random.uniform(2, 4)
            r2 = np.random.uniform(2, 8)
            args.learning_rate = 10 ** (-r1)
            args.weight_decay = 10 ** (-r2)

            hyperparameter_dict = dict(
                # kernel_size = np.random.choice([3,5,7]),
                # num_classes=,
                hidden_dims=np.random.choice([32, 64, 128]),
                # num_layers=np.random.choice([1, 2, 3, 4]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "MSResNet":
            # args.learning_rate = np.random.uniform(1e-2, 1e-4) # 1e-(np.random.uniform())
            # args.weight_decay = np.random.uniform(1e-2, 1e-8)
            r1 = np.random.uniform(2, 4)
            r2 = np.random.uniform(2, 8)
            args.learning_rate = 10 ** (-r1)
            args.weight_decay = 10 ** (-r2)

            hyperparameter_dict = dict(
                # input_dim=,
                # num_classes=,
                # layers=,
                hidden_dims=np.random.choice([32, 64, 128])
            )

        elif args.model == "TransformerEncoder":
            # args.learning_rate = np.random.uniform(1e-2, 1e-4) # 1e-(np.random.uniform())
            # args.weight_decay = np.random.uniform(1e-2, 1e-8)
            r1 = np.random.uniform(2, 4)
            r2 = np.random.uniform(2, 8)
            args.learning_rate = 10 ** (-r1)
            args.weight_decay = 10 ** (-r2)

            d_model = int(np.random.choice([32, 64, 128, 256, 512], 1))
            n_head = int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1))

            hyperparameter_dict = dict(
                n_head=n_head, # np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1),
                n_layers=int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1)),
                dropout=np.random.uniform(0, 0.8),
                d_model=d_model,
                d_k=d_model//n_head,
                # input_dim=10,
                # len_max_seq=100,
                d_word_vec=d_model,
                d_inner=d_model*4,
                d_v=d_model//n_head # 64
                # num_classes=6
            )

        elif args.model == "TempCNN":
            # args.learning_rate = np.random.uniform(1e-2, 1e-4) # 1e-(np.random.uniform())
            # args.weight_decay = np.random.uniform(1e-2, 1e-8)
            r1 = np.random.uniform(2, 4)
            r2 = np.random.uniform(2, 8)
            args.learning_rate = 10 ** (-r1)
            args.weight_decay = 10 ** (-r2)

            hyperparameter_dict = dict(
                kernel_size=np.random.choice([3, 5, 7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                dropout=np.random.uniform(0, 0.8)
                # num_layers = np.random.choice([1,2,3,4])
            )

        else:
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")

        args.hyperparameter = hyperparameter_dict
        hyperparameter_string = dict2str(hyperparameter_dict)

        # define a descriptive model name that contains all the hyperparameters
        # change to /home/ga63cuh/Documents/Logs - trainlog namechange?
        if args.model == "LSTM":
            args.store = f"/tmp/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/LSTM-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "MSResNet":
            args.store = f"/tmp/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/Documents/Logs/MSResNet-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TransformerEncoder":
            args.store = f"/tmp/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TransformerEncoder-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        elif args.model == "TempCNN":
            args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"
            args.logdir = f"/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TempCNN-{args.learning_rate}-{args.weight_decay}-{hyperparameter_string}"

        train(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='select model architecture')
    args = parser.parse_args()
    return args


modelpara = parse_args()
tune(modelpara)