import sys
sys.path.append("./models")
sys.path.append("..")
import argparse
import numpy as np
from train import train
import torch

def tune(args):
    args.mode = "validation"

    while True:

        # common hyperparameter
        args.learning_rate = 10 ** (-(np.random.uniform(2, 4)))
        args.weight_decay = 10 ** (-(np.random.uniform(2, 8)))

        # model-specific hyperparameter
        if args.model == "OmniScaleCNN":

            # unclear of paramenter_number_of_layer_list should be tuned.
            # waiting for https://github.com/Wensi-Tang/OS-CNN/issues/1 response
            args.hyperparameter = dict()

        elif args.model == "LSTM":

            args.hyperparameter = dict(
                hidden_dims=int(np.random.choice([32, 64, 128])),
                num_layers=int(np.random.choice([1, 2, 3, 4])),
                bidirectional=bool(np.random.choice([True, False])),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "MSResNet":

            args.hyperparameter = dict(
                hidden_dims=np.random.choice([32, 64, 128])
            )

        elif args.model == "TransformerEncoder":

            d_model = int(np.random.choice([32, 64, 128, 256, 512], 1))

            args.hyperparameter = dict(
                d_model=d_model,
                n_head=int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1)),
                n_layers=int(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], 1)),
                d_inner=d_model * 4,
                dropout=np.random.uniform(0, 0.8),
            )

        elif args.model == "TempCNN":

            args.hyperparameter = dict(
                kernel_size=np.random.choice([3, 5, 7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "StarRNN":

            args.hyperparameter = dict(
                hidden_dims=np.random.choice([32, 64, 128]),
                num_layers=np.random.choice([1, 2, 3, 4]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "InceptionTime":
            args.hyperparameter = dict(
                num_layers=np.random.choice([1, 2, 3, 4]),
                hidden_dims=np.random.choice([32, 64, 128])
            )

        else:
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder',"
                             "'TempCNN','StarRNN' or 'InceptionTime'")

        try:
            train(args)
        except Exception as e:
            print("received error "+str(e))
            continue

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
    parser.add_argument(
        '--preload-ram', action='store_true', help='load dataset into RAM upon initialization')
	parser.add_argument('--level', type=str, default="L1C", help='select level either "L1C" or "L2A". Default will select "L1C".')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

if __name__ == '__main__':
    args = parse_args()
    tune(args)