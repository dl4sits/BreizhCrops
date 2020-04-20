from argparse import Namespace
import numpy as np

from train import train # import the train() function from the train.py script


def tune(modelarg):
    # default parameters
    args = Namespace(
        mode="validation",
        model=modelarg,
        epochs=10,
        datapath="../data",
        batchsize=256,
        workers=0,
        device="cuda",
        logdir="D:/Hiwi_lokal/Logs"
    )

    while True:
        if args.model == "LSTM":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                # kernel_size = np.random.choice([3,5,7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                num_layers=np.random.choice([1, 2, 3, 4]),
                dropout=np.random.uniform(0, 0.8)
            )

        elif args.model == "MSResNet":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                input_dim = 1,
                num_classes = 10,
                hidden_dims = np.random.choice([32, 64, 128])
            )

        elif args.model == "TransformerEncoder":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                input_dim=10,
                len_max_seq=100,
                d_word_vec=512,
                d_model=512,
                d_inner=2048,
                n_layers=6,
                n_head=8,
                d_k=64,
                d_v=64,
                dropout=np.random.uniform(0.1, 0.3),
                num_classes=6
            )

        elif args.model == "TempCNN":
            args.learning_rate = np.random.uniform(1e-2, 1e-4)
            args.weight_decay = np.random.uniform(1e-2, 1e-8)

            hyperparameter_dict = dict(
                kernel_size=np.random.choice([3, 5, 7]),
                hidden_dims=np.random.choice([32, 64, 128]),
                dropout=np.random.uniform(0, 0.8)
                # num_layers = np.random.choice([1,2,3,4])
            )

        else:
            raise ValueError("invalid model argument. choose from 'LSTM','MSResNet','TransformerEncoder', or 'TempCNN'")

  
        # parse and add the hyperparameter string
        args.hyperparameter = hyperparameter_dict
  
        # define a descriptive model name that contains all the hyperparameters
        if args.model == "LSTM":
            args.store = f"/tmp/LSTM-{args.learning_rate}-{args.hyperparameter}"
            args.logdir = "D:/Hiwi_lokal/Logs/LSTM-{args.learning_rate}-{args.hyperparameter}"

        elif args.model == "MSResNet":
            args.store = f"/tmp/MSResNet-{args.learning_rate}-{args.hyperparameter}"
            args.logdir = "D:/Hiwi_lokal/Logs/MSResNet-{args.learning_rate}-{args.hyperparameter}"

        elif args.model == "TransformerEncoder":
            args.store = f"/tmp/TransformerEncoder-{args.learning_rate}-{args.hyperparameter}"
            args.logdir = "D:/Hiwi_lokal/Logs/TransformerEncoder-{args.learning_rate}-{args.hyperparameter}"

        elif args.model == "TempCNN":
            args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.hyperparameter}"
            args.logdir = "D:/Hiwi_lokal/Logs/TempCNN-{args.learning_rate}-{args.hyperparameter}"
  
        # start training
        train(args)
