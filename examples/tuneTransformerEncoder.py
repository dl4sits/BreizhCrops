from argparse import Namespace
import numpy as np

from train import train

# default parameters
args = Namespace(
    mode="validation",
    model="TempCNN",
    epochs=1,
    datapath="../data",
    batchsize=256,
    workers=0,
    device="cpu",
    logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs"
)

# train until manually stopped with random hyperparameters
while True:

  args.learning_rate = np.random.uniform(1e-2,1e-4)
  args.weight_decay = np.random.uniform(1e-3,1e-5)
  
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
    dropout=np.random.uniform(0.1,0.3),
    num_classes=6
  )
  
# parse and add the hyperparameter string
  args.hyperparameter = hyperparameter_dict# dict2str(hyperparameter_dict)
  
# define a descriptive model name that contains all the hyperparameters
  args.store = f"/tmp/LSTM-{args.learning_rate}-{args.hyperparameter}"
  
# start training
  train(args)  
