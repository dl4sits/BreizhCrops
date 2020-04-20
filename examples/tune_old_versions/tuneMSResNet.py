from argparse import Namespace
import numpy as np

from train import train # import the train() function from the train.py script

# default parameters
args = Namespace(
    mode="validation",
    model="MSRestNet",
    epochs=1,
    datapath="../data",
    batchsize=256,
    workers=0,
    device="cpu",
    logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs"
)

# train until manually stopped with random hyperparapeters
while True:

  args.learning_rate = np.random.uniform(1e-2,1e-4)
  args.weight_decay = np.random.uniform(1e-3,1e-5)
  
#  hyperparameter_dict = dict(
  input_dim = 1
  num_classes = 10
  hidden_dims = np.random.choice([32,64,128])

#  )
  
# parse and add the hyperparameter string
#  args.hyperparameter = hyperparameter_dict# dict2str(hyperparameter_dict)
  
# define a descriptive model name that contains all the hyperparameters
  args.store = f"/tmp/LSTM-{args.learning_rate}-{args.hyperparameter}"
  
# start training
  train(args)  
