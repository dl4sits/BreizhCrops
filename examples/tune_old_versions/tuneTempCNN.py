from argparse import Namespace
import numpy as np

from train import train # import the train() function from the train.py script

# default parameters
args = Namespace(
    mode="validation",
    model="TempCNN",
    epochs=10,
    datapath="../data",
    batchsize=256,
    workers=0,
    device="cpu",
    logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs"
)

#def dict2str(hyperparameter_dict):
  #"""convenience function dict(a=1,b=2) -> 'a=1,b=2' """
  #return ",".join([f"{k}={v}" for k,v in hyperparameter_dict.items()])
  
# train until manually stopped
while True:
  #droput_array = np.random.uniform(0.4,0.6)
# randomly sample some hyperparameters
  args.learning_rate = np.random.uniform(1e-2,1e-4) # between 0 and 1. Smaller learning rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs. large = unstable training, tiny = process gets stuck/ unaible to train
  args.weight_decay = np.random.uniform(1e-3,1e-5) # reduce complexity
  
  hyperparameter_dict = dict(
    kernel_size = np.random.choice([3,5,7]),
    hidden_dims = np.random.choice([32,64,128]),
    dropout = np.random.uniform(0.4,0.6) # drop e.g randomly 50 % of units from each layer
    # num_layers = np.random.choice([1,2,3,4])
  )
  
# parse and add the hyperparameter string
  args.hyperparameter = hyperparameter_dict# dict2str(hyperparameter_dict)
  
# define a descriptive model name that contains all the hyperparameters
  args.store = f"/tmp/TempCNN-{args.learning_rate}-{args.hyperparameter}"
  
  args.logdir="/home/ga63cuh/nas/ga63cuh/ga63cuh/Hiwi/Logs/TempCNN-{args.learning_rate}-{args.hyperparameter}"
  
# start training
  train(args)  
