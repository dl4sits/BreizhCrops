import sys
import os

this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, ".."))

from examples.train import train, parse_args
import pytest
import torch

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/tmp')


@pytest.mark.parametrize("model", ["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn",
                                   "transformer"])  # "inceptiontime"
def test_training(model):
    args = parse_args()
    args.mode = "unittest"
    args.datapath = TESTS_DATA_ROOT
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model = model
    args.epochs = 1
    args.learning_rate = 0.01
    args.weight_decay = 4e-06

    train(args)
