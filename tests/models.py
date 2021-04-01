import sys
import os
this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, ".."))

from breizhcrops.models import pretrained
from breizhcrops import BreizhCrops
from examples.train import test_epoch
import pytest
import torch
import numpy

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/tmp')


@pytest.mark.parametrize("model", ["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn",
                                   "transformer"])  # "inceptiontime"
def test_evaluate_models(model):
    bzh = BreizhCrops(region="frh04", root=TESTS_DATA_ROOT, load_timeseries=True)

    dataloader = torch.utils.data.DataLoader(bzh)
    model_pre = pretrained(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    losses, ytrue, ypred, score, fieldlist = test_epoch(model_pre, criterion, dataloader, device)

    pred_tmp = (ytrue == ypred)*1
    size = int(pred_tmp.shape[0])
    pred = [None] * size
    for i in range(0, size):
        pred[i] = float(pred_tmp[i])
    pred_acc = numpy.mean(pred)
    '''
    if model == "inceptiontime":
        acc = 0.8
    '''
    if model == "lstm":
        acc = 0.8
    elif model == "msresnet":
        acc = 0.78
    elif model == "omniscalecnn":
        acc = 0.77
    elif model == "starrnn":
        acc = 0.79
    elif model == "tempcnn":
        acc = 0.79
    elif model == "transformer":
        acc = 0.8

    tolerance = (acc/100)*2

    assert pred_acc >= (acc-tolerance), f"Model seems to underperform. {model} has {pred_acc} overall accuracy " \
                                        f"but expected {acc} overall accuracy."


if __name__ == "__main__":
    test_evaluate_models("omniscalecnn")
