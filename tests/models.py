import sys
import os

this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, ".."))

from breizhcrops.models import pretrained
from breizhcrops import BreizhCrops
from examples.train import test_epoch
import pytest
import torch

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/tmp')
TESTS_BATCH_SIZE = int(os.environ.get('TESTS_BATCH_SIZE', '256'))

def evaluate_models(model, region, expected_accuracies, tolerance):

    bzh = BreizhCrops(region=region, root=TESTS_DATA_ROOT, load_timeseries=True)

    dataloader = torch.utils.data.DataLoader(bzh, batch_size=TESTS_BATCH_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_pre = pretrained(model, device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    losses, ytrue, ypred, score, fieldlist = test_epoch(model_pre, criterion, dataloader, device)

    pred_acc = (ytrue.cpu() == ypred.cpu()).float().mean().numpy()

    assert pred_acc >= (expected_accuracies[
                            model] - tolerance), f"Model seems to underperform. {model} has {pred_acc:.2f} overall accuracy " \
                                                 f"but expected {expected_accuracies[model]:.2f} overall accuracy."


@pytest.mark.parametrize("model", ["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn",
                                   "transformer"])  # "inceptiontime"
def test_evaluate_models(model):

    expected_accuracies = {
        "lstm": 0.8,
        "msresnet": 0.78,
        "omniscalecnn": 0.77,
        "starrnn": 0.79,
        "tempcnn": 0.79,
        "transformer": 0.8,
        "inceptiontime": 0.8
    }
    tolerance = 0.02

    evaluate_models(model, "frh04", expected_accuracies, tolerance)


@pytest.mark.parametrize("model", ["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn",
                                   "transformer"])  # "inceptiontime"
def test_evaluate_models_fast(model):
    """A test on the significantly smaller belle-ile region only to run tests.
    Accuracies on belle-ile are significantly lower because fields are quite out-of-distribution
    (belle-ile is an island off the coast while the training data is in brittany proper)"""
    expected_accuracies = {
        "lstm": 0.65,
        "msresnet": 0.62,
        "omniscalecnn": 0.6,
        "starrnn": 0.60,
        "tempcnn": 0.60,
        "transformer": 0.69,
        "inceptiontime": 0.8
    }
    tolerance = 0.04

    evaluate_models(model, "belle-ile", expected_accuracies, tolerance)

if __name__ == "__main__":
    test_evaluate_models("omniscalecnn")
