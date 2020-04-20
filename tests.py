from breizhcrops import BreizhCrops
from examples.train import get_model
import torch

def test_get_model():
    batchsize = 16
    ndims = 13
    num_classes = 10
    sequencelength=24

    data = torch.rand(batchsize, sequencelength, ndims)
    for model in ["TempCNN", "StarRNN", "LSTM", "TransformerEncoder", "InceptionTime", "MSResNet"]:
        model = get_model(model, ndims=ndims, num_classes=num_classes, sequencelength=sequencelength,
                          device=torch.device("cpu"))
        pred = model(data)
        assert pred.shape == (batchsize, num_classes)

def test_init_breizhcrops():
    datapath = "/tmp"

    BreizhCrops(region="frh04", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh01", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh02", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh03", root=datapath, load_timeseries=False)
    
def test_load_classmapping_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).load_classmapping(None)

def test_get_codes_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).get_codes()