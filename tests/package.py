import sys
import os
this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder,".."))
from breizhcrops import BreizhCrops
import breizhcrops
import torch
import os
from examples.train import get_model
import pytest
from itertools import product

TESTS_DATA_ROOT = os.environ.get('TESTS_DATA_ROOT', '/tmp')

def test_get_model():
    batchsize = 16
    ndims = 13
    num_classes = 10
    sequencelength = 24

    data = torch.rand(batchsize, sequencelength, ndims)
    for model in ["TempCNN", "StarRNN", "LSTM", "TransformerEncoder", "InceptionTime", "MSResNet"]:
        model = get_model(model, ndims=ndims, num_classes=num_classes, sequencelength=sequencelength,
                          device=torch.device("cpu"))
        pred = model(data)
        assert pred.shape == (batchsize, num_classes)


def test_init_breizhcrops():

    BreizhCrops(region="frh04", root=TESTS_DATA_ROOT, load_timeseries=False)
    BreizhCrops(region="frh01", root=TESTS_DATA_ROOT, load_timeseries=False)
    BreizhCrops(region="frh02", root=TESTS_DATA_ROOT, load_timeseries=False)
    BreizhCrops(region="frh03", root=TESTS_DATA_ROOT, load_timeseries=False)
    BreizhCrops(region="belle-ile", root=TESTS_DATA_ROOT, load_timeseries=False)

    BreizhCrops(region="frh04", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh01", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh02", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh03", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")
    BreizhCrops(region="belle-ile", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")


def test_pretrained():
    x = torch.zeros(1, 45, 13)
    #for model in ["omniscalecnn", "lstm", "tempcnn", "msresnet", "InceptionTime", "starrnn", "transformer"]:
    for model in ["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn", "transformer"]:
        breizhcrops.models.pretrained(model)(x)


def test_breizhcrops_index_columnames():
    l1c = BreizhCrops(region="frh01", root=TESTS_DATA_ROOT, load_timeseries=False)
    l2a = BreizhCrops(region="frh01", root=TESTS_DATA_ROOT, load_timeseries=False, level="L2A")
    reference = ['CODE_CULTU', 'path', 'meanCLD', 'sequencelength', 'id', 'classid', 'classname', 'region']
    reference.sort()

    assert len(list(l1c.index.columns)) == len(reference)
    assert len(list(l2a.index.columns)) == len(reference)

    l1c_list = list(l1c.index.columns)
    l2a_list = list(l2a.index.columns)
    l1c_list.sort()
    l2a_list.sort()

    for colref, coll1c, coll2a in zip(reference, l1c_list, l2a_list):
        assert colref == coll1c
        assert colref == coll2a





#def test_raw_processing():
#    BreizhCrops(region="frh03", root="/tmp, load_timeseries=True, level="L2A", recompile_h5_from_csv=True)


def test_urls():
    import requests
    from breizhcrops.datasets.urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, SHP_URLs, H5_URLs

    def check(url_or_dict):
        """
        recursively check if urls are valid (assuming sync and share urls)
        e.g. https://syncandshare.lrz.de/dl/fiKfoL1VW9jiDXPgnVXu7ZFK/frh04.tar.gz
        """
        if url_or_dict is None:
            pass # None is allowed placeholder
        elif isinstance(url_or_dict, str):
            # sync and share urls return a html page with /getlink/ which is much faster
            # than checking the download "/dl/" file
            url = url_or_dict.replace("/dl/", "/getlink/")

            response = requests.get(url)
            code = response.status_code
            if not code < 400:
                raise ValueError(f"url {url} returned code {code}")
        elif isinstance(url_or_dict, dict):
            for v in url_or_dict.values():
                check(v)

    check(CODESURL)
    check(CLASSMAPPINGURL)
    check(INDEX_FILE_URLs)
    check(SHP_URLs)
    check(H5_URLs)


def test_belle_ile():
    BreizhCrops(region="belle-ile", root=TESTS_DATA_ROOT, load_timeseries=False).geodataframe()
    dataset = BreizhCrops(region="belle-ile", root=TESTS_DATA_ROOT, level="L1C")
    dataset[0]
    dataset = BreizhCrops(region="belle-ile", root=TESTS_DATA_ROOT, level="L2A")
    dataset[0]


def test_get_codes_breizhcrops():
    BreizhCrops(region="frh04", root=TESTS_DATA_ROOT, load_timeseries=False).get_codes()

@pytest.mark.parametrize("model,ndims,num_classes,sequencelength",
                         product(["omniscalecnn", "lstm", "tempcnn", "msresnet", "starrnn",
                                   "transformer", "inceptiontime"],
                                 [12,13,20], # ndims
                                 [10,20], # num classes
                                 [20,45]) # sequencelength
                        )
def test_models_dummy_data(model, ndims, num_classes, sequencelength):
    device = "cpu"

    batch_size = 16
    X = torch.zeros(batch_size, sequencelength, ndims).to(device)

    torchmodel = get_model(model, ndims, num_classes, sequencelength, device)
    y_logprobabilities = torchmodel(X)
    assert y_logprobabilities.shape == (batch_size, num_classes), "model prediction shape inconsistent with num classes"
    y_scores = y_logprobabilities.exp()
    assert torch.isclose(y_scores.sum(1), torch.ones(batch_size).to(device)).all(), "class probabilities do not sum to one"

