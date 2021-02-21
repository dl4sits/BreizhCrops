from breizhcrops import BreizhCrops
from examples.train import get_model
import breizhcrops
import torch
import pytest


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
    datapath = "/tmp"

    BreizhCrops(region="frh04", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh01", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh02", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh03", root=datapath, load_timeseries=False)
    BreizhCrops(region="belle-ile", root=datapath, load_timeseries=False)

    BreizhCrops(region="frh04", root=datapath, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh01", root=datapath, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh02", root=datapath, load_timeseries=False, level="L2A")
    BreizhCrops(region="frh03", root=datapath, load_timeseries=False, level="L2A")
    BreizhCrops(region="belle-ile", root=datapath, load_timeseries=False, level="L2A")


def test_pretrained():
    x = torch.zeros(1, 45, 13)
    for model in ["omniscalecnn", "lstm", "tempcnn", "msresnet", "inceptiontime", "starrnn", "transformer"]:
        breizhcrops.models.pretrained(model)(x)


def test_breizhcrops_index_columnames():
    l1c = BreizhCrops(region="frh01", root="/tmp", load_timeseries=False)
    l2a = BreizhCrops(region="frh01", root="/tmp", load_timeseries=False, level="L2A")
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


@pytest.mark.skip(reason="skipping. requires downloading 100mb which is too heavy for a test")
def test_breizhcrops_geodataframe():
    BreizhCrops(region="frh01", root="/tmp", load_timeseries=False).geodataframe()
    BreizhCrops(region="frh01", root="/tmp", load_timeseries=False, level="L2A").geodataframe()


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
            url = url_or_dict.replace("/dl/","/getlink/")

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
    BreizhCrops(region="belle-ile", root="/tmp", load_timeseries=False).geodataframe()
    dataset = BreizhCrops(region="belle-ile", root="/tmp", level="L1C")
    dataset[0]
    dataset = BreizhCrops(region="belle-ile", root="/tmp", level="L2A")
    dataset[0]


def test_get_codes_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).get_codes()


@pytest.mark.skip(reason="skipping. takes to long for a test")
def test_write_index():
    BreizhCrops(region="frh01", root="/tmp", load_timeseries=True, level="L1C",recompile_h5_from_csv=True, year=2017)
    BreizhCrops(region="frh02", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2017)
    BreizhCrops(region="frh03", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2017)
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2017)
    BreizhCrops(region="belle-ile", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True,
                year=2018)

    BreizhCrops(region="frh01", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2018)
    BreizhCrops(region="frh02", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2018)
    BreizhCrops(region="frh03", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2018)
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=True, level="L1C", recompile_h5_from_csv=True, year=2018)

