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

def test_urls():
    import requests
    from breizhcrops.datasets.urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, SHP_URLs, H5_URLs

    def check(url_or_dict):
        """
        recursively check if urls are valid (assuming sync and share urls)
        e.g. https://syncandshare.lrz.de/dl/fiKfoL1VW9jiDXPgnVXu7ZFK/frh04.tar.gz
        """
        if url_or_dict is None:
            pass #None is allowed placeholder
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





def test_get_codes_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).get_codes()