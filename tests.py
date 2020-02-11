from breizhcrops.models import TempCNN, LSTM, TransformerEncoder, MSResNet
from breizhcrops import BreizhCrops

def test_initTempCNN():
    assert isinstance(TempCNN(input_dim=13, num_classes=10, sequencelength=50), TempCNN)

def test_initLongShortTermMemory():
    assert isinstance(LSTM(input_dim=1, hidden_dims=3, num_classes=5, num_layers=1, dropout=0.2, bidirectional=False,
                use_layernorm=True), LSTM)

def test_initTransformerEncoder():
    assert isinstance(TransformerEncoder(input_dim=13, len_max_seq=100,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, num_classes=6), TransformerEncoder)

def test_initMSResNet():
    assert isinstance(MSResNet(input_dim=13), MSResNet)

def test_init_breizhcrops():
    datapath = "/tmp"
    transform = None
    target_transform = None

    BreizhCrops(region="frh04", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh01", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh02", root=datapath, load_timeseries=False)
    BreizhCrops(region="frh03", root=datapath, load_timeseries=False)
    
def test_load_classmapping_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).load_classmapping(None)

def test_load_classmapping_breizhcrops():
    BreizhCrops(region="frh04", root="/tmp", load_timeseries=False).get_codes()