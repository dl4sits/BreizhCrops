from breizhcrops.models import TempCNN, LSTM, Transformer, MSResNet, InceptionTime, StarRNN
from breizhcrops import BreizhCrops

def test_initTempCNN():
    assert isinstance(TempCNN(input_dim=13, num_classes=10, sequencelength=50), TempCNN)

def test_initLongShortTermMemory():
    assert isinstance(LSTM(input_dim=1, hidden_dims=3, num_classes=5, num_layers=1, dropout=0.2, bidirectional=False,
                use_layernorm=True), LSTM)

def test_initTransformerEncoder():
    assert isinstance(Transformer(input_dim=8, num_classes=2), Transformer)

def test_initInceptionTime():
    assert isinstance(InceptionTime(input_dim=28, hidden_dims=32, num_classes=2, num_layers=2), InceptionTime)

def test_initMSResNet():
    assert isinstance(MSResNet(input_dim=13), MSResNet)

def test_initStarRNN():
    assert isinstance(StarRNN(input_dim=13, hidden_dims=128, nclasses=13, num_rnn_layers=4, dropout=0.2,
                              bidirectional=False,use_batchnorm=False, use_layernorm=True), StarRNN)

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