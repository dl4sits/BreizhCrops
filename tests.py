from breizhcrops.models import TempCNN, LSTM, TransformerEncoder, MSResNet

def test_initTempCNN():
    assert isinstance(TempCNN(input_dim=13, nclasses=10, sequence_length=50), TempCNN)

def test_initLongShortTermMemory():
    assert isinstance(LSTM(input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=1, dropout=0.2, bidirectional=False,
                 use_batchnorm=False, use_layernorm=True), LSTM)

def test_initTransformerEncoder():
    assert isinstance(TransformerEncoder(in_channels=13, len_max_seq=100,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, nclasses=6), TransformerEncoder)

def test_initMSResNet():
    assert isinstance(MSResNet(input_channel=13), MSResNet)