"""InceptionTime's feature extractor should train and survive checkpointing."""

import torch
from breizhcrops.models import InceptionTime


def test_inceptiontime_registration_and_checkpoint():
    torch.manual_seed(0)
    model = InceptionTime(num_classes=3, input_dim=13, num_layers=3, hidden_dims=16)
    params = dict(model.named_parameters())
    assert "inception_modules_list.0.bottleneck.weight" in params
    assert "inception_modules_list.0.convolutions.0.weight" in params
    assert "shortcut_layer_list.0.sc.0.weight" in params

    x = torch.randn(4, 45, 13)
    y = torch.tensor([0, 1, 2, 1])
    before = model.inception_modules_list[0].bottleneck.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    torch.nn.functional.nll_loss(model(x), y).backward()
    optimizer.step()
    assert not torch.equal(before, model.inception_modules_list[0].bottleneck.weight)

    model.eval()
    assert not model.inception_modules_list[0].bn_relu[0].training
    restored = InceptionTime(num_classes=3, input_dim=13, num_layers=3, hidden_dims=16)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), model(x))
