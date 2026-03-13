import torch

from edge_opt.model import SmallCNN
from edge_opt.pruning import structured_channel_prune


def test_structured_pruning_remaps_classifier_channels() -> None:
    model = SmallCNN(conv1_channels=4, conv2_channels=6, num_classes=3)
    with torch.no_grad():
        model.conv2.weight.fill_(0.0)
        model.conv2.bias.fill_(0.0)
        model.conv2.weight[1].fill_(10.0)
        model.conv2.weight[4].fill_(8.0)
        model.conv2.weight[5].fill_(6.0)
        model.classifier.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(3, 6))

    pruned = structured_channel_prune(model, pruning_level=0.5)

    assert pruned.classifier.in_features == pruned.conv2.out_channels
    expected_keep = torch.tensor([1, 4, 5])
    assert torch.allclose(pruned.classifier.weight, model.classifier.weight[:, expected_keep])
