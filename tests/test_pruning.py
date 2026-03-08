import torch

from edge_opt.model import SmallCNN
from edge_opt.pruning import structured_channel_prune


def test_structured_pruning_remaps_classifier_channels() -> None:
    model = SmallCNN(conv1_channels=4, conv2_channels=6, num_classes=3)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(3, 6))

    pruned = structured_channel_prune(model, pruning_level=0.5)

    assert pruned.classifier.in_features == pruned.conv2.out_channels
    # remapped classifier columns should be a subset of original channels
    for row in pruned.classifier.weight:
        assert torch.all(torch.isin(row, model.classifier.weight.flatten()))
