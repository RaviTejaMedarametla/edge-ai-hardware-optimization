import torch
from torch import nn

from edge_opt.model import SmallCNN
from edge_opt.pruning import structured_channel_prune


def test_structured_pruning_remaps_classifier_channels() -> None:
    model = SmallCNN(conv1_channels=4, conv2_channels=6, num_classes=3)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(3, 6))

    pruned = structured_channel_prune(model, pruning_level=0.5)

    assert pruned.classifier.in_features == pruned.conv2.out_channels
    for row in pruned.classifier.weight:
        assert torch.all(torch.isin(row, model.classifier.weight.flatten()))


def test_pruning_on_custom_cnn() -> None:
    class CustomNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 10, kernel_size=3)
            self.linear = nn.Linear(10 * 26 * 26, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            x = x.flatten(start_dim=1)
            return self.linear(x)

    model = CustomNet()
    pruned = structured_channel_prune(model, pruning_level=0.3)
    assert pruned.conv.out_channels < 10
    assert pruned.linear.in_features == pruned.conv.out_channels * 26 * 26
