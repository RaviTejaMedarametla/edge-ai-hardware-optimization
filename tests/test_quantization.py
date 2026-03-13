import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from edge_opt.model import SmallCNN
from edge_opt.quantization import to_int8


def test_quantization_metadata(tmp_path) -> None:
    model = SmallCNN()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    metadata_path = tmp_path / "quant_meta.json"
    _ = to_int8(model, loader, calibration_batches=2, metadata_path=metadata_path)
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    module_keys = metadata.get("modules", {}).keys()
    assert any("conv1" in k for k in module_keys)
    assert any("conv2" in k for k in module_keys)
    assert any("classifier" in k for k in module_keys)
