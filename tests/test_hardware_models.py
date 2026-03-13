from edge_opt.hardware_models import AnalyticalHardwareModel
from edge_opt.model import SmallCNN


def test_analytical_model_placeholder() -> None:
    model = SmallCNN()
    hw_model = AnalyticalHardwareModel(memory_bandwidth_gbps=10.0)
    assert hw_model is not None
    assert model is not None
