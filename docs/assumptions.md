# Assumptions and Limitations

This document clarifies the approximations made in hardware-aware metrics.

## Memory Estimation
- **Parameter memory**: Exact (from `state_dict`).
- **Activation memory**: Estimated as the sum of all layer outputs during a forward pass. This represents an **upper bound** because tensors may be reused (e.g., in-place operations) or freed early. For peak memory, we provide a separate `peak_activation_memory` function that also approximates but is more realistic.

## Latency
- Measured on CPU using PyTorch's benchmark utility or manual timing. This reflects software execution on the host machine, not on target edge hardware. Scaling via `cpu_frequency_scale` is a linear approximation and does not account for cache effects, pipeline stalls, etc.

## Energy Proxy
- `energy_proxy_j = power_watts × latency_s`. This is a linear proxy assuming constant power draw, which is unrealistic for real hardware (power varies with utilization). Use only for relative comparisons within the same experiment.

## Roofline Model
- The roofline analysis assumes a simple memory-bound / compute-bound dichotomy based on arithmetic intensity. The knee point is computed as `peak_compute / memory_bandwidth`. This is a first-order approximation and does not consider complex memory hierarchies or bandwidth contention.

## Quantization Simulation
- Our fake-quantization simulates integer arithmetic by quantizing and dequantizing activations. It does **not** simulate quantized kernels; it merely provides a proxy for accuracy under quantization. For true hardware performance, use a real quantized backend (e.g., PyTorch's `fbgemm`).

## Pruning
- Structured pruning removes entire channels/filters based on L1-norm. This may break residual connections or complex topologies. The current implementation assumes a feed-forward graph; use with caution for non-sequential models.
