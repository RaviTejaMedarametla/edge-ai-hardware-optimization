# Hardware Analysis Methodology

## Objective

This repository provides first-order hardware-aware estimates for memory pressure, compute density, and precision trade-offs without changing the baseline model API.

## Layer-wise breakdown

`layerwise_breakdown.csv` reports for `conv1`, `conv2`, and `classifier`:

- output activation elements
- parameter bytes
- activation bytes
- MAC estimates

These values are shape-derived estimates from the baseline batch configuration and are intended for relative comparison.

## Bandwidth and utilization estimate

`hardware_summary.csv` derives:

- estimated total bytes moved (parameters + activations)
- estimated total MACs
- achieved bandwidth approximation (`bytes / latency`)
- configured memory bandwidth (`memory_bandwidth_gbps`)
- bandwidth utilization ratio
- achieved GMAC/s estimate

## Precision and quantization trade-off table

`precision_tradeoffs.csv` aggregates sweep results by precision mode:

- mean accuracy
- mean latency
- mean memory footprint
- mean energy proxy
- acceptance ratio under active memory budget

## Failure modes and caveats

- Estimates do not include cache-miss penalties or kernel launch overhead details.
- INT8 execution path may vary by backend implementation and calibration data quality.
- CPU host contention can significantly affect measured latency and derived utilization.
- Activation memory reported is per-layer output footprint and not full runtime peak memory.

## Edge and constrained scenarios

For low-memory systems, reduce `active_memory_budget_mb` and compare acceptance ratio changes in `precision_tradeoffs.csv`.
For bandwidth-constrained studies, lower `memory_bandwidth_gbps` to stress utilization estimates and identify compute- vs transfer-bound regions.
