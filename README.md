# Edge AI Hardware Optimization

Reference pipeline for evaluating compact CNN deployments under edge-device constraints.
The repository prioritizes deterministic execution, measurable trade-offs, and low-complexity implementation suitable for iterative experimentation.

## Scope

The pipeline trains and evaluates a compact CNN on MNIST-family datasets and then applies constrained deployment analysis:

1. Train a baseline model.
2. Sweep structured channel-pruning levels.
3. Evaluate FP32, FP16, and INT8 variants.
4. Enforce SRAM-style memory budgets and mark rejected candidates.
5. Measure latency, throughput, memory footprint, and energy proxy.
6. Generate Pareto frontiers for latency-accuracy and energy-accuracy.
7. Simulate lower CPU frequency operation and stream-vs-batch inference.

## System design motivations

- **Single-model family**: A fixed network topology isolates pruning and precision effects from architecture search noise.
- **Constraint-first filtering**: Memory budget checks run before frontier analysis so infeasible candidates do not distort operating-point selection.
- **CPU-focused execution path**: The default environment reflects common edge integration constraints where accelerator access may be limited.
- **Config-driven workflow**: Most experiment knobs are externalized in YAML to support repeatable benchmark sweeps.

## Architectural trade-offs

- **Accuracy vs latency**: More aggressive pruning often reduces latency and memory at the cost of representational capacity.
- **Precision vs stability**: FP16 and INT8 can reduce memory/compute cost, but quantization calibration quality directly affects accuracy retention.
- **Determinism vs speed**: Deterministic settings improve reproducibility but can reduce backend autotuning opportunities.
- **Simple metrics vs full hardware counters**: Current latency and energy outputs are software-level estimates; hardware counters must be integrated separately for final silicon sign-off.

## Performance model and constraints

Per candidate, the pipeline reports:

- Accuracy
- Latency (ms)
- Throughput (samples/s)
- Model memory footprint (MB)
- Energy proxy (J): `latency_seconds * power_watts`

Constraint handling:

- `memory_budgets_mb`: reported as per-budget violation flags.
- `active_memory_budget_mb`: hard acceptance threshold used in sweep summaries and Pareto filtering.

## Failure modes and bottlenecks

- Dataset download or cache corruption can block startup.
- DataLoader worker settings may be suboptimal on low-core hosts.
- INT8 calibration with too few batches can bias activation ranges.
- CPU frequency scaling is modeled, not measured from hardware telemetry.
- Throughput estimates are sensitive to selected batch size and host load.

## Scalability considerations

- Sweep cardinality scales with `len(pruning_levels) * len(precisions)`.
- Increasing subset sizes improves statistical confidence but increases runtime.
- Additional architectures can be introduced without changing CLI behavior by extending internal model factories.

## Assumptions

- CPU-only execution path is representative of target deployment constraints.
- Power draw is provided as a fixed scalar in configuration.
- Model size from state dict is an acceptable first-order memory proxy.
- MNIST/Fashion-MNIST are used as controlled benchmarking datasets.

## Limitations

- Hardware analysis is estimate-based and does not replace PMU-level profiling.
- No confidence intervals or multi-seed aggregation by default.
- Operator-level profiling is not emitted unless explicitly enabled.
- Quantization path is calibrated on training data loader samples.


## Benchmarking rigor

- Latency is measured across repeated benchmark windows (`benchmark_repeats`) and reported with mean, standard deviation, and p95 in output artifacts.
- Sweep comparisons should use the same dataset subset, batch size, and memory budget settings to avoid cross-run drift.
- For publication-grade claims, run multiple seeds and aggregate externally.

## Reproducibility controls

`configs/default.yaml` includes deterministic controls:

- `seed`: global model/training randomness control.
- `dataloader_seed`: loader shuffle seed.
- `num_workers`: loader worker count for stable host behavior.
- `benchmark_repeats`: repeated latency windows for variability reporting.

## Repository layout

- `scripts/run_pipeline.py` — CLI entry point.
- `configs/default.yaml` — deterministic baseline configuration.
- `src/edge_opt/` — model, pruning, quantization, metrics, sweep, and deployment modules.
- `docs/` — architecture and hardware-analysis notes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/run_pipeline.py --config configs/default.yaml
```

## Output artifacts

Generated in `output_dir`:

- `sweep_results.csv`
- `pareto_frontier_latency.csv`
- `pareto_frontier_energy.csv`
- `summary.json`
- `layerwise_breakdown.csv`
- `precision_tradeoffs.csv`
- `hardware_summary.csv`
- `accuracy_vs_latency.png`
- `accuracy_vs_energy.png`
- `accuracy_vs_memory.png`
- `layerwise_activation_memory.png`
- `layerwise_macs.png`


## Hardware-aware outputs

- `memory_bandwidth_gbps` in config is used to estimate bandwidth utilization from measured latency.
- Layer-wise tables capture activation and parameter footprints to highlight bottleneck layers.
- Precision summary tables show mean behavior and acceptance ratio under memory limits.
