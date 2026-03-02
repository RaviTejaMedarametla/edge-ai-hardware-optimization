# Edge AI Hardware Optimization (Low Power)

This project is a stable, practical baseline for **hardware-aware AI optimization** on edge and semiconductor targets.
It keeps implementation simple while modeling realistic deployment constraints: SRAM limits, low-power CPU operation, latency/energy tradeoffs, and precision-aware compression.

## What this pipeline covers

1. Train a compact CNN on MNIST or Fashion-MNIST.
2. Record baseline accuracy, latency, throughput, memory, and energy proxy.
3. Apply **structured channel pruning**.
4. Evaluate **FP32**, **FP16**, and **INT8** variants.
5. Simulate SRAM-constrained deployment with configurable budgets (1MB / 2MB / 4MB).
6. Reject models above the active memory budget and report violations per budget.
7. Simulate low-power CPU frequency scaling with an explicit latency multiplier.
8. Sweep pruning and precision to generate tradeoff and Pareto analyses.
9. Simulate batch vs streaming inference on CPU.

## Why this matters for edge silicon

Real edge hardware is constrained by:
- **on-chip memory** (SRAM),
- **power envelope**,
- **inference latency requirements**.

High accuracy alone is not enough. Models must satisfy memory and timing limits while preserving acceptable output quality. This repository demonstrates a simple but realistic path for making those tradeoffs measurable and reproducible.

## Metrics and constraints

For every candidate model:
- **Accuracy**
- **Latency (ms)**
- **Throughput (samples/s)**
- **Memory footprint (MB)**
- **Energy proxy (J)** = `latency × configured power`

Constraint logic:
- Report violations against each configured memory budget.
- Mark candidates as rejected if they exceed `active_memory_budget_mb`.
- Exclude rejected candidates from Pareto frontier generation.

## Visual outputs

Generated under `outputs/`:
- `accuracy_vs_latency.png`
- `accuracy_vs_energy.png`
- `accuracy_vs_memory.png`
- `pareto_frontier_latency.csv`
- `pareto_frontier_energy.csv`

Rejected (memory-violating) candidates are shown separately in the plots.

## Project structure

- `scripts/run_pipeline.py` – end-to-end pipeline runner
- `configs/default.yaml` – deterministic and constraint-aware settings
- `src/edge_opt/model.py` – compact CNN + deterministic seed setup
- `src/edge_opt/pruning.py` – structured channel pruning
- `src/edge_opt/quantization.py` – FP16 + INT8 conversion
- `src/edge_opt/metrics.py` – accuracy, latency, throughput, memory, energy, violations
- `src/edge_opt/experiments.py` – sweep execution + plots + Pareto frontiers
- `src/edge_opt/deploy.py` – batch/streaming deployment simulation + CPU scaling

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/run_pipeline.py --config configs/default.yaml
```

## Configuration notes

Key options in `configs/default.yaml`:
- `memory_budgets_mb`: list of SRAM budgets to report against.
- `active_memory_budget_mb`: hard budget used to accept/reject models.
- `cpu_frequency_scale`: low-power CPU factor (e.g., `0.7` means slower clock and higher effective latency).

Keep this project intentionally simple: adjust config first before adding new infrastructure.
