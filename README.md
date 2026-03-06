# Edge AI Hardware Optimization

## Overview
This repository implements a hardware-aware machine learning pipeline for evaluating compact convolutional models under memory and latency constraints on edge-class systems. The project is maintained as part of a broader AI systems engineering portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

## System Architecture
The pipeline follows a deterministic, config-driven workflow:
1. Load configuration and reproducibility controls.
2. Train a baseline compact CNN.
3. Apply structured channel pruning across configured sweep levels.
4. Evaluate precision variants (FP32, FP16, INT8).
5. Enforce memory budget constraints.
6. Benchmark latency, throughput, memory footprint, and energy proxy.
7. Export tabular artifacts and Pareto frontiers.

Core components are organized under `src/edge_opt/`:
- `model.py`: model definition and construction.
- `experiments.py`: training/evaluation orchestration.
- `pruning.py` and `quantization.py`: optimization passes.
- `hardware.py` and `metrics.py`: systems-oriented measurements.
- `deploy.py` and `reporting.py`: deployment analysis and artifact generation.

## Features
- Deterministic experiment controls through YAML configuration.
- Structured pruning sweeps with precision-aware evaluation.
- Constraint-first filtering for memory-limited deployment scenarios.
- Latency/throughput/memory/energy-proxy reporting.
- Pareto frontier generation for latency-accuracy and energy-accuracy trade-offs.
- Layer-wise hardware-analysis outputs for bottleneck inspection.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Usage
Run the default benchmark pipeline:
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Artifacts are written to the configured `output_dir`, including sweep tables, frontier CSV files, summary metadata, and analysis plots.

## Reproducibility
Reproducibility controls are configured in `configs/default.yaml`, including:
- global seed values
- dataloader seed settings
- deterministic worker configuration
- repeated benchmark windows for variability reporting

To preserve benchmark comparability across runs, keep dataset subset sizing, batch size, memory budget, and seed settings consistent.

## Related Projects
This repository is part of a broader portfolio of AI systems engineering projects:
- `neural-network-systems`
- `digit-classification-benchmark`
- `edge-ai-model-optimization`
- `hospital-analytics-pipeline`
- `nba-data-engineering`
- `ai-systems-ml-platform`

## Repository Naming Note
Professional rename suggestion (manual GitHub operation, not applied automatically):
- `edge-ai-hardware-optimization` → `edge-ai-model-optimization`
