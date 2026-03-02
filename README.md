# Edge AI Hardware Optimization (Low Power)

A simple and stable project for realistic edge and semiconductor deployment tradeoff studies.

## Focus

This pipeline models practical low-power constraints while staying easy to run and modify:
- SRAM memory budgets
- energy-aware acceptance rules
- dynamic architecture scaling
- pruning and quantization search
- CPU low-frequency simulation

## Low-power model search

The search module evaluates candidate models built from:
1. **Dynamic architecture scaling** (`width_multipliers`)
2. **Structured channel pruning** (`pruning_levels`)
3. **Precision modes** (`fp32`, `fp16`, `int8`)

Each candidate is measured on:
- accuracy
- latency
- throughput
- memory footprint
- energy proxy (`latency × power`)

Candidates are accepted only if they satisfy both:
- active SRAM budget (`active_memory_budget_mb`)
- energy constraint (`max_energy_j`)

## Visualisation

Outputs include:
- `accuracy_vs_latency.png`
- `accuracy_vs_energy.png`
- `accuracy_vs_memory.png`
- `pareto_frontier_latency.csv`
- `pareto_frontier_energy.csv`
- `feasible_results.csv`

Plots separate accepted and rejected candidates for clear decision making.

## Real-world deployment angle

This mirrors common edge hardware decisions:
- fit within limited SRAM,
- keep energy and latency inside product constraints,
- choose the highest-accuracy feasible model rather than highest-accuracy overall.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/run_pipeline.py --config configs/default.yaml
```

## Key config knobs

- `memory_budgets_mb`: reported SRAM limits
- `active_memory_budget_mb`: hard memory cutoff
- `max_energy_j`: hard energy cutoff
- `cpu_frequency_scale`: low-power CPU scaling factor
- `width_multipliers`: dynamic architecture scaling factors
- `pruning_levels`: pruning sweep points
- `precisions`: precision sweep points
