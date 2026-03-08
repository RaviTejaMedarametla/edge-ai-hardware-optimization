# Research-Grade Improvement Plan

This document captures concrete repository changes to improve scientific credibility and engineering reliability.

## 1) Replace custom YAML parser with `yaml.safe_load` + schema validation

### Change
- Replaced ad-hoc line parsing in `src/edge_opt/config.py` with `yaml.safe_load`.
- Added strict dataclass validation in `ExperimentConfig.__post_init__` with explicit range/type checks.

### Why this improves credibility
- Eliminates parser edge cases and silent misparsing.
- Fails fast on invalid configurations, which supports reproducibility and auditability.

## 2) Remove hardcoded architecture assumptions

### Change
- `SmallCNN` now uses adaptive global pooling (`AdaptiveAvgPool2d((1,1))`), decoupling classifier input from fixed `28x28` assumptions.
- `structured_channel_prune` now remaps classifier channels directly, removing hardcoded `7*7` flatten logic.
- `estimate_layerwise_stats` no longer uses fixed 4-byte values; it derives parameter bytes from tensor `element_size()` and accepts activation byte width.

### Why this improves credibility
- Enables fairer comparison across input sizes and precision settings.
- Reduces hidden assumptions that can invalidate benchmark conclusions.

## 3) Add core unit tests

### Change
- Added unit tests for config loading/validation, pruning remapping behavior, Pareto filtering, sweep error capture, metrics CI fields, and hardware byte accounting.

### Why this improves credibility
- Guards against regression in research-critical metrics and data processing.
- Makes refactoring safer and supports reproducible claims.

## 4) Add per-variant error handling in sweep

### Change
- Wrapped each pruning/precision evaluation in `try/except`.
- Errors are recorded per row (`error` column) while continuing remaining sweep combinations.

### Why this improves credibility
- Prevents one failed quantization path from discarding all experiment evidence.
- Makes failure modes explicit in outputs.

## 5) Remove unused dependencies

### Change
- Removed `onnx` and `onnxruntime` from `requirements.txt`.
- Added `pytest` for test automation.

### Why this improves credibility
- Cleaner dependency graph and lower environment ambiguity.
- Dependencies now directly map to implemented functionality.

## 6) Document energy as a proxy metric

### Change
- `PerfMetrics` now includes `energy_proxy_note` explicitly stating metric semantics:
  - `power_watts × latency_s`
  - Not measured on-device power.

### Why this improves credibility
- Prevents overclaiming physical power accuracy.
- Aligns with research transparency best practices.

## 7) Improve statistical rigor

### Change
- Added repeated accuracy trials with standard deviation and approximate 95% confidence interval fields.
- Configurable via `benchmark_trials`.

### Why this improves credibility
- Replaces single-point accuracy reporting with uncertainty-aware summaries.
- Improves reliability of performance comparisons.

## 8) Add CI automation

### Change
- Added GitHub Actions workflow `.github/workflows/ci.yml` running `pytest` on push/PR.

### Why this improves credibility
- Enforces baseline quality checks continuously.
- Improves trust that reported pipeline behavior is maintained over time.

## Trade-offs

- Added validation and repeated trials introduce modest runtime overhead.
- Adaptive pooling changes model inductive bias and may slightly alter baseline accuracy trends.
- Sweep-level exception capture requires downstream analysis to consider `error` rows explicitly.
