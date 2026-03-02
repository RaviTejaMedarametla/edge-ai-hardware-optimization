# Architecture Notes

## Pipeline stages

1. **Configuration load** (`edge_opt.config`): parse YAML into a typed dataclass.
2. **Dataset and loader setup** (`edge_opt.data`): build deterministic train/validation loaders.
3. **Baseline training** (`edge_opt.experiments.train_model`): train compact CNN.
4. **Optimization sweep** (`edge_opt.experiments.run_sweep`): apply pruning and precision variants.
5. **Metric collection** (`edge_opt.metrics`): compute accuracy, latency, throughput, memory, and energy proxy.
6. **Constraint filtering**: classify candidates by active memory budget.
7. **Reporting**: save sweep tables, Pareto frontiers, plots, and summary JSON.

## Design decisions

- A compact CNN is used to keep iteration cycle times short while retaining realistic convolutional operator behavior.
- Structured pruning removes whole channels to preserve dense kernels and straightforward deployment compatibility.
- Precision conversion is explicit (`fp32`, `fp16`, `int8`) to keep evaluation paths auditable.
- Pareto frontier generation is performed after constraint filtering to avoid infeasible configurations.

## Operational constraints

- CPU execution only in the default pipeline.
- No distributed training support.
- Quantization backend defaults to `fbgemm`.

## Deployment challenges

- Batch-size sensitivity can mask single-request latency behavior.
- Memory headroom margins in production typically require tighter limits than nominal model-size estimates.
- Host-level contention (co-scheduled workloads, thermal throttling) can significantly alter latency distributions.

## Recommended extensions

- Add multi-seed experiment orchestration and confidence intervals.
- Integrate hardware counters for cache, bandwidth, and instruction-level profiling.
- Introduce artifact manifests with model checksum and dataset version metadata.
