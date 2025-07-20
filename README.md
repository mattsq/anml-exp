# anml-exp
A unified interface to prototype and test anomaly detection methods.

## Benchmarking

Run ``evaluator.py`` to execute a benchmark and produce a JSON result
conforming to ``results/results-schema.json``.

```bash
python evaluator.py \
    --dataset toy-blobs \
    --model isolation_forest \
    --output results/example.json
```
