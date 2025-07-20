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

## Leaderboard

Run ``leaderboard.py`` to benchmark all built-in tabular datasets and generate a
Markdown leaderboard. Results are stored under ``results/leaderboard``.

```bash
python leaderboard.py --hardware CPU-unknown
```
