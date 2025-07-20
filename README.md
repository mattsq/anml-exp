# anml-exp
A unified interface to prototype and test anomaly detection methods.

## Benchmarking

Run ``anml_exp.cli`` to execute a benchmark and produce a JSON result
conforming to ``anml_exp/resources/results-schema.json``.

```bash
python -m anml_exp.cli benchmark \
    --dataset toy-blobs \
    --model isolation_forest \
    --output results/example.json
```

## Leaderboard

Run the ``leaderboard`` command to benchmark all built-in tabular datasets and
generate a Markdown leaderboard. Results are stored under ``results/leaderboard``.

```bash
python -m anml_exp.cli leaderboard --hardware CPU-unknown
```
