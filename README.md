# anml-exp
A unified interface to prototype and test anomaly detection methods.

## Project Structure

```
.
├── src/
│   └── anml_exp/
│       ├── models/
│       ├── data/
│       ├── benchmarks/
│       ├── registry/
│       ├── resources/
│       └── cli.py
├── tests/
├── docs/
├── uv.lock
```

## Installation

Create a reproducible environment with `uv` and install optional extras as needed:

```bash
uv sync --frozen
pip install -e ".[torch,plot]"
```

## Benchmarking

Run the `anml-exp` command to execute a benchmark and produce a JSON result conforming to `anml_exp/resources/results-schema.json`.

```bash
anml-exp benchmark \
    --dataset toy-blobs \
    --model isolation_forest \
    --output results/example.json
```

## Leaderboard

Run the `leaderboard` command to benchmark all built-in tabular datasets and generate a Markdown leaderboard. Results are stored under `results/leaderboard`.

```bash
anml-exp leaderboard --hardware '{"device_type":"CPU","vendor":"unknown","model":"unknown","driver":"N/A","num_devices":1,"notes":"example"}'
```

## Hardware descriptor

Use the `--hardware` flag to record the execution environment. It accepts a short string or JSON object with keys `device_type`, `vendor`, `model`, `driver`, `num_devices` and `notes`.

## Artefact registry

Models may be saved and loaded using `anml_exp.registry.Registry`, which stores pickled artefacts under a semantic version and verifies their SHA-256 digest.
