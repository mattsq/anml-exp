# Documentation

## Project structure

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

Create the Python environment from the lock file before installing the project:

```bash
uv sync --frozen
pip install -e ".[torch,plot]"
```

## Loading datasets

Use :func:`anml_exp.data.load_dataset` to obtain deterministic splits. The example below fetches the synthetic ``toy-blobs`` data::

    from anml_exp.data import load_dataset

    X_train, _ = load_dataset("toy-blobs", split="train")
    X_test, y_test = load_dataset("toy-blobs", split="test")

## Running the evaluator

Invoke the CLI with the ``benchmark`` subcommand::

    anml-exp benchmark \
        --dataset toy-blobs \
        --model isolation_forest \
        --output results/example.json

## Available models

- ``isolation_forest``
- ``local_outlier_factor``
- ``one_class_svm``
- ``autoencoder``
- ``pca``
- ``deep_svdd``
- ``usad``

This writes a JSON file compatible with ``anml_exp/resources/results-schema.json``.

## Leaderboard

Aggregate results for all built-in tabular datasets::

    anml-exp leaderboard --hardware '{"device_type":"CPU","vendor":"unknown","model":"unknown","driver":"N/A","num_devices":1,"notes":"example"}'

## Hardware descriptor format

The ``--hardware`` argument accepts either a short string or a JSON object with the following fields:

```
{
  "device_type": "GPU",
  "vendor": "NVIDIA",
  "model": "RTX A6000",
  "driver": "535.104",
  "num_devices": 1,
  "notes": "desktop workstation"
}
```

These keys are stored in the results JSON under the ``hardware`` section.

## Artefact registry

Models can persist their state to the filesystem using ``anml_exp.registry.Registry``::

    from anml_exp.registry import Registry

    registry = Registry("./model_store")
    model.save(registry, "isolation_forest", "0.1.0")
    loaded = type(model).load(registry, "isolation_forest", "0.1.0")

The registry verifies SHA-256 digests on save and load.

## Available datasets

The following dataset identifiers can be passed to ``anml_exp.data.load_dataset``:

- ``toy-blobs``
- ``toy-circles``
- ``breast-cancer``
- ``wine``
- ``digits``
