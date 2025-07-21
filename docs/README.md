# Documentation

## Installation

Create the Python environment from the lock file before installing the project:

```bash
uv pip sync -r uv.lock
pip install -e ".[torch,plot]"
```

## Loading datasets

Use :func:`anml_exp.data.load_dataset` to obtain deterministic splits. The
example below fetches the synthetic ``toy-blobs`` data::

    from anml_exp.data import load_dataset

    X_train, _ = load_dataset("toy-blobs", split="train")
    X_test, y_test = load_dataset("toy-blobs", split="test")

## Running the evaluator

The benchmark runner can be invoked from the command line::

    python -m anml_exp.cli benchmark \
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

Use the ``leaderboard`` command to benchmark all built-in tabular datasets and aggregate the results into a Markdown table::

    python -m anml_exp.cli leaderboard --hardware '{"device_type":"CPU","vendor":"unknown","model":"unknown","driver":"N/A","num_devices":1,"notes":"example"}'


## Available datasets

The following dataset identifiers can be passed to
``anml_exp.data.load_dataset``:

- ``toy-blobs``
- ``toy-circles``
- ``breast-cancer``
- ``wine``
- ``digits``
