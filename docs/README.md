# Documentation

## Loading datasets

Use :func:`datasets.registry.load_dataset` to obtain deterministic splits. The
example below fetches the synthetic ``toy-blobs`` data::

    from datasets.registry import load_dataset

    X_train, _ = load_dataset("toy-blobs", split="train")
    X_test, y_test = load_dataset("toy-blobs", split="test")

## Running the evaluator

The benchmark runner can be invoked from the command line::

    python evaluator.py \
        --dataset toy-blobs \
        --model isolation_forest \
        --output results/example.json

## Available models

- ``isolation_forest``
- ``local_outlier_factor``
- ``one_class_svm``

This writes a JSON file compatible with ``results/results-schema.json``.

## Available datasets

The following dataset identifiers can be passed to
``datasets.registry.load_dataset``:

- ``toy-blobs``
- ``toy-circles``
- ``breast-cancer``
