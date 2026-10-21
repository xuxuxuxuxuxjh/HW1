"""Evaluate an in-memory model with standard metrics; no model files are read.

Python: report = test("validation.csv", model)
Terminal: python test.py --train training.csv --data validation.csv

Input: a CSV and the model bundle returned by train() in the same process.
Output: a dictionary containing dataset, task, n_samples, and metrics.
Feature and target names come from the in-memory bundle.
Editing predict() is optional when your model supports the interface below.
Five-fold validation is provided by validate() in train.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics

__test__ = False  # Prevent pytest from collecting this evaluation entry point.


def predict(model, X, task):
    """Return one scalar prediction or P(Y=1) for every row, in the same order."""

    # ==================== TODO START (OPTIONAL) ====================


    # Leave this block unchanged for models with predict(X), or for classifiers with predict_proba(X) and classes_.

    if task == "regression":
        return model.predict(X)
    # Probability column order is determined by classes_, not an assumption.
    classes = np.asarray(model.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Expected a classifier with classes 0 and 1.")
    return model.predict_proba(X)[:, np.flatnonzero(classes == 1)[0]]


    # ===================== TODO END (OPTIONAL) =====================


# Keep metric definitions unchanged for consistent evaluation.
def test(data_path, model, output_path=None, predictions_path=None, *, rows=None):
    """Evaluate without fitting or loading a model; unlabeled data have metrics=None.

    rows optionally selects CSV rows by position for train.validate().
    The output paths save scores or predictions only, never the model.
    """
    if not isinstance(model, dict) or not {"model", "metadata"}.issubset(model):
        raise TypeError("Pass the model bundle returned by train(), not a filename.")
    bundle = model
    metadata = bundle["metadata"]
    task, target = metadata["task"], metadata["target"]
    if task not in {"regression", "binary_classification"}:
        raise ValueError(f"Unsupported task: {task}")
    protected = {Path(data_path).resolve()}
    outputs = [Path(p).resolve() for p in (output_path, predictions_path) if p is not None]
    if protected.intersection(outputs) or len(outputs) != len(set(outputs)):
        raise ValueError("Output files must be distinct from the input and each other.")

    # Preserve literal identifiers such as 001 and NA when exporting predictions.
    data = pd.read_csv(data_path, keep_default_na=False, dtype={"sample_id": str})
    if rows is not None:
        data = data.iloc[rows]
    data = data.reset_index(drop=True)
    X = data.loc[:, metadata["feature_names"]].replace(
        ["", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"], np.nan
    ).apply(pd.to_numeric, errors="raise")
    if not len(X) or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Evaluation data must be nonempty and contain no infinite features.")
    # Prediction receives features only, never test labels.
    p = np.asarray(predict(bundle["model"], X, task), dtype=np.float64)
    if p.shape != (len(X),) or not np.isfinite(p).all():
        raise ValueError("Return one finite prediction per input row.")
    if task == "binary_classification" and ((p < 0) | (p > 1)).any():
        raise ValueError("Class-1 probabilities must be in [0, 1].")

    scores = None  # Without true labels, error cannot be calculated.
    if target in data:
        y = pd.to_numeric(data[target], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Evaluation targets must be finite.")
        if task == "regression":
            mse = float(metrics.mean_squared_error(y, p))
            scores = {"mse": mse, "rmse": float(np.sqrt(mse)),
                      "mae": float(metrics.mean_absolute_error(y, p)),
                      "r2": float(metrics.r2_score(y, p)) if len(y) > 1 else None}
        else:
            if not set(y).issubset({0, 1}):
                raise ValueError("Classification targets must be 0 or 1.")
            # Fixed conventions: threshold 0.5; float64 log-loss clipping at 1e-15.
            labels = (p >= 0.5).astype(int)
            scores = {"log_loss": float(metrics.log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1])),
                      "roc_auc": float(metrics.roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                      "accuracy": float(metrics.accuracy_score(y, labels)),
                      "f1": float(metrics.f1_score(y, labels, zero_division=0))}
        if any(v is not None and not np.isfinite(v) for v in scores.values()):
            raise ValueError("A metric overflowed; check predictions and target magnitudes.")
    report = {"dataset": metadata["dataset"], "task": task, "n_samples": len(X), "metrics": scores}
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if predictions_path is not None:
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": data["sample_id"], "prediction": p}).to_csv(predictions_path, index=False)
    return report


# Use held-out labeled rows to measure generalization, not the training rows.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train once, then evaluate in the same process.")
    parser.add_argument("--train", required=True, help="Labeled training CSV")
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    parser.add_argument("--predictions")
    args = parser.parse_args()
    # Training uses --train; --data is used only for prediction and scoring.
    from train import train
    model = train(args.train, method=args.method, seed=args.seed)
    report = test(args.data, model, args.output, args.predictions)
    print(json.dumps(report, indent=2, allow_nan=False))
