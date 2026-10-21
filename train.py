"""Train and validate models in memory; no model files are written.

Python:
    from train import train, validate
    model = train("datasets/cooling_regression/train.csv")
    fold_scores, summary = validate("datasets/cooling_regression/train.csv")

Input: a labeled CSV with metadata.json in the same directory.
Output: a bundle containing the fitted model, preprocessing, and metadata.
Configure library models in make_model; implement custom training in fit_model.
Pass the returned bundle directly to test(data_path, model) in the same process.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def make_model(dataset_name, method="baseline", random_state=2026):
    """Return a fresh, unfitted estimator that includes learned preprocessing."""
    regression = {"sarcos_regression", "cooling_regression"}
    classification = {"credit_classification", "execution_classification"}
    if dataset_name not in regression | classification:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ==================== TODO START ====================


    # Define preprocessing, features, models, and hyperparameters here.
    # Add other methods, such as "bagging" and "improved".
    # Keep learned preprocessing in the pipeline.
    if method != "baseline":
        raise NotImplementedError(f"Implement {method!r} in make_model.")

    estimator = (Ridge(alpha=1.0) if dataset_name in regression else
                 LogisticRegression(C=1.0, max_iter=2000, random_state=random_state))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


    # ===================== TODO END =====================


def fit_model(X, y, dataset_name, method="baseline", random_state=2026):
    """Train on X and y and return the fitted model, including preprocessing.

    Use method="custom" to implement an algorithm without a library estimator.
    Each call must create fresh model and preprocessing state.
    """
    if method == "custom":
        # ==================== TODO START ====================


        # Implement training here using X (features) and y (targets).
        # Fit preprocessing only on X and keep its fitted state with the model.
        # A custom model does not need a fit() method: training happens here.

        raise NotImplementedError("Implement custom training and return a fitted model.")


        # ===================== TODO END =====================

    model = make_model(dataset_name, method, random_state=random_state)
    # For library models, fit() runs their training algorithm. Pipeline.fit()
    # first fits preprocessing, then trains the estimator on transformed inputs.
    model.fit(X, y)
    return model


def train(data_path, method="baseline", seed=2026, *, rows=None):
    """Return a fitted model bundle without saving it.

    rows optionally selects CSV rows by position; validate() uses it to
    train each fold on its training observations only. None uses all rows.
    """
    data_path = Path(data_path)
    metadata = json.loads((data_path.parent / "metadata.json").read_text())
    features, target = metadata["feature_names"], metadata["target"]
    task = metadata["task"]
    if task not in {"regression", "binary_classification"}:
        raise ValueError(f"Unsupported task: {task}")
    if {"sample_id", "fold", target}.intersection(features):
        raise ValueError("Identifiers, folds, and the target cannot be features.")
    data = pd.read_csv(data_path, dtype={"sample_id": str})
    if rows is not None:
        data = data.iloc[rows]
    data = data.reset_index(drop=True)
    # Exclude identifiers, fold assignments, and targets from model inputs.
    X = data.loc[:, features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data[target], errors="raise").to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all() or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Use nonempty data, finite targets, and no infinite features.")
    if task == "binary_classification" and set(y) != {0, 1}:
        raise ValueError("Binary classifier training requires both classes 0 and 1.")

    model = fit_model(X, y, metadata["dataset"], method, random_state=seed)
    return {"model": model, "metadata": metadata, "method": method, "seed": seed}


def _validation_ids(frame, description):
    if "sample_id" not in frame:
        raise ValueError(f"{description} must contain sample_id.")
    ids = frame["sample_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError(f"{description} must have unique, nonblank sample IDs.")
    return ids


def validate(data_path, method="baseline", seed=2026, train_size=None):
    """Return per-fold training/validation metrics and their means and SDs.

    train_size reduces training rows only; validation rows stay fixed.
    Mean and sample standard deviation (ddof=1) summarize the five folds.
    Original CSV files are unchanged, and no model is saved or loaded.
    """
    # Import here so training and evaluation modules can be used independently.
    import test as testing

    data_path = Path(data_path)
    metadata = json.loads((data_path.parent / "metadata.json").read_text())
    task = metadata["task"]
    if task not in {"regression", "binary_classification"}:
        raise ValueError(f"Unsupported task: {task}")
    if train_size is not None and (
        isinstance(train_size, bool)
        or not isinstance(train_size, (int, np.integer)) or train_size < 1
    ):
        raise ValueError("train_size must be a positive integer or None.")

    options = {"dtype": {"sample_id": str}, "keep_default_na": False}
    data = pd.read_csv(data_path, **options)
    folds = pd.read_csv(data_path.parent / "cv_folds.csv", **options)
    if list(folds.columns) != ["sample_id", "fold"]:
        raise ValueError("cv_folds.csv must contain exactly sample_id and fold.")
    ids, fold_ids = _validation_ids(data, "Data"), _validation_ids(folds, "Fold assignments")
    if set(ids) != set(fold_ids):
        raise ValueError("Data and fold assignments must contain identical sample IDs.")
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise")
    if set(folds["fold"]) != set(range(5)):
        raise ValueError("Fold assignments must contain exactly labels 0 through 4.")
    fold_id = ids.map(folds.set_index("sample_id")["fold"]).to_numpy(dtype=int)
    y = None
    if task == "binary_classification":
        y = pd.to_numeric(data[metadata["target"]], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all() or set(y) != {0, 1}:
            raise ValueError("Binary classification data must contain both classes 0 and 1.")

    rows = []
    for fold in range(5):
        train_rows = np.flatnonzero(fold_id != fold)
        validation_rows = np.flatnonzero(fold_id == fold)
        if train_size is not None:
            if train_size > len(train_rows):
                raise ValueError(f"train_size exceeds the {len(train_rows)} training rows in fold {fold}.")
            if train_size < len(train_rows):
                train_rows = resample(
                    train_rows, replace=False, n_samples=int(train_size),
                    random_state=seed + fold,
                    stratify=None if y is None else y[train_rows],
                )
        if y is not None and set(y[train_rows]) != {0, 1}:
            raise ValueError("Each classifier training subset must contain both classes; increase train_size.")
        model = train(data_path, method=method, seed=seed, rows=train_rows)
        row = {"fold": fold, "n_train": len(train_rows), "n_validation": len(validation_rows)}
        for split, indices in (("train", train_rows), ("validation", validation_rows)):
            scores = testing.test(data_path, model, rows=indices)["metrics"]
            if scores is None:
                raise ValueError("Cross-validation requires labeled data.")
            row.update({f"{split}_{metric}": value for metric, value in scores.items()})
        rows.append(row)
    fold_scores = pd.DataFrame(rows)
    metrics = [column for column in fold_scores if column.startswith(("train_", "validation_"))]
    summary = fold_scores[metrics].agg(["mean", "std"]).T.rename_axis("metric")
    return fold_scores, summary


# For cross-validation, pass only the training rows and fit a fresh model per fold.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training without saving a model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    model = train(args.data, method=args.method, seed=args.seed)
    print(f"Trained {model['metadata']['dataset']} ({args.method}) in memory.")
    print("Use train() and test() in the same Python session to keep and evaluate it.")
