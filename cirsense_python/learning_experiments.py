import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import Iterable

import numpy as np

from baselines import PCABaseline, PCAClassifier
from config import ProjectConfig, build_default_config
from csi_to_cir import build_projection_matrix
from dataset_manifest import build_manifest, save_manifest_csv
from evaluate import classification_metrics, regression_metrics, save_rows_csv
from learning_data import (
    LearningDataset,
    class_label,
    file_level_split,
    load_learning_dataset,
    prepare_learning_dataset,
    save_learning_dataset,
)
from models import (
    build_cnn_enhanced_multitask,
    build_cnn_lstm_enhanced_multitask,
    build_cnn_lstm_multitask,
    build_cnn_multitask,
    build_dnn_enhanced_multitask,
    build_multiview_fusion_multitask,
)
from train import predict_multitask_model, train_multitask_model
from visualize import plot_confusion_matrix, plot_loss_curve, plot_predictions


DEFAULT_METHODS = ("pca_svm", "pca_knn", "cnn", "cnn_lstm")
ALL_METHODS = DEFAULT_METHODS + (
    "cnn_enhanced",
    "cnn_lstm_enhanced",
    "dnn_enhanced",
    "dnn_ensemble",
    "dnn_robust_ensemble",
    "physics_baseline",
    "physics_residual",
    "physics_residual_ensemble",
    "multiview_fusion",
    "multiview_residual",
    "bpm_hybrid_fusion",
    "bpm_dual_fusion",
)

STRICT_DISTANCE_CLASS_NAMES = (
    "d_2_0m",
    "d_2_25m",
    "d_3_0m",
    "d_3_175m",
    "d_4_0m",
    "d_5_0m",
    "d_6_0m",
    "d_7_0m",
    "d_10_0m",
    "d_15_0m",
    "d_20_0m",
)
STRICT_DISTANCE_EDGES = np.asarray(
    [2.125, 2.625, 3.0875, 3.5875, 4.5, 5.5, 6.5, 8.5, 12.5, 17.5],
    dtype=float,
)


def run_learning_suite(
    run_name: str,
    tasks: Iterable[str] = ("bpm", "distance"),
    methods: Iterable[str] = DEFAULT_METHODS,
    max_files: int | None = None,
    test_run: bool = False,
    epochs: int | None = None,
    save_preprocessed: bool = False,
    use_preprocessed: bool = False,
    rebuild_dataset: bool = False,
    source_run_name: str | None = None,
    physics_run_name: str | None = "full_experiment_final",
    distance_class_mode: str = "coarse",
    augment_bpm_windows: bool = False,
    bpm_signal_mode: str = "legacy",
) -> Path:
    config = build_default_config()
    if epochs is not None:
        config.train.epochs = int(epochs)
    run_root = config.data.test_output_root if max_files is not None or test_run else config.data.output_root
    run_dir = run_root / run_name / "learning"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        {
            "run_name": run_name,
            "tasks": list(tasks),
            "methods": list(methods),
            "max_files": max_files,
            "test_run": test_run,
            "epochs": int(config.train.epochs),
            "random_seed": int(config.train.random_seed),
            "validation_ratio": float(config.train.validation_ratio),
            "save_preprocessed": save_preprocessed,
            "use_preprocessed": use_preprocessed,
            "rebuild_dataset": rebuild_dataset,
            "source_run_name": source_run_name,
            "physics_run_name": physics_run_name,
            "distance_class_mode": distance_class_mode,
            "augment_bpm_windows": augment_bpm_windows,
            "bpm_signal_mode": bpm_signal_mode,
        },
        run_dir / "experiment_config.json",
    )

    records = build_manifest(config)
    save_manifest_csv(records, run_dir / "dataset_manifest.csv", config.data.dataset_root)
    summary_rows: list[dict] = []

    for task in tasks:
        task = str(task)
        task_config = _config_for_task(config, task)
        task_dir = run_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = task_dir / "prepared_learning_dataset.npz"
        source_dataset_path = _source_dataset_path(config, source_run_name, task)
        if dataset_path.exists() and not rebuild_dataset:
            _log(f"[learning {task}] using prepared dataset cache: {dataset_path}")
            dataset = load_learning_dataset(task, dataset_path)
        elif source_dataset_path is not None and source_dataset_path.exists() and not rebuild_dataset:
            _log(f"[learning {task}] using source prepared dataset cache: {source_dataset_path}")
            dataset = load_learning_dataset(task, source_dataset_path)
        else:
            _log(f"[learning {task}] preparing learning dataset...")
            dataset = prepare_learning_dataset(
                task,
                records,
                task_config,
                build_projection_matrix(task_config.signal),
                cache_dir=task_dir / "preprocessed",
                max_files=max_files,
                save_preprocessed=save_preprocessed,
                use_preprocessed=use_preprocessed,
                augment_bpm_windows=augment_bpm_windows,
                bpm_signal_mode=bpm_signal_mode,
                progress=_log,
            )
            save_learning_dataset(dataset, dataset_path)
        dataset = _apply_label_mode(dataset, distance_class_mode)
        _save_split_rows(dataset, task_dir / "file_split.csv")

        for method in methods:
            method = str(method)
            _log(f"[learning {task}] evaluating {method}...")
            method_dir = task_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            if method in ("pca_svm", "pca_knn"):
                row = _run_pca_method(dataset, method, method_dir)
            elif method == "physics_baseline":
                row = _run_physics_baseline(dataset, method_dir, task_config, physics_run_name)
            elif method in (
                "cnn",
                "cnn_lstm",
                "cnn_enhanced",
                "cnn_lstm_enhanced",
                "dnn_enhanced",
                "dnn_ensemble",
                "dnn_robust_ensemble",
                "physics_residual",
                "physics_residual_ensemble",
                "multiview_fusion",
                "multiview_residual",
                "bpm_hybrid_fusion",
                "bpm_dual_fusion",
            ):
                row = _run_deep_method(dataset, method, method_dir, task_config, physics_run_name)
            else:
                raise ValueError(f"Unknown learning method: {method}")
            row["task"] = task
            row["method"] = method
            summary_rows.append(row)
            _write_json(row, method_dir / "metrics.json")

    save_rows_csv(summary_rows, run_dir / "metrics_summary.csv")
    _log(f"Saved learning metrics summary to: {run_dir / 'metrics_summary.csv'}")
    return run_dir


def _run_pca_method(dataset: LearningDataset, method: str, method_dir: Path) -> dict:
    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError(f"{method} needs train and evaluation samples.")
    kind = "svm" if method == "pca_svm" else "knn"
    regressor = PCABaseline(kind=kind).fit(dataset.ml_features[train_idx], dataset.y_regression[train_idx])
    classifier = PCAClassifier(kind=kind).fit(dataset.ml_features[train_idx], dataset.y_class[train_idx])
    pred_reg = regressor.predict(dataset.ml_features[eval_idx])
    pred_class = classifier.predict(dataset.ml_features[eval_idx]).astype(int)
    return _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)


def _run_physics_baseline(
    dataset: LearningDataset,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    eval_name, eval_idx = _evaluation_indices(dataset)
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    pred_reg = physics_pred[eval_idx]
    pred_class = _classes_from_regression(dataset, pred_reg)
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row["physics_run_name"] = str(_resolved_physics_run_name(config, physics_run_name))
    row["physics_feature_dim"] = int(physics_features.shape[1])
    return row


def _run_deep_method(
    dataset: LearningDataset,
    method: str,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None = "full_experiment_final",
) -> dict:
    if method in ("dnn_ensemble", "dnn_robust_ensemble"):
        return _run_dnn_ensemble(dataset, method, method_dir, config)
    if method in ("physics_residual", "physics_residual_ensemble"):
        return _run_physics_residual(dataset, method, method_dir, config, physics_run_name)
    if method in ("multiview_fusion", "multiview_residual"):
        return _run_multiview_method(dataset, method, method_dir, config, physics_run_name)
    if method == "bpm_hybrid_fusion":
        return _run_hybrid_fusion(dataset, method_dir, config, physics_run_name)
    if method == "bpm_dual_fusion":
        return _run_dual_fusion(dataset, method_dir, config, physics_run_name)

    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    if method == "dnn_enhanced":
        x = dataset.ml_features
    else:
        x = dataset.sequence_inputs if "lstm" in method else dataset.cnn_inputs
    x = _standardize_deep_inputs(x, train_idx)
    if method == "cnn":
        input_channels = int(x.shape[1])
        model = build_cnn_multitask(input_channels, len(dataset.class_names))
        classification_weight = 0.25
    elif method == "cnn_lstm":
        input_channels = int(x.shape[1])
        model = build_cnn_lstm_multitask(input_channels, len(dataset.class_names))
        classification_weight = 0.25
    elif method == "cnn_enhanced":
        input_channels = int(x.shape[1])
        model = build_cnn_enhanced_multitask(input_channels, len(dataset.class_names))
        classification_weight = 0.35
    elif method == "cnn_lstm_enhanced":
        input_channels = int(x.shape[1])
        model = build_cnn_lstm_enhanced_multitask(input_channels, len(dataset.class_names))
        classification_weight = 0.35
    elif method == "dnn_enhanced":
        model = build_dnn_enhanced_multitask(int(x.shape[1]), len(dataset.class_names))
        classification_weight = 0.40
    else:
        raise ValueError(f"Unknown deep method: {method}")

    y_mean = float(np.mean(dataset.y_regression[train_idx]))
    y_std = float(np.std(dataset.y_regression[train_idx]))
    y_std = y_std if y_std > 1e-12 else 1.0
    y_scaled = (dataset.y_regression - y_mean) / y_std
    history = train_multitask_model(
        model,
        x,
        y_scaled,
        dataset.y_class,
        train_idx,
        dataset.split["val"],
        config.train,
        classification_weight=classification_weight,
        class_loss_weights=_class_loss_weights(dataset.y_class, train_idx, len(dataset.class_names)),
    )
    pred_scaled, pred_class = predict_multitask_model(model, x[eval_idx])
    pred_reg = pred_scaled * y_std + y_mean
    plot_loss_curve(
        np.asarray(history.train_loss),
        np.asarray(history.val_loss),
        method_dir / "train_loss.png",
    )
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row["epochs"] = int(config.train.epochs)
    row["train_final_loss"] = float(history.train_loss[-1])
    row["val_final_loss"] = float(history.val_loss[-1])
    row["best_epoch"] = int(history.best_epoch)
    row["best_val_loss"] = float(history.best_val_loss)
    row["regression_train_mean"] = y_mean
    row["regression_train_std"] = y_std
    row["classification_weight"] = classification_weight
    return row


def _run_dnn_ensemble(
    dataset: LearningDataset,
    method: str,
    method_dir: Path,
    config: ProjectConfig,
) -> dict:
    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    x = _standardize_deep_inputs(dataset.ml_features, train_idx)
    n_classes = len(dataset.class_names)
    y_mean = float(np.mean(dataset.y_regression[train_idx]))
    y_std = float(np.std(dataset.y_regression[train_idx]))
    y_std = y_std if y_std > 1e-12 else 1.0
    y_scaled = (dataset.y_regression - y_mean) / y_std
    class_weights = _class_loss_weights(dataset.y_class, train_idx, n_classes)
    seeds = (42, 101, 202, 303, 404)
    classification_weight = 0.22 if method == "dnn_robust_ensemble" else 0.35
    pred_scaled_runs = []
    pred_class_runs = []
    histories = []

    for seed in seeds:
        ensemble_train_config = replace(config.train, random_seed=seed)
        model = build_dnn_enhanced_multitask(int(x.shape[1]), n_classes)
        history = train_multitask_model(
            model,
            x,
            y_scaled,
            dataset.y_class,
            train_idx,
            dataset.split["val"],
            ensemble_train_config,
            classification_weight=classification_weight,
            class_loss_weights=class_weights,
        )
        pred_scaled, pred_class = predict_multitask_model(model, x[eval_idx])
        pred_scaled_runs.append(np.asarray(pred_scaled, dtype=float))
        pred_class_runs.append(np.asarray(pred_class, dtype=int))
        histories.append(history)

    stacked_pred = np.stack(pred_scaled_runs, axis=0)
    if method == "dnn_robust_ensemble":
        pred_scaled_final = np.median(stacked_pred, axis=0)
    else:
        pred_scaled_final = np.mean(stacked_pred, axis=0)
    pred_reg = pred_scaled_final * y_std + y_mean
    votes = np.stack(pred_class_runs, axis=0)
    pred_class = np.asarray(
        [np.bincount(votes[:, col], minlength=n_classes).argmax() for col in range(votes.shape[1])],
        dtype=int,
    )
    train_loss = np.mean(np.asarray([history.train_loss for history in histories], dtype=float), axis=0)
    val_loss = np.mean(np.asarray([history.val_loss for history in histories], dtype=float), axis=0)
    plot_loss_curve(train_loss, val_loss, method_dir / "train_loss.png")
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row["epochs"] = int(config.train.epochs)
    row["ensemble_size"] = int(len(seeds))
    row["ensemble_seeds"] = ";".join(str(seed) for seed in seeds)
    row["classification_weight"] = classification_weight
    row["regression_aggregation"] = "median" if method == "dnn_robust_ensemble" else "mean"
    row["regression_train_mean"] = y_mean
    row["regression_train_std"] = y_std
    row["best_epoch"] = ";".join(str(history.best_epoch) for history in histories)
    row["best_val_loss"] = float(np.mean([history.best_val_loss for history in histories]))
    row["train_final_loss"] = float(train_loss[-1])
    row["val_final_loss"] = float(val_loss[-1])
    return row


def _run_physics_residual(
    dataset: LearningDataset,
    method: str,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    if method == "physics_residual_ensemble":
        return _run_physics_residual_ensemble(dataset, method, method_dir, config, physics_run_name)

    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    x = _standardize_deep_inputs(np.concatenate([dataset.ml_features, physics_features], axis=1), train_idx)
    n_classes = len(dataset.class_names)
    residual = dataset.y_regression - physics_pred
    residual_mean, residual_std, residual_scaled = _scale_training_target(residual, train_idx)
    model = build_dnn_enhanced_multitask(int(x.shape[1]), n_classes)
    classification_weight = 0.18
    history = train_multitask_model(
        model,
        x,
        residual_scaled,
        dataset.y_class,
        train_idx,
        dataset.split["val"],
        config.train,
        classification_weight=classification_weight,
        class_loss_weights=_class_loss_weights(dataset.y_class, train_idx, n_classes),
    )
    pred_residual_scaled, _ = predict_multitask_model(model, x[eval_idx])
    pred_residual = pred_residual_scaled * residual_std + residual_mean
    pred_reg = physics_pred[eval_idx] + pred_residual
    pred_class = _classes_from_regression(dataset, pred_reg)
    plot_loss_curve(
        np.asarray(history.train_loss),
        np.asarray(history.val_loss),
        method_dir / "train_loss.png",
    )
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row.update(
        {
            "epochs": int(config.train.epochs),
            "physics_run_name": str(_resolved_physics_run_name(config, physics_run_name)),
            "physics_feature_dim": int(physics_features.shape[1]),
            "residual_train_mean": residual_mean,
            "residual_train_std": residual_std,
            "classification_weight": classification_weight,
            "train_final_loss": float(history.train_loss[-1]),
            "val_final_loss": float(history.val_loss[-1]),
            "best_epoch": int(history.best_epoch),
            "best_val_loss": float(history.best_val_loss),
        }
    )
    return row


def _run_physics_residual_ensemble(
    dataset: LearningDataset,
    method: str,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    del method
    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    x = _standardize_deep_inputs(np.concatenate([dataset.ml_features, physics_features], axis=1), train_idx)
    n_classes = len(dataset.class_names)
    residual = dataset.y_regression - physics_pred
    residual_mean, residual_std, residual_scaled = _scale_training_target(residual, train_idx)
    class_weights = _class_loss_weights(dataset.y_class, train_idx, n_classes)
    seeds = (42, 101, 202, 303, 404)
    classification_weight = 0.16
    pred_residual_runs = []
    histories = []

    for seed in seeds:
        ensemble_train_config = replace(config.train, random_seed=seed)
        model = build_dnn_enhanced_multitask(int(x.shape[1]), n_classes)
        history = train_multitask_model(
            model,
            x,
            residual_scaled,
            dataset.y_class,
            train_idx,
            dataset.split["val"],
            ensemble_train_config,
            classification_weight=classification_weight,
            class_loss_weights=class_weights,
        )
        pred_residual_scaled, _ = predict_multitask_model(model, x[eval_idx])
        pred_residual_runs.append(np.asarray(pred_residual_scaled, dtype=float))
        histories.append(history)

    pred_residual_scaled = np.median(np.stack(pred_residual_runs, axis=0), axis=0)
    pred_reg = physics_pred[eval_idx] + pred_residual_scaled * residual_std + residual_mean
    pred_class = _classes_from_regression(dataset, pred_reg)
    train_loss = np.mean(np.asarray([history.train_loss for history in histories], dtype=float), axis=0)
    val_loss = np.mean(np.asarray([history.val_loss for history in histories], dtype=float), axis=0)
    plot_loss_curve(train_loss, val_loss, method_dir / "train_loss.png")
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row.update(
        {
            "epochs": int(config.train.epochs),
            "ensemble_size": int(len(seeds)),
            "ensemble_seeds": ";".join(str(seed) for seed in seeds),
            "physics_run_name": str(_resolved_physics_run_name(config, physics_run_name)),
            "physics_feature_dim": int(physics_features.shape[1]),
            "residual_train_mean": residual_mean,
            "residual_train_std": residual_std,
            "regression_aggregation": "median",
            "classification_weight": classification_weight,
            "best_epoch": ";".join(str(history.best_epoch) for history in histories),
            "best_val_loss": float(np.mean([history.best_val_loss for history in histories])),
            "train_final_loss": float(train_loss[-1]),
            "val_final_loss": float(val_loss[-1]),
        }
    )
    return row


def _run_multiview_method(
    dataset: LearningDataset,
    method: str,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    train_idx = dataset.split["train"]
    eval_name, eval_idx = _evaluation_indices(dataset)
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    x, view_dims = _multiview_inputs(dataset, physics_features, train_idx)
    n_classes = len(dataset.class_names)
    residual_mode = method == "multiview_residual"
    if residual_mode:
        train_target = dataset.y_regression - physics_pred
        target_mean, target_std, y_scaled = _scale_training_target(train_target, train_idx)
        classification_weight = 0.18
    else:
        target_mean, target_std, y_scaled = _scale_training_target(dataset.y_regression, train_idx)
        classification_weight = 0.32
    model = build_multiview_fusion_multitask(view_dims, n_classes)
    history = train_multitask_model(
        model,
        x,
        y_scaled,
        dataset.y_class,
        train_idx,
        dataset.split["val"],
        config.train,
        classification_weight=classification_weight,
        class_loss_weights=_class_loss_weights(dataset.y_class, train_idx, n_classes),
    )
    pred_scaled, pred_class = predict_multitask_model(model, x[eval_idx])
    pred_reg = pred_scaled * target_std + target_mean
    if residual_mode:
        pred_reg = physics_pred[eval_idx] + pred_reg
        pred_class = _classes_from_regression(dataset, pred_reg)
    plot_loss_curve(
        np.asarray(history.train_loss),
        np.asarray(history.val_loss),
        method_dir / "train_loss.png",
    )
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row.update(
        {
            "epochs": int(config.train.epochs),
            "physics_run_name": str(_resolved_physics_run_name(config, physics_run_name)),
            "physics_feature_dim": int(physics_features.shape[1]),
            "view_dims": ";".join(str(dim) for dim in view_dims),
            "target_mode": "residual" if residual_mode else "direct",
            "target_train_mean": target_mean,
            "target_train_std": target_std,
            "classification_weight": classification_weight,
            "train_final_loss": float(history.train_loss[-1]),
            "val_final_loss": float(history.val_loss[-1]),
            "best_epoch": int(history.best_epoch),
            "best_val_loss": float(history.best_val_loss),
        }
    )
    return row


def _run_hybrid_fusion(
    dataset: LearningDataset,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    """Blend robust DNN predictions with physics-residual predictions using validation data only."""
    if dataset.task != "bpm":
        raise ValueError("bpm_hybrid_fusion is only intended for the BPM task.")
    val_idx = np.asarray(dataset.split["val"], dtype=int)
    eval_name, eval_idx = _evaluation_indices(dataset)
    if len(val_idx) == 0:
        raise ValueError("Hybrid fusion needs a validation split to tune the blending weight.")
    predict_idx = np.unique(np.concatenate([val_idx, eval_idx])).astype(int)
    position = {int(index): pos for pos, index in enumerate(predict_idx)}
    val_positions = np.asarray([position[int(index)] for index in val_idx], dtype=int)
    eval_positions = np.asarray([position[int(index)] for index in eval_idx], dtype=int)

    dnn_pred_all, _, dnn_meta = _predict_dnn_robust_for_indices(dataset, config, predict_idx)
    residual_pred_all, _, residual_meta = _predict_physics_residual_for_indices(
        dataset,
        config,
        physics_run_name,
        predict_idx,
    )

    alpha_grid = np.linspace(0.0, 1.0, 41)
    best_alpha = 0.5
    best_score = float("inf")
    best_validation = {}
    target_std = float(np.std(dataset.y_regression[dataset.split["train"]]))
    target_std = target_std if target_std > 1e-12 else 1.0
    for alpha in alpha_grid:
        blended_val = alpha * dnn_pred_all[val_positions] + (1.0 - alpha) * residual_pred_all[val_positions]
        validation_metrics = regression_metrics(dataset.y_regression[val_idx], blended_val)
        validation_metrics.update(
            classification_metrics(dataset.y_class[val_idx], _classes_from_regression(dataset, blended_val))
        )
        score = (
            validation_metrics["mae"] / target_std
            + 0.35 * (1.0 - validation_metrics["f1_macro"])
            + 0.20 * (1.0 - validation_metrics["accuracy"])
        )
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_validation = validation_metrics

    pred_reg = best_alpha * dnn_pred_all[eval_positions] + (1.0 - best_alpha) * residual_pred_all[eval_positions]
    pred_class = _classes_from_regression(dataset, pred_reg)
    row = _save_method_outputs(dataset, eval_name, eval_idx, pred_reg, pred_class, method_dir)
    row.update(
        {
            "epochs": int(config.train.epochs),
            "hybrid_alpha_dnn": best_alpha,
            "hybrid_alpha_physics_residual": float(1.0 - best_alpha),
            "hybrid_validation_score": best_score,
            "hybrid_validation_mae": float(best_validation.get("mae", float("nan"))),
            "hybrid_validation_accuracy": float(best_validation.get("accuracy", float("nan"))),
            "hybrid_validation_f1_macro": float(best_validation.get("f1_macro", float("nan"))),
            "dnn_best_epoch": dnn_meta["best_epoch"],
            "physics_residual_best_epoch": residual_meta["best_epoch"],
            "physics_run_name": str(_resolved_physics_run_name(config, physics_run_name)),
        }
    )
    return row


def _run_dual_fusion(
    dataset: LearningDataset,
    method_dir: Path,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> dict:
    if dataset.task != "bpm":
        raise ValueError("bpm_dual_fusion is only intended for the BPM task.")
    eval_name, eval_idx = _evaluation_indices(dataset)
    dnn_pred, _, dnn_meta = _predict_dnn_robust_for_indices(dataset, config, eval_idx)
    _, multiview_class, multiview_meta = _predict_multiview_for_indices(
        dataset,
        config,
        physics_run_name,
        eval_idx,
        residual_mode=False,
    )
    row = _save_method_outputs(dataset, eval_name, eval_idx, dnn_pred, multiview_class, method_dir)
    row.update(
        {
            "epochs": int(config.train.epochs),
            "regression_source": "dnn_robust_ensemble",
            "classification_source": "multiview_fusion",
            "dnn_best_epoch": dnn_meta["best_epoch"],
            "multiview_best_epoch": multiview_meta["best_epoch"],
            "physics_run_name": str(_resolved_physics_run_name(config, physics_run_name)),
        }
    )
    return row


def _predict_dnn_robust_for_indices(
    dataset: LearningDataset,
    config: ProjectConfig,
    predict_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_idx = dataset.split["train"]
    x = _standardize_deep_inputs(dataset.ml_features, train_idx)
    n_classes = len(dataset.class_names)
    y_mean = float(np.mean(dataset.y_regression[train_idx]))
    y_std = float(np.std(dataset.y_regression[train_idx]))
    y_std = y_std if y_std > 1e-12 else 1.0
    y_scaled = (dataset.y_regression - y_mean) / y_std
    class_weights = _class_loss_weights(dataset.y_class, train_idx, n_classes)
    seeds = (42, 101, 202, 303, 404)
    pred_scaled_runs = []
    pred_class_runs = []
    histories = []
    for seed in seeds:
        ensemble_train_config = replace(config.train, random_seed=seed)
        model = build_dnn_enhanced_multitask(int(x.shape[1]), n_classes)
        history = train_multitask_model(
            model,
            x,
            y_scaled,
            dataset.y_class,
            train_idx,
            dataset.split["val"],
            ensemble_train_config,
            classification_weight=0.22,
            class_loss_weights=class_weights,
        )
        pred_scaled, pred_class = predict_multitask_model(model, x[predict_idx])
        pred_scaled_runs.append(np.asarray(pred_scaled, dtype=float))
        pred_class_runs.append(np.asarray(pred_class, dtype=int))
        histories.append(history)

    pred_scaled_final = np.median(np.stack(pred_scaled_runs, axis=0), axis=0)
    pred_reg = pred_scaled_final * y_std + y_mean
    votes = np.stack(pred_class_runs, axis=0)
    pred_class = np.asarray(
        [np.bincount(votes[:, col], minlength=n_classes).argmax() for col in range(votes.shape[1])],
        dtype=int,
    )
    return pred_reg, pred_class, {
        "best_epoch": ";".join(str(history.best_epoch) for history in histories),
        "best_val_loss": float(np.mean([history.best_val_loss for history in histories])),
    }


def _predict_multiview_for_indices(
    dataset: LearningDataset,
    config: ProjectConfig,
    physics_run_name: str | None,
    predict_idx: np.ndarray,
    residual_mode: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_idx = dataset.split["train"]
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    x, view_dims = _multiview_inputs(dataset, physics_features, train_idx)
    n_classes = len(dataset.class_names)
    if residual_mode:
        train_target = dataset.y_regression - physics_pred
        target_mean, target_std, y_scaled = _scale_training_target(train_target, train_idx)
        classification_weight = 0.18
    else:
        target_mean, target_std, y_scaled = _scale_training_target(dataset.y_regression, train_idx)
        classification_weight = 0.32
    model = build_multiview_fusion_multitask(view_dims, n_classes)
    history = train_multitask_model(
        model,
        x,
        y_scaled,
        dataset.y_class,
        train_idx,
        dataset.split["val"],
        config.train,
        classification_weight=classification_weight,
        class_loss_weights=_class_loss_weights(dataset.y_class, train_idx, n_classes),
    )
    pred_scaled, pred_class = predict_multitask_model(model, x[predict_idx])
    pred_reg = pred_scaled * target_std + target_mean
    if residual_mode:
        pred_reg = physics_pred[predict_idx] + pred_reg
    return pred_reg, pred_class.astype(int), {
        "best_epoch": str(history.best_epoch),
        "best_val_loss": float(history.best_val_loss),
        "classification_weight": classification_weight,
    }


def _predict_physics_residual_for_indices(
    dataset: LearningDataset,
    config: ProjectConfig,
    physics_run_name: str | None,
    predict_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_idx = dataset.split["train"]
    physics_pred, physics_features = _physics_arrays(dataset, config, physics_run_name)
    x = _standardize_deep_inputs(np.concatenate([dataset.ml_features, physics_features], axis=1), train_idx)
    n_classes = len(dataset.class_names)
    residual = dataset.y_regression - physics_pred
    residual_mean, residual_std, residual_scaled = _scale_training_target(residual, train_idx)
    model = build_dnn_enhanced_multitask(int(x.shape[1]), n_classes)
    history = train_multitask_model(
        model,
        x,
        residual_scaled,
        dataset.y_class,
        train_idx,
        dataset.split["val"],
        config.train,
        classification_weight=0.18,
        class_loss_weights=_class_loss_weights(dataset.y_class, train_idx, n_classes),
    )
    pred_residual_scaled, _ = predict_multitask_model(model, x[predict_idx])
    pred_reg = physics_pred[predict_idx] + pred_residual_scaled * residual_std + residual_mean
    pred_class = _classes_from_regression(dataset, pred_reg)
    return pred_reg, pred_class, {
        "best_epoch": str(history.best_epoch),
        "best_val_loss": float(history.best_val_loss),
    }


def _save_method_outputs(
    dataset: LearningDataset,
    eval_name: str,
    eval_idx: np.ndarray,
    pred_reg: np.ndarray,
    pred_class: np.ndarray,
    method_dir: Path,
) -> dict:
    truth_reg = dataset.y_regression[eval_idx]
    truth_class = dataset.y_class[eval_idx]
    pred_reg = np.asarray(pred_reg, dtype=float).reshape(-1)
    pred_class = np.asarray(pred_class, dtype=int).reshape(-1)
    metrics = regression_metrics(truth_reg, pred_reg)
    metrics.update(classification_metrics(truth_class, pred_class))
    metrics["evaluation_split"] = eval_name
    metrics["n_train"] = int(len(dataset.split["train"]))
    metrics["n_val"] = int(len(dataset.split["val"]))
    metrics["n_test"] = int(len(dataset.split["test"]))
    metrics["n_eval"] = int(len(eval_idx))

    predictions = []
    for local_idx, dataset_idx in enumerate(eval_idx):
        class_true = int(truth_class[local_idx])
        class_pred = int(pred_class[local_idx])
        predictions.append(
            {
                "file": dataset.files[int(dataset_idx)],
                "subset": dataset.subsets[int(dataset_idx)],
                "scene": dataset.scenes[int(dataset_idx)],
                "evaluation_split": eval_name,
                "target": float(truth_reg[local_idx]),
                "prediction": float(pred_reg[local_idx]),
                "abs_error": float(abs(pred_reg[local_idx] - truth_reg[local_idx])),
                "class_true": class_true,
                "class_true_name": dataset.class_names[class_true],
                "class_prediction": class_pred,
                "class_prediction_name": dataset.class_names[class_pred],
            }
        )
    save_rows_csv(predictions, method_dir / "predictions.csv")

    from sklearn.metrics import confusion_matrix

    labels = np.arange(len(dataset.class_names))
    matrix = confusion_matrix(truth_class, pred_class, labels=labels)
    plot_confusion_matrix(matrix, dataset.class_names, method_dir / "confusion_matrix.png")
    plot_predictions(truth_reg, pred_reg, method_dir / "prediction_scatter.png")
    np.savetxt(method_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    return metrics


def _apply_label_mode(dataset: LearningDataset, distance_class_mode: str) -> LearningDataset:
    if dataset.task != "distance":
        return dataset
    if distance_class_mode not in ("coarse", "strict"):
        raise ValueError(f"Unknown distance class mode: {distance_class_mode}")
    if distance_class_mode == "coarse":
        return dataset
    dataset.y_class = np.digitize(dataset.y_regression, STRICT_DISTANCE_EDGES, right=True).astype(int)
    dataset.class_names = STRICT_DISTANCE_CLASS_NAMES
    dataset.split = file_level_split(dataset.y_class)
    return dataset


def _standardize_deep_inputs(x: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=int)
    if x.ndim == 2:
        axes = (0,)
    elif x.ndim == 3:
        axes = (0, 2)
    elif x.ndim == 4:
        axes = (0, 2, 3)
    else:
        axes = (0,)
    mean = np.mean(x[train_idx], axis=axes, keepdims=True)
    std = np.std(x[train_idx], axis=axes, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32)


def _scale_training_target(target: np.ndarray, train_idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=int)
    mean = float(np.mean(target[train_idx]))
    std = float(np.std(target[train_idx]))
    std = std if std > 1e-12 else 1.0
    return mean, std, ((target - mean) / std).astype(np.float32)


def _classes_from_regression(dataset: LearningDataset, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if dataset.task == "distance" and tuple(dataset.class_names) == STRICT_DISTANCE_CLASS_NAMES:
        return np.digitize(values, STRICT_DISTANCE_EDGES, right=True).astype(int)
    return np.asarray([class_label(dataset.task, float(value)) for value in values], dtype=int)


def _multiview_inputs(
    dataset: LearningDataset,
    physics_features: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    views = [
        np.asarray(dataset.ml_features, dtype=np.float32),
        np.asarray(dataset.cnn_inputs, dtype=np.float32).reshape(len(dataset.files), -1),
        np.asarray(dataset.sequence_inputs, dtype=np.float32).reshape(len(dataset.files), -1),
        np.asarray(physics_features, dtype=np.float32),
    ]
    standardized = [_standardize_deep_inputs(view, train_idx) for view in views]
    view_dims = tuple(int(view.shape[1]) for view in standardized)
    return np.concatenate(standardized, axis=1).astype(np.float32), view_dims


def _physics_arrays(
    dataset: LearningDataset,
    config: ProjectConfig,
    physics_run_name: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _load_physics_rows(config, dataset.task, physics_run_name)
    predictions = []
    features = []
    missing = []
    for file_path in dataset.files:
        key = _file_key(file_path)
        row = rows.get(key)
        if row is None:
            missing.append(Path(file_path).name)
            continue
        prediction = _physics_prediction_from_row(dataset.task, row)
        predictions.append(prediction)
        features.append(_physics_features_from_row(dataset.task, row, prediction))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Physics result CSV does not cover {len(missing)} {dataset.task} samples. "
            f"Missing examples: {preview}. Run the matching core task first or pass --physics-run-name."
        )
    return np.asarray(predictions, dtype=np.float32), np.asarray(features, dtype=np.float32)


def _load_physics_rows(
    config: ProjectConfig,
    task: str,
    physics_run_name: str | None,
) -> dict[str, dict[str, str]]:
    resolved_run = _resolved_physics_run_name(config, physics_run_name)
    result_path = _physics_result_path(config, resolved_run, task)
    if not result_path.exists():
        raise FileNotFoundError(
            f"Physics result CSV not found: {result_path}. "
            "Run core_full first, or pass --physics-run-name with an existing core run base name."
        )
    rows: dict[str, dict[str, str]] = {}
    with result_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows[_file_key(row.get("file", ""))] = row
    return rows


def _resolved_physics_run_name(config: ProjectConfig, physics_run_name: str | None) -> str:
    candidates = []
    if physics_run_name:
        candidates.append(str(physics_run_name))
    candidates.extend(["full_experiment_final", "core_full_final"])
    for candidate in candidates:
        if _physics_result_path(config, candidate, "bpm").exists() or _physics_result_path(config, candidate, "distance").exists():
            return candidate
    return candidates[0]


def _physics_result_path(config: ProjectConfig, run_name: str, task: str) -> Path:
    if task == "bpm":
        return config.data.output_root / f"{run_name}_core_bpm" / "bpm_results.csv"
    if task == "distance":
        return config.data.output_root / f"{run_name}_core_distance" / "distance_results.csv"
    raise ValueError(f"Unknown learning task: {task}")


def _physics_prediction_from_row(task: str, row: dict[str, str]) -> float:
    if task == "bpm":
        return _float_field(row, "prediction_bpm")
    if task == "distance":
        return _float_field(row, "prediction_distance_m")
    raise ValueError(f"Unknown learning task: {task}")


def _physics_features_from_row(task: str, row: dict[str, str], prediction: float) -> np.ndarray:
    if task == "bpm":
        estimated_distance = _float_field(row, "estimated_distance_m", default=0.0)
    else:
        estimated_distance = prediction
    feature_names = (
        "physics_prediction",
        "dynamic_delay_taps",
        "dynamic_tap_index",
        "smoothing_window_length",
        "estimated_distance_m",
    )
    _assert_no_label_leakage_feature_names(feature_names)
    return np.asarray(
        [
            prediction,
            _float_field(row, "dynamic_delay_taps", default=0.0),
            _float_field(row, "dynamic_tap_index", default=0.0),
            _float_field(row, "smoothing_window_length", default=0.0),
            estimated_distance,
        ],
        dtype=np.float32,
    )


def _assert_no_label_leakage_feature_names(feature_names: tuple[str, ...]) -> None:
    banned_tokens = (
        "ground_truth",
        "target",
        "label",
        "abs_error",
        "inferred_distance",
        "true",
    )
    leaked = [
        name
        for name in feature_names
        if any(token in name.lower() for token in banned_tokens)
    ]
    if leaked:
        raise ValueError(f"Potential label leakage in feature names: {', '.join(leaked)}")


def _float_field(row: dict[str, str], name: str, default: float | None = None) -> float:
    value = row.get(name, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            raise ValueError(f"Missing numeric field {name!r} in physics results.")
        return float(default)


def _file_key(file_path: str) -> str:
    value = str(file_path)
    try:
        name = PureWindowsPath(value).name
    except Exception:
        name = Path(value).name
    if not name:
        name = Path(value.replace("\\", "/")).name
    return name.lower()


def _class_loss_weights(y_class: np.ndarray, train_idx: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(y_class, dtype=int)[np.asarray(train_idx, dtype=int)]
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = np.sum(counts) / (n_classes * counts)
    weights = weights / np.mean(weights)
    return np.clip(weights, 0.5, 3.0).astype(np.float32)


def _evaluation_indices(dataset: LearningDataset) -> tuple[str, np.ndarray]:
    if len(dataset.split["test"]):
        return "test", dataset.split["test"]
    if len(dataset.split["val"]):
        return "validation", dataset.split["val"]
    return "train", dataset.split["train"]


def _save_split_rows(dataset: LearningDataset, output_path: Path) -> None:
    split_for_index = {}
    for split_name, indices in dataset.split.items():
        for index in indices:
            split_for_index[int(index)] = split_name
    rows = []
    for index, file_path in enumerate(dataset.files):
        label = int(dataset.y_class[index])
        rows.append(
            {
                "file": file_path,
                "subset": dataset.subsets[index],
                "scene": dataset.scenes[index],
                "split": split_for_index.get(index, "unassigned"),
                "target": float(dataset.y_regression[index]),
                "class_label": label,
                "class_name": dataset.class_names[label],
            }
        )
    save_rows_csv(rows, output_path)


def _write_json(value: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def _source_dataset_path(config: ProjectConfig, source_run_name: str | None, task: str) -> Path | None:
    if not source_run_name:
        return None
    candidates = [
        config.data.output_root / source_run_name / "learning" / task / "prepared_learning_dataset.npz",
        config.data.test_output_root / source_run_name / "learning" / task / "prepared_learning_dataset.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _config_for_task(config: ProjectConfig, task: str) -> ProjectConfig:
    if task == "distance":
        config.processing.sampling_rate_hz = 500.0
    elif task == "bpm":
        config.processing.sampling_rate_hz = 200.0
    else:
        raise ValueError(f"Unknown learning task: {task}")
    return config


def _log(message: str) -> None:
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run classical and deep CIRSense learning comparisons.")
    parser.add_argument("--tasks", nargs="+", choices=["bpm", "distance"], default=["bpm", "distance"])
    parser.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=list(DEFAULT_METHODS))
    parser.add_argument("--run-name", default="learning_comparison")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-preprocessed", action="store_true")
    parser.add_argument("--use-preprocessed", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--source-run-name",
        default=None,
        help="Reuse prepared_learning_dataset.npz from an existing learning run without rebuilding features.",
    )
    parser.add_argument(
        "--physics-run-name",
        default="full_experiment_final",
        help="Base run name for core physics CSVs, e.g. full_experiment_final uses full_experiment_final_core_bpm.",
    )
    parser.add_argument(
        "--distance-class-mode",
        choices=["coarse", "strict"],
        default="coarse",
        help="Use coarse near/mid/far/long distance classes or stricter distance bins for Acc/F1.",
    )
    parser.add_argument(
        "--bpm-window-augment",
        action="store_true",
        help="Add train-only respiration windows when rebuilding the BPM learning dataset.",
    )
    parser.add_argument(
        "--bpm-signal-mode",
        choices=["legacy", "quality"],
        default="legacy",
        help=(
            "BPM signal extraction for learning datasets. 'legacy' uses the original "
            "CIRSense dynamic-path waveform; 'quality' searches multi-antenna/tap "
            "candidates by respiration-spectrum quality."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_learning_suite(
        args.run_name,
        tasks=args.tasks,
        methods=args.methods,
        max_files=args.max_files,
        test_run=args.test_run,
        epochs=args.epochs,
        save_preprocessed=args.save_preprocessed,
        use_preprocessed=args.use_preprocessed,
        rebuild_dataset=args.rebuild_dataset,
        source_run_name=args.source_run_name,
        physics_run_name=args.physics_run_name,
        distance_class_mode=args.distance_class_mode,
        augment_bpm_windows=args.bpm_window_augment,
        bpm_signal_mode=args.bpm_signal_mode,
    )


if __name__ == "__main__":
    main()
