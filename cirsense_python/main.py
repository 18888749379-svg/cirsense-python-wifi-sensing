import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from cirsense_core import (
    estimate_respiration_bpm,
    preprocess_csi_for_cirsense,
    run_cirsense_from_preprocessed,
)
from config import build_default_config
from csi_to_cir import build_projection_matrix
from data_loader import list_data_files, load_csi_sample
from evaluate import regression_metrics, save_rows_csv
from feature_extraction import build_feature_bundle
from multitarget_distance import distance_matching_errors, estimate_multitarget_distances


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CIRSense Python pipeline.")
    parser.add_argument("--task", choices=["bpm", "distance"], default="bpm")
    parser.add_argument(
        "--subset",
        choices=["breathe", "distance", "nlos_breathe", "multitarget_distance"],
        default=None,
        help="Dataset subset under CIRSense_dataset. Defaults to breathe for bpm and distance for distance.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fs", type=float, default=None, help="Sampling rate in Hz.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for quick tests.")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Save this run under output_test even when it processes all files.",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save CIR, spectrum and time-frequency images.")
    parser.add_argument("--plot-limit", type=int, default=5, help="Maximum number of files to plot.")
    parser.add_argument("--save-features", action="store_true", help="Save compact feature .npz files.")
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save smoothed CSI/CIR preprocessing caches under this run directory.",
    )
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Reuse preprocessing caches from this run directory when they exist.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Save preprocessing caches and skip sensing estimates, CSV rows and plots.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name. Quick --max-files runs default to output_test; full runs use outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing CSV in the run directory and skip completed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config()
    if args.fs is not None:
        config.processing.sampling_rate_hz = args.fs
    elif args.task == "distance":
        config.processing.sampling_rate_hz = 500.0

    subset = args.subset
    if subset is None:
        subset = config.data.respiration_subdir if args.task == "bpm" else config.data.distance_subdir

    if args.data_dir is None:
        data_dir = _subset_to_dir(config, subset)
    else:
        data_dir = args.data_dir
    is_multitarget_distance = args.task == "distance" and subset == "multitarget_distance"

    run_name = _build_run_name(args, subset)
    run_root = _select_run_root(config, args)
    run_dir = run_root / run_name
    output_path = _resolve_output_path(args.output, run_dir, args.task)
    feature_dir = run_dir / "features"
    figure_dir = run_dir / "figures"
    checkpoint_dir = run_dir / "checkpoints"
    preprocessed_dir = run_dir / "preprocessed"

    _log(f"Task: {args.task}")
    _log(f"Dataset subset: {subset}")
    _log(f"Dataset directory: {data_dir}")
    _log(f"Output root: {run_root}")
    _log(f"Run directory: {run_dir}")
    if args.preprocess_only:
        _log(f"Preprocessing cache directory: {preprocessed_dir}")
    else:
        _log(f"Results CSV: {output_path}")

    projection = build_projection_matrix(config.signal)
    files = list_data_files(data_dir)
    if args.max_files is not None:
        files = files[: args.max_files]
    _log(f"Files to process: {len(files)}")

    rows = _load_existing_rows(output_path) if args.resume else []
    processed_files = {row.get("file") for row in rows if row.get("file")}
    if args.resume and rows:
        _log(f"Resume enabled: loaded {len(rows)} completed row(s).")

    plotted_count = _count_existing_plot_dirs(figure_dir) if args.resume else 0

    for file_index, file_path in enumerate(files):
        file_path = Path(file_path)
        label = f"[{file_index + 1}/{len(files)}] {file_path.name}"
        file_key = str(file_path)
        if file_key in processed_files:
            _log(f"{label} skipped because it already exists in the results CSV.")
            continue

        _log(f"{label} loading CSI data...")
        sample = load_csi_sample(file_path, config.signal)
        _log(
            f"{label} loaded shape={sample.csi.shape}, "
            f"inferred_distance={sample.distance_m}."
        )

        def progress(done_frames: int, total_frames: int) -> None:
            _log(f"{label} preprocessing progress: {done_frames}/{total_frames} frames")

        cache_path = preprocessed_dir / f"{_safe_sample_id(file_path, config.data.dataset_root)}_preprocessed.npz"
        if args.use_preprocessed and cache_path.exists():
            _log(f"{label} loading preprocessing cache...")
            intermediates = _load_preprocessed_cache(cache_path)
        else:
            if args.use_preprocessed:
                _log(f"{label} preprocessing cache not found; creating it from raw CSI...")
            else:
                _log(f"{label} preprocessing CSI and estimating CIR...")
            intermediates = preprocess_csi_for_cirsense(
                sample.csi,
                projection,
                config.signal,
                config.processing,
                progress_callback=progress,
            )

        if args.save_preprocessed or args.preprocess_only:
            _log(f"{label} saving preprocessing cache...")
            _save_preprocessed_cache(intermediates, cache_path)
        if args.preprocess_only:
            _log(f"{label} preprocessing cache ready; skipped sensing estimation.")
            continue

        _log(f"{label} running CIRSense estimates...")
        result, bpm = run_cirsense_from_preprocessed(
            intermediates["smooth_csi"],
            intermediates["smooth_cir"],
            sample.time_s,
            projection,
            config.signal,
            config.processing,
        )
        _log(f"{label} extracting amplitude, phase, spectrum and time-frequency features...")
        features = build_feature_bundle(
            intermediates["smooth_csi"],
            intermediates["smooth_cir"],
            result.respiration_signal,
            config.processing.sampling_rate_hz,
        )
        multitarget_result = None
        if is_multitarget_distance:
            target_count = _target_count_for_sample(sample)
            _log(f"{label} estimating {target_count} target distance candidates from CIR profile...")
            multitarget_result = estimate_multitarget_distances(
                intermediates["smooth_cir"],
                config.signal,
                target_count=target_count,
            )

        row = {
            "file": str(file_path),
            "run_name": run_name,
            "subset": str(Path(file_path).parent.relative_to(config.data.dataset_root))
            if config.data.dataset_root in Path(file_path).parents
            else Path(file_path).parent.name,
            "inferred_distance_m": sample.distance_m,
            "dynamic_delay_taps": result.dynamic_delay_taps,
            "dynamic_tap_index": result.dynamic_tap_index,
            "smoothing_window_length": int(intermediates["window_length"]),
        }
        if args.task == "distance":
            prediction = result.estimated_distance_m
            target = sample.distance_m
            row["prediction_distance_m"] = prediction
            row["ground_truth_distance_m"] = target
            if multitarget_result is not None:
                _add_multitarget_fields(row, multitarget_result, sample.target_distances_m)
        else:
            prediction = bpm
            target = sample.bpm
            if target is None and sample.ground_truth_signal is not None:
                target = estimate_respiration_bpm(
                    sample.ground_truth_signal,
                    sample.ground_truth_time_s,
                    config.processing,
                )
            row["prediction_bpm"] = prediction
            row["ground_truth_bpm"] = target
            row["estimated_distance_m"] = result.estimated_distance_m

        if target is not None:
            row["abs_error"] = abs(float(prediction) - float(target))

        if args.save_features:
            _log(f"{label} saving feature file...")
            _save_feature_file(features, feature_dir / f"{_safe_sample_id(file_path, config.data.dataset_root)}_features.npz")
        if args.save_plots and plotted_count < args.plot_limit:
            _log(f"{label} saving plots...")
            _save_plot_set(
                file_path,
                intermediates,
                result,
                features,
                figure_dir,
                config.data.dataset_root,
                multitarget_result=multitarget_result,
                ground_truth_distances_m=sample.target_distances_m,
            )
            plotted_count += 1

        rows.append(row)
        processed_files.add(file_key)
        _save_checkpoint(row, checkpoint_dir / f"{_safe_sample_id(file_path, config.data.dataset_root)}.json")
        save_rows_csv(rows, output_path)
        _log(f"{label} done. Row saved immediately.")
        print(row, flush=True)

    if rows:
        save_rows_csv(rows, output_path)
        _log(f"Saved results to: {output_path}")

    y_true, y_pred = _collect_metric_pairs(rows, args.task)
    if y_true:
        metrics = regression_metrics(np.asarray(y_true), np.asarray(y_pred))
        print("Metrics:", metrics, flush=True)
    multitarget_metrics = _collect_multitarget_metrics(rows)
    if multitarget_metrics:
        print("Multitarget metrics:", multitarget_metrics, flush=True)


def _subset_to_dir(config, subset: str) -> Path:
    if subset == "breathe":
        return config.data.dataset_root / config.data.respiration_subdir
    if subset == "distance":
        return config.data.dataset_root / config.data.distance_subdir
    if subset == "nlos_breathe":
        return config.data.dataset_root / config.data.nlos_respiration_subdir
    if subset == "multitarget_distance":
        return config.data.dataset_root / config.data.multitarget_subdir / "distance"
    raise ValueError(f"Unknown subset: {subset}")


def _build_run_name(args, subset: str) -> str:
    if args.run_name:
        return _sanitize_name(args.run_name)
    prefix = "quick" if args.max_files is not None or args.test_run else "full"
    return _sanitize_name(f"{prefix}_{args.task}_{subset}")


def _select_run_root(config, args) -> Path:
    if args.max_files is not None or args.test_run:
        return config.data.test_output_root
    return config.data.output_root


def _resolve_output_path(output_arg, run_dir: Path, task: str) -> Path:
    if output_arg is None:
        return run_dir / f"{task}_results.csv"
    output_path = Path(output_arg)
    if output_path.is_absolute():
        return output_path
    return run_dir / output_path


def _load_existing_rows(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with output_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _collect_metric_pairs(rows: list[dict], task: str) -> tuple[list[float], list[float]]:
    y_true: list[float] = []
    y_pred: list[float] = []
    if task == "distance":
        true_key = "ground_truth_distance_m"
        pred_key = "prediction_distance_m"
    else:
        true_key = "ground_truth_bpm"
        pred_key = "prediction_bpm"
    for row in rows:
        truth = _try_float(row.get(true_key))
        prediction = _try_float(row.get(pred_key))
        if truth is not None and prediction is not None:
            y_true.append(truth)
            y_pred.append(prediction)
    return y_true, y_pred


def _try_float(value) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_sample_id(file_path: Path, dataset_root: Path) -> str:
    file_path = Path(file_path)
    try:
        raw = str(file_path.relative_to(dataset_root))
    except ValueError:
        raw = file_path.name
    raw = raw.replace(file_path.suffix, "")
    return _sanitize_name(raw.replace("\\", "_").replace("/", "_"))


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "run"


def _save_checkpoint(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(row), file, ensure_ascii=False, indent=2)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _target_count_for_sample(sample) -> int:
    if sample.target_distances_m is not None and len(sample.target_distances_m):
        return int(len(sample.target_distances_m))
    return 2


def _add_multitarget_fields(row: dict, multitarget_result, ground_truth_distances_m) -> None:
    predictions = np.asarray(multitarget_result.target_distances_m, dtype=float).reshape(-1)
    row["multitarget_count"] = int(len(predictions))
    row["prediction_target_distances_m"] = _format_float_list(predictions)
    row["prediction_target_taps"] = _format_float_list(multitarget_result.target_tap_values)
    row["prediction_target_scores"] = _format_float_list(multitarget_result.target_scores)
    if ground_truth_distances_m is not None:
        truth = np.asarray(ground_truth_distances_m, dtype=float).reshape(-1)
        row["ground_truth_target_distances_m"] = _format_float_list(truth)
        errors = distance_matching_errors(predictions, truth)
        if errors is not None:
            row["multitarget_mae_m"] = float(np.mean(errors))
            row["multitarget_max_error_m"] = float(np.max(errors))
            for idx, error in enumerate(errors, start=1):
                row[f"target{idx}_abs_error_m"] = float(error)
    for idx, distance in enumerate(predictions, start=1):
        row[f"target{idx}_distance_m"] = float(distance)


def _format_float_list(values) -> str:
    values = np.asarray(values, dtype=float).reshape(-1)
    return ";".join(f"{value:.6g}" for value in values)


def _collect_multitarget_metrics(rows: list[dict]) -> dict[str, float]:
    errors = []
    max_errors = []
    for row in rows:
        mae = _try_float(row.get("multitarget_mae_m"))
        max_error = _try_float(row.get("multitarget_max_error_m"))
        if mae is not None:
            errors.append(mae)
        if max_error is not None:
            max_errors.append(max_error)
    if not errors:
        return {}
    return {
        "mae": float(np.mean(errors)),
        "max_abs_error": float(np.max(max_errors)) if max_errors else float(np.max(errors)),
    }


def _count_existing_plot_dirs(figure_dir: Path) -> int:
    if not figure_dir.exists():
        return 0
    return sum(1 for item in figure_dir.iterdir() if item.is_dir())


def _log(message: str) -> None:
    print(message, flush=True)


def _save_feature_file(features: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **features)


def _save_preprocessed_cache(intermediates: dict[str, np.ndarray], output_path: Path) -> None:
    cache = {
        "smooth_csi": intermediates["smooth_csi"],
        "smooth_cir": intermediates["smooth_cir"],
        "dominant_delay_taps": intermediates["dominant_delay_taps"],
        "window_length": intermediates["window_length"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)


def _load_preprocessed_cache(cache_path: Path) -> dict[str, np.ndarray]:
    required = ("smooth_csi", "smooth_cir", "dominant_delay_taps", "window_length")
    with np.load(cache_path, allow_pickle=False) as cache:
        missing = [key for key in required if key not in cache]
        if missing:
            raise KeyError(f"Preprocessing cache {cache_path} is missing arrays: {missing}")
        return {key: np.asarray(cache[key]) for key in required}


def _save_plot_set(
    file_path: Path,
    intermediates: dict,
    result,
    features: dict[str, np.ndarray],
    figure_dir: Path,
    dataset_root: Path,
    multitarget_result=None,
    ground_truth_distances_m=None,
) -> None:
    from visualize import (
        plot_cir_magnitude,
        plot_delay_variance,
        plot_distance_profile,
        plot_respiration_signal,
        plot_spectrum,
        plot_time_frequency,
    )
    import matplotlib.pyplot as plt

    sample_dir = figure_dir / _safe_sample_id(file_path, dataset_root)
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_cir_magnitude(intermediates["smooth_cir"], sample_dir / "cir_magnitude.png")
    plot_respiration_signal(result.respiration_signal, None, sample_dir / "respiration_signal.png")
    plot_spectrum(
        features["respiration_frequency_hz"],
        features["respiration_spectrum"],
        sample_dir / "respiration_spectrum.png",
    )
    plot_time_frequency(
        features["stft_frequency_hz"],
        features["stft_time_s"],
        features["time_frequency_map"],
        sample_dir / "time_frequency_map.png",
    )
    plot_delay_variance(
        result.delay_candidates,
        result.variance_curve,
        sample_dir / "dynamic_path_variance.png",
    )
    if multitarget_result is not None:
        plot_distance_profile(
            multitarget_result.distance_axis_m,
            multitarget_result.mean_profile,
            multitarget_result.detection_score,
            multitarget_result.target_distances_m,
            sample_dir / "distance_profile.png",
            ground_truth_distances_m=ground_truth_distances_m,
        )
    plt.close("all")


if __name__ == "__main__":
    main()
