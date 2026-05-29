import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from config import build_default_config


CORE_EXPECTED_IMAGES = (
    "cir_magnitude.png",
    "dynamic_path_variance.png",
    "respiration_signal.png",
    "respiration_spectrum.png",
    "time_frequency_map.png",
)


def audit_run(run_name: str, root: Optional[Path] = None) -> dict:
    config = build_default_config()
    candidates = []
    if root is not None:
        candidates.append(Path(root) / run_name)
    else:
        candidates.extend([config.data.test_output_root / run_name, config.data.output_root / run_name])

    run_dir = next((path for path in candidates if path.exists()), None)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found for {run_name}. Checked: {candidates}")

    report = {"run_name": run_name, "run_dir": str(run_dir), "core": [], "learning": []}
    suite_summary = run_dir / "suite_summary.json"
    if suite_summary.exists():
        payload = json.loads(suite_summary.read_text(encoding="utf-8"))
        run_root = Path(payload.get("run_root", run_dir.parent))
        for child in payload.get("core_runs", []):
            child_dir = run_root / child["run_name"]
            report["core"].append(_audit_core_run(child_dir, child.get("subset")))
        learning_run = payload.get("learning_run")
        if learning_run:
            report["learning"].append(_audit_learning_run(Path(learning_run)))
    else:
        core = _audit_core_run(run_dir, None)
        if core["csv_found"]:
            report["core"].append(core)
        learning_dir = run_dir / "learning"
        if learning_dir.exists():
            report["learning"].append(_audit_learning_run(learning_dir))

    return report


def save_report(report: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_report(report: dict) -> None:
    print(f"Audit run: {report['run_name']}", flush=True)
    print(f"Run dir: {report['run_dir']}", flush=True)
    for item in report["core"]:
        print(
            f"[core] {item['run_dir']} rows={item['rows']} "
            f"mean_abs_error={item.get('mean_abs_error')} figures={item['figure_dirs']} "
            f"missing_images={len(item['missing_images'])}",
            flush=True,
        )
        if item.get("multitarget_mean_error") is not None:
            print(f"       multitarget_mean_error={item['multitarget_mean_error']}", flush=True)
    for item in report["learning"]:
        print(f"[learning] {item['run_dir']} rows={item['rows']}", flush=True)
        for metric in item["metrics"]:
            print(
                f"       {metric['task']}/{metric['method']}: "
                f"mae={metric.get('mae')} acc={metric.get('accuracy')} f1={metric.get('f1_macro')}",
                flush=True,
            )


def _audit_core_run(run_dir: Path, subset_hint: Optional[str]) -> dict:
    result_csv = _first_existing(run_dir, ["bpm_results.csv", "distance_results.csv"])
    if result_csv is None:
        return {
            "run_dir": str(run_dir),
            "subset": subset_hint,
            "csv_found": False,
            "rows": 0,
            "figure_dirs": 0,
            "missing_images": [],
        }

    rows = _read_csv(result_csv)
    abs_errors = [_float(row.get("abs_error")) for row in rows]
    abs_errors = [value for value in abs_errors if value is not None]
    multitarget_errors = [_float(row.get("multitarget_mae_m")) for row in rows]
    multitarget_errors = [value for value in multitarget_errors if value is not None]
    figure_root = run_dir / "figures"
    missing_images = _missing_core_images(figure_root, subset_hint)
    return {
        "run_dir": str(run_dir),
        "subset": subset_hint,
        "csv_found": True,
        "csv": str(result_csv),
        "rows": len(rows),
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
        "max_abs_error": float(np.max(abs_errors)) if abs_errors else None,
        "multitarget_mean_error": float(np.mean(multitarget_errors)) if multitarget_errors else None,
        "figure_dirs": _count_dirs(figure_root),
        "missing_images": missing_images,
    }


def _audit_learning_run(learning_dir: Path) -> dict:
    metrics_csv = learning_dir / "metrics_summary.csv"
    if not metrics_csv.exists():
        return {"run_dir": str(learning_dir), "csv_found": False, "rows": 0, "metrics": []}
    rows = _read_csv(metrics_csv)
    metrics = []
    for row in rows:
        metrics.append(
            {
                "task": row.get("task"),
                "method": row.get("method"),
                "mae": _float(row.get("mae")),
                "rmse": _float(row.get("rmse")),
                "accuracy": _float(row.get("accuracy")),
                "f1_macro": _float(row.get("f1_macro")),
                "evaluation_split": row.get("evaluation_split"),
                "n_eval": _int(row.get("n_eval")),
            }
        )
    missing_artifacts = []
    for row in rows:
        task = row.get("task")
        method = row.get("method")
        if not task or not method:
            continue
        method_dir = learning_dir / task / method
        for name in ("predictions.csv", "confusion_matrix.png", "prediction_scatter.png"):
            if not (method_dir / name).exists():
                missing_artifacts.append(str(method_dir / name))
    return {
        "run_dir": str(learning_dir),
        "csv_found": True,
        "csv": str(metrics_csv),
        "rows": len(rows),
        "metrics": metrics,
        "missing_artifacts": missing_artifacts,
    }


def _missing_core_images(figure_root: Path, subset_hint: Optional[str]) -> list[str]:
    if not figure_root.exists():
        return []
    missing = []
    for sample_dir in figure_root.iterdir():
        if not sample_dir.is_dir():
            continue
        expected = list(CORE_EXPECTED_IMAGES)
        if subset_hint == "multitarget_distance":
            expected.append("distance_profile.png")
        for image_name in expected:
            if not (sample_dir / image_name).exists():
                missing.append(str(sample_dir / image_name))
    return missing


def _first_existing(directory: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_dir())


def _float(value) -> Optional[float]:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value) -> Optional[int]:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def parse_args():
    parser = argparse.ArgumentParser(description="Audit core and learning experiment outputs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_run(args.run_name, args.root)
    output = args.output or Path(report["run_dir"]) / "audit_report.json"
    save_report(report, output)
    print_report(report)
    print(f"Saved audit report to: {output}", flush=True)


if __name__ == "__main__":
    main()
