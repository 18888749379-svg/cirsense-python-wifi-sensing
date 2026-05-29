import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from cirsense_core import estimate_respiration_bpm
from config import ProjectConfig, build_default_config
from data_loader import infer_distance_from_filename


@dataclass(frozen=True)
class ManifestRecord:
    file: Path
    subset: str
    task: str
    scene: str
    distance_m: Optional[float]
    target_distances_m: tuple[float, ...]
    bpm: Optional[float]
    n_frames: Optional[int]

    def to_row(self, dataset_root: Path) -> dict:
        return {
            "file": str(self.file),
            "relative_file": str(self.file.relative_to(dataset_root)),
            "subset": self.subset,
            "task": self.task,
            "scene": self.scene,
            "distance_m": self.distance_m,
            "target_distances_m": ";".join(f"{value:.6g}" for value in self.target_distances_m),
            "bpm": self.bpm,
            "n_frames": self.n_frames,
        }


def build_manifest(config: Optional[ProjectConfig] = None) -> list[ManifestRecord]:
    config = config or build_default_config()
    records: list[ManifestRecord] = []
    records.extend(_scan_flat_subset(config, "breathe", "bpm"))
    records.extend(_scan_flat_subset(config, "nlos_breathe", "bpm"))
    records.extend(_scan_flat_subset(config, "distance", "distance"))
    records.extend(_scan_flat_subset(config, "multitarget/distance", "multitarget_distance"))
    records.extend(_scan_multitarget_breathe(config))
    return records


def save_manifest_csv(records: Iterable[ManifestRecord], output_path: Path, dataset_root: Path) -> None:
    rows = [record.to_row(dataset_root) for record in records]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No manifest rows to save.")
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_manifest(records: Iterable[ManifestRecord]) -> dict[str, dict]:
    records = list(records)
    summary: dict[str, dict] = {}
    for subset in sorted({record.subset for record in records}):
        subset_records = [record for record in records if record.subset == subset]
        distances = [record.distance_m for record in subset_records if record.distance_m is not None]
        bpms = [record.bpm for record in subset_records if record.bpm is not None]
        summary[subset] = {
            "files": len(subset_records),
            "scenes": sorted({record.scene for record in subset_records}),
            "distance_values_m": sorted({float(value) for value in distances}),
            "bpm_range": [float(np.min(bpms)), float(np.max(bpms))] if bpms else [],
        }
    return summary


def _scan_flat_subset(config: ProjectConfig, relative_subset: str, task: str) -> list[ManifestRecord]:
    subset_dir = config.data.dataset_root / relative_subset
    if not subset_dir.exists():
        return []
    return [
        _record_from_mat(file_path, relative_subset.replace("\\", "/"), task, config)
        for file_path in sorted(subset_dir.glob("*.mat"))
    ]


def _scan_multitarget_breathe(config: ProjectConfig) -> list[ManifestRecord]:
    root = config.data.dataset_root / "multitarget"
    records: list[ManifestRecord] = []
    for file_path in sorted(root.glob("target*_breathe/*/*.mat")):
        relative_subset = str(file_path.parent.relative_to(config.data.dataset_root)).replace("\\", "/")
        records.append(_record_from_mat(file_path, relative_subset, "multitarget_bpm", config))
    return records


def _record_from_mat(
    file_path: Path,
    subset: str,
    task: str,
    config: ProjectConfig,
) -> ManifestRecord:
    metadata = _load_metadata(file_path)
    target_distances = _vector(metadata.get("ground_truth_distance"))
    distance = float(target_distances[0]) if len(target_distances) else infer_distance_from_filename(file_path.name)
    bpm = None
    if "gt" in metadata and "gt_t" in metadata:
        bpm = estimate_respiration_bpm(metadata["gt"], metadata["gt_t"], config.processing)
    time_slice = metadata.get("t_slice")
    n_frames = int(np.asarray(time_slice).size) if time_slice is not None else None
    return ManifestRecord(
        file=Path(file_path),
        subset=subset,
        task=task,
        scene=_scene_from_path(file_path, subset),
        distance_m=distance,
        target_distances_m=tuple(float(value) for value in target_distances),
        bpm=bpm,
        n_frames=n_frames,
    )


def _load_metadata(file_path: Path) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to build the dataset manifest.") from exc
    raw = loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["ground_truth_distance", "gt", "gt_t", "t_slice"],
    )
    return {key: np.asarray(value) for key, value in raw.items() if not key.startswith("__")}


def _scene_from_path(file_path: Path, subset: str) -> str:
    if subset == "nlos_breathe":
        return "nlos"
    if subset == "multitarget/distance":
        return "multitarget"
    if subset.startswith("multitarget/target"):
        parts = Path(subset).parts
        return "_".join(parts[1:]) if len(parts) > 1 else "multitarget"
    prefix = file_path.stem.split("_", maxsplit=1)[0]
    return prefix or "unknown"


def _vector(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def parse_args():
    parser = argparse.ArgumentParser(description="Scan CIRSense metadata into a CSV manifest.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config()
    records = build_manifest(config)
    output = args.output or config.data.test_output_root / "dataset_manifest.csv"
    save_manifest_csv(records, output, config.data.dataset_root)
    print(f"Saved {len(records)} manifest rows to: {output}", flush=True)
    print("Summary:", summarize_manifest(records), flush=True)


if __name__ == "__main__":
    main()
