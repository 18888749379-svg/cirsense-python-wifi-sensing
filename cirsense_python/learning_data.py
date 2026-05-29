from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from cirsense_core import (
    prepare_respiration_waveform,
    preprocess_csi_for_cirsense,
    run_cirsense_from_preprocessed,
)
from config import ProjectConfig
from csi_to_cir import CIRProjection
from data_loader import load_csi_sample
from dataset_manifest import ManifestRecord
from feature_extraction import cir_time_image, statistical_features, stft_features


DISTANCE_CLASS_NAMES = ("near_0_3m", "mid_3_6m", "far_6_10m", "long_gt10m")
BPM_CLASS_NAMES = ("slow_lt15", "normal_15_18", "normal_18_21", "fast_21_25", "fast_gt25")


@dataclass
class LearningDataset:
    task: str
    files: list[str]
    scenes: list[str]
    subsets: list[str]
    y_regression: np.ndarray
    y_class: np.ndarray
    class_names: tuple[str, ...]
    ml_features: np.ndarray
    cnn_inputs: np.ndarray
    sequence_inputs: np.ndarray
    split: dict[str, np.ndarray]


def prepare_learning_dataset(
    task: str,
    records: list[ManifestRecord],
    config: ProjectConfig,
    projection: CIRProjection,
    cache_dir: Optional[Path] = None,
    max_files: Optional[int] = None,
    save_preprocessed: bool = False,
    use_preprocessed: bool = False,
    augment_bpm_windows: bool = False,
    bpm_signal_mode: str = "legacy",
    progress: Optional[Callable[[str], None]] = None,
) -> LearningDataset:
    if bpm_signal_mode not in ("legacy", "quality"):
        raise ValueError("bpm_signal_mode must be 'legacy' or 'quality'.")
    candidates = _records_for_task(task, records)
    candidates = _evenly_limit_records(task, candidates, max_files)
    if not candidates:
        raise ValueError(f"No manifest records contain labels for learning task {task}.")

    files: list[str] = []
    scenes: list[str] = []
    subsets: list[str] = []
    y_regression = []
    y_class = []
    ml_features = []
    cnn_inputs = []
    sequence_inputs = []
    forced_split: dict[str, list[int]] | None = None
    record_split_names: dict[int, str] = {}
    if task == "bpm" and augment_bpm_windows:
        base_labels = np.asarray([class_label(task, float(record.bpm)) for record in candidates], dtype=int)
        base_split = file_level_split(base_labels)
        forced_split = {"train": [], "val": [], "test": []}
        for split_name, indices in base_split.items():
            for record_index in indices:
                record_split_names[int(record_index)] = split_name

    for index, record in enumerate(candidates, start=1):
        _log(progress, f"[learning {task} {index}/{len(candidates)}] {record.file.name} loading...")
        sample = load_csi_sample(record.file, config.signal)
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{_safe_cache_name(record)}_preprocessed.npz"
        intermediates = _get_preprocessed(
            sample.csi,
            record,
            config,
            projection,
            cache_path=cache_path,
            save_preprocessed=save_preprocessed,
            use_preprocessed=use_preprocessed,
            progress=progress,
        )

        if task == "bpm":
            result, _ = run_cirsense_from_preprocessed(
                intermediates["smooth_csi"],
                intermediates["smooth_cir"],
                sample.time_s,
                projection,
                config.signal,
                config.processing,
            )
            if bpm_signal_mode == "quality":
                respiration_signal = _best_respiration_signal(
                    result.respiration_signal,
                    intermediates["smooth_cir"],
                    config.processing.sampling_rate_hz,
                    config.processing,
                    config.signal.center_tap_index,
                )
            else:
                # Legacy mode keeps the original CIRSense-derived dynamic path
                # waveform. It produced the strongest BPM learning results in
                # the current experiments, so it is the default for reproducible
                # report runs.
                respiration_signal = np.asarray(result.respiration_signal, dtype=float).reshape(-1)
            target = float(record.bpm)
            split_name = record_split_names.get(index - 1, "train") if forced_split is not None else None
            windows = [respiration_signal]
            if augment_bpm_windows and split_name == "train":
                windows.extend(_respiration_training_windows(respiration_signal, config.processing.sampling_rate_hz))
            for window in windows:
                _append_learning_sample(
                    files,
                    scenes,
                    subsets,
                    y_regression,
                    y_class,
                    ml_features,
                    cnn_inputs,
                    sequence_inputs,
                    record,
                    target,
                    _respiration_ml_features(window, config.processing.sampling_rate_hz),
                    _respiration_cnn_input(window, config.processing.sampling_rate_hz),
                    _respiration_sequence_input(window),
                    task,
                    forced_split,
                    split_name,
                )
            continue
        elif task == "distance":
            selected_cir = _select_first_antenna(intermediates["smooth_cir"])
            ml_vector = statistical_features(selected_cir)
            cnn_input = _distance_cnn_input(selected_cir)
            sequence_input = _distance_sequence_input(selected_cir)
            target = float(record.distance_m)
        else:
            raise ValueError(f"Unknown learning task: {task}")

        _append_learning_sample(
            files,
            scenes,
            subsets,
            y_regression,
            y_class,
            ml_features,
            cnn_inputs,
            sequence_inputs,
            record,
            target,
            ml_vector,
            cnn_input,
            sequence_input,
            task,
            forced_split,
            record_split_names.get(index - 1),
        )

    y_class_array = np.asarray(y_class, dtype=int)
    if forced_split is None:
        split = file_level_split(y_class_array)
    else:
        split = {name: np.asarray(indices, dtype=int) for name, indices in forced_split.items()}
    return LearningDataset(
        task=task,
        files=files,
        scenes=scenes,
        subsets=subsets,
        y_regression=np.asarray(y_regression, dtype=np.float32),
        y_class=y_class_array,
        class_names=class_names(task),
        ml_features=np.stack(ml_features),
        cnn_inputs=np.stack(cnn_inputs),
        sequence_inputs=np.stack(sequence_inputs),
        split=split,
    )


def save_learning_dataset(dataset: LearningDataset, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        files=np.asarray(dataset.files),
        scenes=np.asarray(dataset.scenes),
        subsets=np.asarray(dataset.subsets),
        y_regression=dataset.y_regression,
        y_class=dataset.y_class,
        class_names=np.asarray(dataset.class_names),
        ml_features=dataset.ml_features,
        cnn_inputs=dataset.cnn_inputs,
        sequence_inputs=dataset.sequence_inputs,
        split_train=dataset.split["train"],
        split_val=dataset.split["val"],
        split_test=dataset.split["test"],
    )


def load_learning_dataset(task: str, input_path: Path) -> LearningDataset:
    with np.load(input_path, allow_pickle=False) as payload:
        return LearningDataset(
            task=task,
            files=[str(value) for value in payload["files"]],
            scenes=[str(value) for value in payload["scenes"]],
            subsets=[str(value) for value in payload["subsets"]],
            y_regression=np.asarray(payload["y_regression"], dtype=np.float32),
            y_class=np.asarray(payload["y_class"], dtype=int),
            class_names=tuple(str(value) for value in payload["class_names"]),
            ml_features=np.asarray(payload["ml_features"], dtype=np.float32),
            cnn_inputs=np.asarray(payload["cnn_inputs"], dtype=np.float32),
            sequence_inputs=np.asarray(payload["sequence_inputs"], dtype=np.float32),
            split={
                "train": np.asarray(payload["split_train"], dtype=int),
                "val": np.asarray(payload["split_val"], dtype=int),
                "test": np.asarray(payload["split_test"], dtype=int),
            },
        )


def class_names(task: str) -> tuple[str, ...]:
    if task == "distance":
        return DISTANCE_CLASS_NAMES
    if task == "bpm":
        return BPM_CLASS_NAMES
    raise ValueError(f"Unknown learning task: {task}")


def class_label(task: str, target: float) -> int:
    if task == "distance":
        return int(np.digitize([target], [3.25, 6.25, 10.25], right=True)[0])
    if task == "bpm":
        return int(np.digitize([target], [15.0, 18.0, 21.0, 25.0], right=False)[0])
    raise ValueError(f"Unknown learning task: {task}")


def _append_learning_sample(
    files: list[str],
    scenes: list[str],
    subsets: list[str],
    y_regression: list[float],
    y_class: list[int],
    ml_features: list[np.ndarray],
    cnn_inputs: list[np.ndarray],
    sequence_inputs: list[np.ndarray],
    record: ManifestRecord,
    target: float,
    ml_vector: np.ndarray,
    cnn_input: np.ndarray,
    sequence_input: np.ndarray,
    task: str,
    forced_split: Optional[dict[str, list[int]]],
    split_name: Optional[str],
) -> None:
    sample_index = len(files)
    files.append(str(record.file))
    scenes.append(record.scene)
    subsets.append(record.subset)
    y_regression.append(float(target))
    y_class.append(class_label(task, float(target)))
    ml_features.append(np.asarray(ml_vector, dtype=np.float32))
    cnn_inputs.append(np.asarray(cnn_input, dtype=np.float32))
    sequence_inputs.append(np.asarray(sequence_input, dtype=np.float32))
    if forced_split is not None and split_name is not None:
        forced_split[split_name].append(sample_index)


def file_level_split(labels: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        if len(indices) >= 5:
            n_test = max(1, int(round(len(indices) * 0.2)))
            n_val = max(1, int(round(len(indices) * 0.15)))
        elif len(indices) >= 3:
            n_test = 1
            n_val = 1
        elif len(indices) == 2:
            n_test = 1
            n_val = 0
        else:
            n_test = 0
            n_val = 0
        test.extend(indices[:n_test].tolist())
        val.extend(indices[n_test : n_test + n_val].tolist())
        train.extend(indices[n_test + n_val :].tolist())

    train = sorted(train)
    val = sorted(val)
    test = sorted(test)
    if not test and len(train) > 1:
        test.append(train.pop())
    if not val and len(train) > 2:
        val.append(train.pop())
    return {
        "train": np.asarray(train, dtype=int),
        "val": np.asarray(sorted(val), dtype=int),
        "test": np.asarray(sorted(test), dtype=int),
    }


def _records_for_task(task: str, records: list[ManifestRecord]) -> list[ManifestRecord]:
    if task == "bpm":
        return [record for record in records if record.subset == "breathe" and record.bpm is not None]
    if task == "distance":
        return [
            record
            for record in records
            if record.subset == "distance" and record.distance_m is not None
        ]
    raise ValueError(f"Unknown learning task: {task}")


def _evenly_limit_records(
    task: str,
    records: list[ManifestRecord],
    max_files: Optional[int],
) -> list[ManifestRecord]:
    if max_files is None or len(records) <= max_files:
        return records
    groups: dict[int, list[ManifestRecord]] = {}
    for record in records:
        target = record.bpm if task == "bpm" else record.distance_m
        groups.setdefault(class_label(task, float(target)), []).append(record)
    ordered_groups = [sorted(group, key=lambda item: str(item.file)) for group in groups.values()]
    selected: list[ManifestRecord] = []
    cursor = 0
    while len(selected) < max_files and ordered_groups:
        group = ordered_groups[cursor % len(ordered_groups)]
        if group:
            selected.append(group.pop(0))
        ordered_groups = [items for items in ordered_groups if items]
        cursor += 1
    return selected


def _get_preprocessed(
    csi: np.ndarray,
    record: ManifestRecord,
    config: ProjectConfig,
    projection: CIRProjection,
    cache_path: Optional[Path],
    save_preprocessed: bool,
    use_preprocessed: bool,
    progress: Optional[Callable[[str], None]],
) -> dict[str, np.ndarray]:
    if cache_path is not None and use_preprocessed and cache_path.exists():
        _log(progress, f"[learning] {record.file.name} using preprocessing cache.")
        return _load_preprocessed(cache_path)

    _log(progress, f"[learning] {record.file.name} preprocessing raw CSI.")
    intermediates = preprocess_csi_for_cirsense(
        csi,
        projection,
        config.signal,
        config.processing,
    )
    if cache_path is not None and save_preprocessed:
        _save_preprocessed(intermediates, cache_path)
    return intermediates


def _save_preprocessed(intermediates: dict[str, np.ndarray], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        smooth_csi=intermediates["smooth_csi"],
        smooth_cir=intermediates["smooth_cir"],
        dominant_delay_taps=intermediates["dominant_delay_taps"],
        window_length=intermediates["window_length"],
    )


def _load_preprocessed(input_path: Path) -> dict[str, np.ndarray]:
    with np.load(input_path, allow_pickle=False) as payload:
        return {
            "smooth_csi": np.asarray(payload["smooth_csi"]),
            "smooth_cir": np.asarray(payload["smooth_cir"]),
            "dominant_delay_taps": np.asarray(payload["dominant_delay_taps"]),
            "window_length": np.asarray(payload["window_length"]),
        }


def _distance_cnn_input(cir: np.ndarray) -> np.ndarray:
    image = cir_time_image(cir, use_phase=True)
    image = _resize_tensor_time(image, 160)
    return _standardize_channels(image)


def _distance_sequence_input(cir: np.ndarray) -> np.ndarray:
    amplitude = np.abs(cir)
    profile = np.mean(amplitude, axis=0)
    variance = np.var(cir, axis=0).real
    return _standardize_channels(np.stack([profile, variance], axis=0))


def _respiration_cnn_input(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    frequencies, times, tf_map = stft_features(signal, sampling_rate_hz)
    del times
    mask = frequencies * 60.0 <= 60.0
    tf_map = tf_map[mask, :] if np.any(mask) else tf_map
    image = _resize_2d(np.asarray(tf_map, dtype=float), 64, 48)
    return _standardize_channels(image[None, :, :])


def _respiration_sequence_input(signal: np.ndarray) -> np.ndarray:
    signal = np.real(np.asarray(signal).reshape(-1))
    return _standardize_channels(_resample_vector(signal, 256)[None, :])


def _best_respiration_signal(
    base_signal: np.ndarray,
    smooth_cir: np.ndarray,
    sampling_rate_hz: float,
    processing_config,
    center_tap_index: int,
) -> np.ndarray:
    candidates = [np.asarray(base_signal).reshape(-1)]
    cir = np.asarray(smooth_cir)
    if cir.ndim == 2:
        cir = cir[:, :, None]
    for antenna in range(cir.shape[2]):
        variance = np.var(cir[:, :, antenna], axis=0).real
        if len(variance) == 0:
            continue
        center = int(center_tap_index)
        start = max(0, center - 2)
        stop = min(len(variance), center + 3)
        variance[start:stop] = 0.0
        top_count = min(3, len(variance))
        for tap_index in np.argsort(variance)[-top_count:]:
            if variance[int(tap_index)] <= 0:
                continue
            candidates.append(cir[:, int(tap_index), antenna])

    best_signal = np.asarray(base_signal).reshape(-1)
    best_score = _respiration_quality_score(best_signal, sampling_rate_hz, processing_config)
    for candidate in candidates:
        try:
            prepared = prepare_respiration_waveform(candidate, sampling_rate_hz, processing_config)
        except Exception:
            continue
        score = _respiration_quality_score(prepared, sampling_rate_hz, processing_config)
        if score > best_score:
            best_score = score
            best_signal = prepared
    return np.asarray(best_signal, dtype=float).reshape(-1)


def _respiration_quality_score(
    signal: np.ndarray,
    sampling_rate_hz: float,
    processing_config,
) -> float:
    x = np.asarray(signal).reshape(-1)
    if np.iscomplexobj(x):
        x = np.real(x)
    x = x - np.mean(x)
    if len(x) < 8 or np.std(x) <= 1e-12:
        return 0.0
    n_fft = max(processing_config.n_fft, int(2 ** np.ceil(np.log2(len(x)))))
    spectrum = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    bpm = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz) * 60.0
    band_mask = (
        (bpm >= processing_config.respiration_bpm_min)
        & (bpm <= processing_config.respiration_bpm_max)
    )
    if not np.any(band_mask):
        return 0.0
    band_power = spectrum[band_mask]
    peak = float(np.max(band_power))
    floor = float(np.median(band_power)) + 1e-12
    total = float(np.sum(band_power)) + 1e-12
    return (peak / floor) * (peak / total)


def _respiration_training_windows(
    signal: np.ndarray,
    sampling_rate_hz: float,
    max_windows: int = 3,
) -> list[np.ndarray]:
    x = np.asarray(signal, dtype=float).reshape(-1)
    if len(x) < 16:
        return []
    window_length = max(int(round(6.0 * sampling_rate_hz)), int(round(len(x) * 0.65)))
    window_length = min(window_length, len(x))
    if window_length >= int(0.95 * len(x)):
        return []
    hop = max(1, window_length // 2)
    windows = []
    for start in range(0, len(x) - window_length + 1, hop):
        windows.append(x[start : start + window_length])
        if len(windows) >= max_windows:
            break
    return windows


def _respiration_ml_features(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    signal = np.real(np.asarray(signal).reshape(-1))
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std > 1e-12:
        signal = signal / std
    n_fft = max(1024, int(2 ** np.ceil(np.log2(max(len(signal), 1)))))
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    bpm = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz) * 60.0
    spectrum_0_60 = np.interp(np.linspace(0.0, 60.0, 96), bpm, spectrum)
    spectrum_0_60 = spectrum_0_60 / max(float(np.max(spectrum_0_60)), 1e-12)
    waveform = _resample_vector(signal, 64)
    stats = np.asarray(
        [np.mean(signal), np.std(signal), np.max(signal), np.min(signal), np.median(signal)],
        dtype=float,
    )
    return np.concatenate([spectrum_0_60, waveform, stats])


def _select_first_antenna(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 3:
        return data[:, :, 0]
    return data


def _resize_tensor_time(tensor: np.ndarray, target_time: int) -> np.ndarray:
    tensor = np.asarray(tensor)
    return np.stack([_resize_2d(channel, target_time, channel.shape[1]) for channel in tensor], axis=0)


def _resize_2d(array: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return np.zeros((target_rows, target_cols), dtype=float)
    row_resized = np.stack([_resample_vector(row, target_cols) for row in array], axis=0)
    columns = [_resample_vector(row_resized[:, column], target_rows) for column in range(target_cols)]
    return np.stack(columns, axis=1)


def _resample_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return np.zeros(target_size, dtype=float)
    if len(values) == 1:
        return np.full(target_size, values[0], dtype=float)
    old_axis = np.linspace(0.0, 1.0, len(values))
    new_axis = np.linspace(0.0, 1.0, target_size)
    return np.interp(new_axis, old_axis, values)


def _standardize_channels(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    flat = array.reshape(array.shape[0], -1)
    mean = np.mean(flat, axis=1)
    std = np.std(flat, axis=1)
    std = np.where(std > 1e-12, std, 1.0)
    shape = (array.shape[0],) + (1,) * (array.ndim - 1)
    return (array - mean.reshape(shape)) / std.reshape(shape)


def _safe_cache_name(record: ManifestRecord) -> str:
    value = str(Path(record.subset) / record.file.stem)
    return value.replace("\\", "_").replace("/", "_")


def _log(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
