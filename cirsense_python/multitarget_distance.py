from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import numpy as np

from config import SignalConfig


@dataclass
class MultitargetDistanceResult:
    target_distances_m: np.ndarray
    target_tap_values: np.ndarray
    target_tap_indices: np.ndarray
    target_scores: np.ndarray
    mean_profile: np.ndarray
    variance_profile: np.ndarray
    detection_score: np.ndarray
    distance_axis_m: np.ndarray
    tap_values: np.ndarray


def estimate_multitarget_distances(
    cir: np.ndarray,
    signal_config: SignalConfig,
    target_count: int = 2,
    antenna_index: int = 0,
    profile_weight: float = 0.45,
    variance_weight: float = 0.55,
    prominence_weight: float = 0.0,
    exclude_los_taps: int = 1,
    min_peak_separation_taps: int = 2,
) -> MultitargetDistanceResult:
    """Estimate multiple target distances from an averaged CIR distance profile.

    The mean CIR magnitude is the main distance profile. A temporal-variance
    term is fused into the detection score so the strong static LOS component
    does not hide weaker moving targets.
    """
    cir = np.asarray(cir, dtype=np.complex128)
    if cir.ndim == 2:
        cir = cir[:, :, None]
    elif cir.ndim == 3:
        if antenna_index is not None and cir.shape[2] == 1:
            cir = cir[:, :, :1]
    else:
        raise ValueError(f"CIR must have shape (time, taps) or (time, taps, antennas), got {cir.shape}.")

    tap_values = np.asarray(signal_config.tap_values, dtype=float)
    distance_axis_m = tap_to_distance_m(tap_values, signal_config)

    mean_by_antenna = np.mean(np.abs(cir), axis=0)
    centered = cir - np.mean(cir, axis=0, keepdims=True)
    raw_variance_by_antenna = np.mean(np.abs(centered) ** 2, axis=0)

    profile_by_antenna = np.apply_along_axis(_normalize, 0, mean_by_antenna)
    variance_by_antenna = np.apply_along_axis(_normalize, 0, raw_variance_by_antenna)
    profile_norm = np.mean(profile_by_antenna, axis=1)
    variance_norm = np.mean(variance_by_antenna, axis=1)

    mean_profile = np.mean(mean_by_antenna, axis=1)
    variance_profile = np.mean(raw_variance_by_antenna, axis=1)
    prominence_score = _normalize(
        _local_prominence(profile_norm, radius=2)
        + _local_prominence(variance_norm, radius=2)
    )
    detection_score = profile_weight * profile_norm + variance_weight * variance_norm
    if prominence_weight > 0.0:
        detection_score = detection_score + prominence_weight * prominence_score

    valid = distance_axis_m > 0.0
    valid &= np.abs(tap_values) > exclude_los_taps
    detection_score = np.where(valid, detection_score, 0.0)

    target_count = max(1, int(target_count))
    peak_indices = _select_peak_indices(
        detection_score,
        target_count=target_count,
        min_peak_separation_taps=max(1, int(min_peak_separation_taps)),
    )

    target_taps = _refine_peak_taps(detection_score, tap_values, peak_indices)
    target_distances = tap_to_distance_m(target_taps, signal_config)
    target_scores = detection_score[peak_indices]
    order = np.argsort(target_distances)
    peak_indices = peak_indices[order]

    return MultitargetDistanceResult(
        target_distances_m=target_distances[order],
        target_tap_values=target_taps[order],
        target_tap_indices=peak_indices.astype(int),
        target_scores=target_scores[order],
        mean_profile=mean_profile,
        variance_profile=variance_profile,
        detection_score=detection_score,
        distance_axis_m=distance_axis_m,
        tap_values=tap_values,
    )


def tap_to_distance_m(tap_values: np.ndarray, signal_config: SignalConfig) -> np.ndarray:
    relative_path_length = np.asarray(tap_values, dtype=float) / signal_config.bandwidth_hz
    relative_path_length *= signal_config.speed_of_light
    return (relative_path_length + signal_config.los_distance_m) / 2.0


def distance_matching_errors(
    predicted_m: np.ndarray,
    truth_m: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if truth_m is None:
        return None
    predicted = np.asarray(predicted_m, dtype=float).reshape(-1)
    truth = np.asarray(truth_m, dtype=float).reshape(-1)
    if len(predicted) == 0 or len(truth) == 0:
        return None
    count = min(len(predicted), len(truth))
    predicted = predicted[:count]
    truth = truth[:count]
    if count > 6:
        return np.abs(np.sort(predicted) - np.sort(truth))

    best_errors = None
    best_total = float("inf")
    for order in permutations(range(count)):
        ordered_pred = predicted[list(order)]
        errors = np.abs(ordered_pred - truth)
        total = float(np.sum(errors))
        if total < best_total:
            best_total = total
            best_errors = errors
    return best_errors


def _select_peak_indices(
    score: np.ndarray,
    target_count: int,
    min_peak_separation_taps: int,
) -> np.ndarray:
    candidate_score = np.asarray(score, dtype=float)
    local_peaks = _local_maxima(candidate_score)
    if len(local_peaks):
        local_peaks = local_peaks[np.argsort(candidate_score[local_peaks])[::-1]]
    selected = _non_maximum_suppression(local_peaks, candidate_score, target_count, min_peak_separation_taps)
    if len(selected) < target_count:
        all_indices = np.argsort(candidate_score)[::-1]
        fallback = _non_maximum_suppression(
            all_indices,
            candidate_score,
            target_count,
            min_peak_separation_taps,
            selected,
        )
        selected = fallback
    return np.asarray(selected[:target_count], dtype=int)


def _local_maxima(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if len(score) < 3:
        return np.arange(len(score))
    indices = []
    for idx in range(1, len(score) - 1):
        if score[idx] > 0.0 and score[idx] >= score[idx - 1] and score[idx] >= score[idx + 1]:
            indices.append(idx)
    return np.asarray(indices, dtype=int)


def _non_maximum_suppression(
    candidates: np.ndarray,
    score: np.ndarray,
    target_count: int,
    min_peak_separation_taps: int,
    initial: Optional[list[int]] = None,
) -> list[int]:
    selected = list(initial or [])
    for idx in candidates:
        idx = int(idx)
        if score[idx] <= 0.0:
            continue
        if any(abs(idx - chosen) < min_peak_separation_taps for chosen in selected):
            continue
        selected.append(idx)
        if len(selected) >= target_count:
            break
    return selected


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.min(values)
    peak = np.max(values)
    if peak <= 1e-12:
        return np.zeros_like(values)
    return values / peak


def _local_prominence(values: np.ndarray, radius: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    radius = max(1, int(radius))
    prominence = np.zeros_like(values)
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        neighborhood = np.concatenate([values[start:idx], values[idx + 1 : end]])
        baseline = float(np.max(neighborhood)) if len(neighborhood) else 0.0
        prominence[idx] = max(0.0, float(values[idx]) - baseline)
    return prominence


def _smooth_1d(values: np.ndarray, window_length: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window_length <= 1 or len(values) < 3:
        return values
    window_length = min(int(window_length), len(values))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= 1:
        return values
    kernel = np.ones(window_length, dtype=float) / window_length
    pad = window_length // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _refine_peak_taps(score: np.ndarray, tap_values: np.ndarray, peak_indices: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    tap_values = np.asarray(tap_values, dtype=float)
    refined = []
    tap_step = float(np.median(np.diff(tap_values))) if len(tap_values) > 1 else 1.0
    for peak_index in np.asarray(peak_indices, dtype=int):
        if peak_index <= 0 or peak_index >= len(score) - 1:
            refined.append(float(tap_values[peak_index]))
            continue
        left = float(score[peak_index - 1])
        center = float(score[peak_index])
        right = float(score[peak_index + 1])
        if center <= max(left, right):
            refined.append(float(tap_values[peak_index]))
            continue
        local_prominence = center - max(left, right)
        if local_prominence < 0.02 * max(float(np.max(score)), 1e-12):
            refined.append(float(tap_values[peak_index]))
            continue
        denominator = left - 2.0 * center + right
        if abs(denominator) <= 1e-12:
            refined.append(float(tap_values[peak_index]))
            continue
        offset = 0.5 * (left - right) / denominator
        offset = float(np.clip(offset, -0.25, 0.25))
        refined.append(float(tap_values[peak_index] + offset * tap_step))
    return np.asarray(refined, dtype=float)
