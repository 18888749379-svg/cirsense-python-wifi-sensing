from dataclasses import dataclass

import numpy as np

from config import ProcessingConfig, SignalConfig
from csi_to_cir import CIRProjection, estimate_cir_from_csi


@dataclass
class DominoOutput:
    clean_csi: np.ndarray
    clean_cir: np.ndarray
    dominant_delay_taps: np.ndarray


def moving_average(data: np.ndarray, window_length: int, axis: int = 0) -> np.ndarray:
    if window_length <= 1:
        return data
    window_length = int(window_length)
    if window_length <= 1:
        return data
    return np.apply_along_axis(_moving_average_1d, axis, data, window_length)


def _moving_average_1d(values: np.ndarray, window_length: int) -> np.ndarray:
    """Centered moving average with truncated endpoints, matching MATLAB movmean."""
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return values
    half_left = window_length // 2
    half_right = window_length - half_left
    cumsum = np.concatenate([[0], np.cumsum(values)])
    out = np.empty(n, dtype=np.result_type(values, np.complex128 if np.iscomplexobj(values) else float))
    for idx in range(n):
        start = max(0, idx - half_left)
        end = min(n, idx + half_right)
        out[idx] = (cumsum[end] - cumsum[start]) / (end - start)
    return out


def compute_window_length(
    sampling_rate_hz: float,
    smoothing_window_s: float,
    n_time: int,
) -> int:
    length = int(min(sampling_rate_hz * smoothing_window_s, max(n_time / 20, 1)))
    length = max(length, 3)
    if length % 2 == 0:
        length += 1
    return min(length, n_time if n_time % 2 == 1 else max(n_time - 1, 3))


def normalize_complex_signal(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.sqrt(np.mean(np.abs(data) ** 2, axis=0, keepdims=True))
    return data / np.maximum(scale, eps)


def domino_compensate(
    csi: np.ndarray,
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
    progress_callback=None,
    progress_every: int = 500,
) -> DominoOutput:
    """Domino-style dominant path alignment for hardware distortion compensation.

    This follows the public MATLAB flow: find the dominant tap per frame, search
    a fractional delay that maximizes the aligned central tap, phase-shift CSI
    with raw `used_sc`, then normalize by the strongest aligned CIR tap.
    """
    csi = np.asarray(csi, dtype=np.complex128)
    if csi.ndim == 2:
        csi = csi[:, :, None]

    initial_cir = estimate_cir_from_csi(csi, projection)
    clean_csi = np.empty_like(csi)
    clean_cir = np.empty_like(initial_cir)
    dominant_delay_taps = np.zeros((csi.shape[0], csi.shape[2]), dtype=float)
    center = signal_config.center_tap_index
    center_row = projection.matrix[center]

    n_time = csi.shape[0]
    for time_idx in range(n_time):
        for ant_idx in range(csi.shape[2]):
            frame = csi[time_idx, :, ant_idx]
            approx_tap_index = int(np.argmax(np.abs(initial_cir[time_idx, :, ant_idx])))
            approx_delay = projection.tap_values[approx_tap_index]
            candidates = np.linspace(
                approx_delay - 0.5,
                approx_delay + 0.5,
                processing_config.n_delay_candidates,
            )
            shifted = _shift_frame_candidates(
                frame,
                candidates,
                projection.used_subcarriers,
                signal_config.fft_size,
            )
            center_values = center_row @ shifted
            best_idx = int(np.argmax(np.abs(center_values)))
            best_delay = float(candidates[best_idx])
            dominant_delay_taps[time_idx, ant_idx] = best_delay

            phase_shift = np.exp(
                2j * np.pi * projection.used_subcarriers * best_delay / signal_config.fft_size
            )
            corrected_frame = frame * phase_shift
            corrected_cir = projection.matrix @ corrected_frame

            normalizer = corrected_cir[np.argmax(np.abs(corrected_cir))]
            clean_csi[time_idx, :, ant_idx] = corrected_frame / normalizer
            clean_cir[time_idx, :, ant_idx] = corrected_cir / normalizer
        if progress_callback is not None and (
            (time_idx + 1) % progress_every == 0 or (time_idx + 1) == n_time
        ):
            progress_callback(time_idx + 1, n_time)

    return DominoOutput(
        clean_csi=clean_csi,
        clean_cir=clean_cir,
        dominant_delay_taps=dominant_delay_taps,
    )


def _shift_frame_candidates(
    frame: np.ndarray,
    candidates: np.ndarray,
    used_subcarriers: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    phase = np.exp(2j * np.pi * np.outer(used_subcarriers, candidates) / fft_size)
    return frame[:, None] * phase
