from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import ProcessingConfig, SignalConfig
from csi_to_cir import CIRProjection, estimate_cir_from_csi
from preprocessing import moving_average


@dataclass
class CIRSenseResult:
    estimated_distance_m: float
    respiration_signal: np.ndarray
    dynamic_delay_taps: float
    dynamic_tap_index: int
    variance_curve: np.ndarray
    delay_candidates: np.ndarray


def dynamic_path_alignment(
    clean_csi: np.ndarray,
    clean_cir: np.ndarray,
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
    antenna_index: int = 0,
) -> CIRSenseResult:
    """Run Dylign-style dynamic path alignment for one target."""
    csi = np.asarray(clean_csi, dtype=np.complex128)
    cir = np.asarray(clean_cir, dtype=np.complex128)
    if csi.ndim == 2:
        csi = csi[:, :, None]
    if cir.ndim == 2:
        cir = cir[:, :, None]

    selected_cir = cir[:, :, antenna_index]
    variance_per_tap = np.var(selected_cir, axis=0)
    variance_per_tap[signal_config.center_tap_index] = 0.0
    dynamic_tap_index = int(np.argmax(variance_per_tap))
    approx_delay = projection.tap_values[dynamic_tap_index]

    delay_candidates = np.linspace(
        approx_delay - 0.5,
        approx_delay + 0.5,
        processing_config.n_delay_candidates,
    )
    selected_csi = csi[:, :, antenna_index]
    variance_curve = _candidate_variances(
        selected_csi,
        delay_candidates,
        projection,
        signal_config,
        processing_config,
    )
    best_idx = int(np.argmax(variance_curve))
    best_delay = float(delay_candidates[best_idx])

    phase_shift = np.exp(
        2j * np.pi * projection.data_subcarrier_indices * best_delay / signal_config.fft_size
    )
    aligned_csi = selected_csi * phase_shift[None, :]
    aligned_cir = estimate_cir_from_csi(aligned_csi[:, :, None], projection)[:, :, 0]
    respiration_signal = np.mean(aligned_csi, axis=1)

    relative_path_length = best_delay / signal_config.bandwidth_hz * signal_config.speed_of_light
    estimated_distance = (relative_path_length + signal_config.los_distance_m) / 2.0

    return CIRSenseResult(
        estimated_distance_m=float(estimated_distance),
        respiration_signal=respiration_signal,
        dynamic_delay_taps=best_delay,
        dynamic_tap_index=dynamic_tap_index,
        variance_curve=variance_curve,
        delay_candidates=delay_candidates,
    )


def estimate_respiration_bpm(
    signal: np.ndarray,
    time_s: Optional[np.ndarray],
    processing_config: ProcessingConfig,
) -> float:
    """Estimate respiration rate by finding the dominant spectral peak."""
    if time_s is None:
        sampling_rate = processing_config.sampling_rate_hz
    else:
        time_s = np.asarray(time_s).reshape(-1)
        sampling_rate = 1.0 / np.median(np.diff(time_s))

    x = prepare_respiration_waveform(signal, sampling_rate, processing_config)

    if len(x) < 4:
        return 0.0

    n_fft = max(processing_config.n_fft, int(2 ** np.ceil(np.log2(len(x)))))
    spectrum = np.abs(np.fft.rfft(x, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)
    bpm = freqs * 60.0
    mask = (
        (bpm >= processing_config.respiration_bpm_min)
        & (bpm <= processing_config.respiration_bpm_max)
    )
    if not np.any(mask):
        return 0.0
    peak_local = int(np.argmax(spectrum[mask]))
    return float(bpm[mask][peak_local])


def run_cirsense_pipeline(
    csi: np.ndarray,
    time_s: Optional[np.ndarray],
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
    return_intermediates: bool = False,
    progress_callback=None,
):
    intermediates = preprocess_csi_for_cirsense(
        csi,
        projection,
        signal_config,
        processing_config,
        progress_callback=progress_callback,
    )
    result, bpm = run_cirsense_from_preprocessed(
        intermediates["smooth_csi"],
        intermediates["smooth_cir"],
        time_s,
        projection,
        signal_config,
        processing_config,
    )
    if return_intermediates:
        return result, bpm, intermediates
    return result, bpm


def preprocess_csi_for_cirsense(
    csi: np.ndarray,
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    from preprocessing import compute_window_length, domino_compensate

    domino = domino_compensate(
        csi,
        projection,
        signal_config,
        processing_config,
        progress_callback=progress_callback,
    )
    window = compute_window_length(
        processing_config.sampling_rate_hz,
        processing_config.smoothing_window_s,
        domino.clean_cir.shape[0],
    )
    smooth_cir = moving_average(domino.clean_cir, window, axis=0)
    smooth_csi = moving_average(domino.clean_csi, window, axis=0)
    return {
        "clean_csi": domino.clean_csi,
        "clean_cir": domino.clean_cir,
        "smooth_csi": smooth_csi,
        "smooth_cir": smooth_cir,
        "dominant_delay_taps": domino.dominant_delay_taps,
        "window_length": np.asarray(window),
    }


def run_cirsense_from_preprocessed(
    smooth_csi: np.ndarray,
    smooth_cir: np.ndarray,
    time_s: Optional[np.ndarray],
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
) -> tuple[CIRSenseResult, float]:
    """Run Dylign and sensing estimation from preprocessed CSI/CIR arrays."""
    result = dynamic_path_alignment(
        smooth_csi,
        smooth_cir,
        projection,
        signal_config,
        processing_config,
    )
    sampling_rate = _sampling_rate_from_time(time_s, processing_config)
    result.respiration_signal = prepare_respiration_waveform(
        result.respiration_signal,
        sampling_rate,
        processing_config,
    )
    bpm = estimate_respiration_bpm(result.respiration_signal, time_s, processing_config)
    return result, bpm


def prepare_respiration_waveform(
    signal: np.ndarray,
    sampling_rate_hz: float,
    processing_config: ProcessingConfig,
) -> np.ndarray:
    """Project, detrend, band-limit and normalize a respiration candidate signal."""
    x = np.asarray(signal).reshape(-1)
    if np.iscomplexobj(x):
        x = _best_angle_projection(x, processing_config, sampling_rate_hz)
    x = np.real(x).astype(float)
    x = _remove_linear_trend(x)
    x = _fft_bandpass(
        x,
        sampling_rate_hz,
        processing_config.respiration_bpm_min / 60.0,
        processing_config.respiration_bpm_max / 60.0,
    )
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-12:
        x = x / std
    return x


def _sampling_rate_from_time(
    time_s: Optional[np.ndarray],
    processing_config: ProcessingConfig,
) -> float:
    if time_s is None:
        return processing_config.sampling_rate_hz
    time_s = np.asarray(time_s).reshape(-1)
    if len(time_s) < 2:
        return processing_config.sampling_rate_hz
    diffs = np.diff(time_s)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return processing_config.sampling_rate_hz
    return float(1.0 / np.median(diffs))


def _candidate_variances(
    csi: np.ndarray,
    delay_candidates: np.ndarray,
    projection: CIRProjection,
    signal_config: SignalConfig,
    processing_config: ProcessingConfig,
) -> np.ndarray:
    center_row = projection.matrix[signal_config.center_tap_index]
    stride = max(1, int(processing_config.candidate_stride))
    csi_subsampled = csi[::stride, :]
    variances = np.zeros(len(delay_candidates), dtype=float)
    for idx, candidate in enumerate(delay_candidates):
        phase_shift = np.exp(
            2j * np.pi * projection.data_subcarrier_indices * candidate / signal_config.fft_size
        )
        h_center = (csi_subsampled * phase_shift[None, :]) @ center_row.T
        variances[idx] = float(np.var(h_center))
    return variances


def _best_angle_projection(
    signal: np.ndarray,
    processing_config: ProcessingConfig,
    sampling_rate_hz: float,
) -> np.ndarray:
    angles = np.linspace(0.0, np.pi, 100)
    real = np.real(signal)
    imag = np.imag(signal)
    projections = np.sin(angles[:, None]) * real[None, :] + np.cos(angles[:, None]) * imag[None, :]
    scores = []
    n_fft = max(processing_config.n_fft, int(2 ** np.ceil(np.log2(len(signal)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz)
    bpm = freqs * 60.0
    mask = (
        (bpm >= processing_config.respiration_bpm_min)
        & (bpm <= processing_config.respiration_bpm_max)
    )
    for projection in projections:
        projected = _remove_linear_trend(projection)
        spectrum = np.abs(np.fft.rfft(projected, n=n_fft)) ** 2
        scores.append(float(np.max(spectrum[mask])) if np.any(mask) else float(np.max(spectrum)))
    return projections[int(np.argmax(scores))]


def _remove_linear_trend(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    if len(x) < 3:
        return x - np.mean(x)
    t = np.linspace(-1.0, 1.0, len(x))
    slope, intercept = np.polyfit(t, x, deg=1)
    return x - (slope * t + intercept)


def _fft_bandpass(signal: np.ndarray, sampling_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    if len(x) < 4:
        return x
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate_hz)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return x
    filtered = np.zeros_like(spectrum)
    filtered[mask] = spectrum[mask]
    return np.fft.irfft(filtered, n=len(x))
