import numpy as np


def amplitude_phase_features(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    amplitude = np.abs(data)
    phase = np.unwrap(np.angle(data), axis=0)
    return np.concatenate([amplitude, phase], axis=1)


def amplitude_phase_maps(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return amplitude and unwrapped phase maps for CSI or CIR time series."""
    data = np.asarray(data)
    amplitude = np.abs(data)
    phase = np.unwrap(np.angle(data), axis=0)
    return amplitude, phase


def statistical_features(data: np.ndarray) -> np.ndarray:
    """Build compact statistics for PCA/SVM/KNN baselines."""
    data = np.asarray(data)
    amplitude = np.abs(data)
    phase = np.unwrap(np.angle(data), axis=0)
    blocks = []
    for arr in (amplitude, phase):
        blocks.extend(
            [
                np.mean(arr, axis=0),
                np.std(arr, axis=0),
                np.max(arr, axis=0),
                np.min(arr, axis=0),
            ]
        )
    return np.concatenate([x.reshape(-1) for x in blocks])


def cir_time_image(cir: np.ndarray, use_phase: bool = True) -> np.ndarray:
    """Convert CIR time series to an image-like tensor for CNN models."""
    cir = np.asarray(cir)
    amplitude = np.abs(cir)
    if not use_phase:
        return amplitude[None, :, :]
    phase = np.unwrap(np.angle(cir), axis=0)
    return np.stack([amplitude, phase], axis=0)


def spectrum_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal).reshape(-1)
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    signal = signal - np.mean(signal)
    n_fft = max(n_fft, int(2 ** np.ceil(np.log2(len(signal)))))
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz)
    return frequencies, spectrum


def stft_features(
    signal: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int | None = None,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a low-frequency time-frequency map for respiration analysis."""
    signal = np.asarray(signal).reshape(-1)
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    signal = signal - np.mean(signal)
    if nperseg is None:
        nperseg = min(len(signal), max(512, len(signal) // 3))
    nperseg = min(int(nperseg), len(signal))
    noverlap = min(max(int(nperseg * 0.9), 0), nperseg - 1)
    n_fft = max(int(n_fft), int(2 ** np.ceil(np.log2(max(nperseg, 1)))))
    try:
        from scipy.signal import stft

        frequencies, times, zxx = stft(
            signal,
            fs=sampling_rate_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        return frequencies, times, np.abs(zxx)
    except ImportError:
        return _numpy_stft(signal, sampling_rate_hz, nperseg=nperseg, n_fft=n_fft)


def build_feature_bundle(
    clean_csi: np.ndarray,
    clean_cir: np.ndarray,
    respiration_signal: np.ndarray,
    sampling_rate_hz: float,
) -> dict[str, np.ndarray]:
    """Extract task-related features used by reports, baselines and optional models."""
    if clean_csi.ndim == 3:
        clean_csi = clean_csi[:, :, 0]
    if clean_cir.ndim == 3:
        clean_cir = clean_cir[:, :, 0]

    csi_amplitude, csi_phase = amplitude_phase_maps(clean_csi)
    cir_amplitude, cir_phase = amplitude_phase_maps(clean_cir)
    frequency_hz, spectrum = spectrum_features(respiration_signal, sampling_rate_hz)
    stft_frequency_hz, stft_time_s, time_frequency = stft_features(
        respiration_signal,
        sampling_rate_hz,
    )

    return {
        "csi_amplitude_mean": np.mean(csi_amplitude, axis=0),
        "csi_phase_mean": np.mean(csi_phase, axis=0),
        "cir_amplitude": cir_amplitude,
        "cir_phase": cir_phase,
        "cir_variance": np.var(clean_cir, axis=0),
        "respiration_frequency_hz": frequency_hz,
        "respiration_spectrum": spectrum,
        "stft_frequency_hz": stft_frequency_hz,
        "stft_time_s": stft_time_s,
        "time_frequency_map": time_frequency,
        "statistical_features": statistical_features(clean_cir),
    }


def _numpy_stft(
    signal: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int = 128,
    n_fft: int = 8192,
    overlap_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = max(8, min(int(nperseg), len(signal)))
    hop = max(1, int(nperseg * (1.0 - overlap_ratio)))
    window = np.hanning(nperseg)
    frames = []
    times = []
    for start in range(0, max(len(signal) - nperseg + 1, 1), hop):
        frame = signal[start : start + nperseg]
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
        frames.append(np.abs(np.fft.rfft(frame * window, n=n_fft)))
        times.append((start + nperseg / 2) / sampling_rate_hz)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate_hz)
    return frequencies, np.asarray(times), np.asarray(frames).T
