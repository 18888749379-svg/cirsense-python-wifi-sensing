from pathlib import Path
from typing import Optional

import numpy as np


def plot_cir_magnitude(cir: np.ndarray, output_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    cir = np.asarray(cir)
    if cir.ndim == 3:
        cir = cir[:, :, 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(np.abs(cir).T, aspect="auto", origin="lower")
    ax.set_xlabel("Time index")
    ax.set_ylabel("CIR tap index")
    ax.set_title("CIR magnitude")
    fig.colorbar(image, ax=ax)
    _save_or_show(fig, output_path)
    return fig


def plot_respiration_signal(signal: np.ndarray, time_s: Optional[np.ndarray], output_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    signal = np.asarray(signal).reshape(-1)
    x_axis = np.arange(len(signal)) if time_s is None else np.asarray(time_s).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x_axis, np.real(signal), color="#1f77b4", linewidth=1.6)
    ax.set_xlabel("Time (s)" if time_s is not None else "Time index")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Processed respiration waveform")
    ax.grid(alpha=0.25)
    _save_or_show(fig, output_path)
    return fig


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction comparison")
    _save_or_show(fig, output_path)
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str] | tuple[str, ...],
    output_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth class")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax)
    _save_or_show(fig, output_path)
    return fig


def plot_loss_curve(
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    output_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="train")
    if len(val_loss):
        ax.plot(epochs, val_loss, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_or_show(fig, output_path)
    return fig


def plot_spectrum(
    frequencies_hz: np.ndarray,
    spectrum: np.ndarray,
    output_path: Optional[Path] = None,
    max_bpm: float = 60.0,
    respiration_range_bpm: tuple[float, float] = (10.0, 37.0),
):
    import matplotlib.pyplot as plt

    bpm = np.asarray(frequencies_hz) * 60.0
    spectrum = np.asarray(spectrum)
    mask = bpm <= max_bpm
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(bpm[mask], spectrum[mask])
    visible_power = spectrum[mask]
    finite_power = visible_power[np.isfinite(visible_power)]
    y_top = float(np.max(finite_power)) if finite_power.size else 1.0
    y_bottom = float(np.min(finite_power)) if finite_power.size else 0.0
    if y_top <= y_bottom:
        y_top = y_bottom + 1.0
    ax.set_ylim(min(0.0, y_bottom), y_top * 1.18)
    band_mask = (
        mask
        & (bpm >= respiration_range_bpm[0])
        & (bpm <= respiration_range_bpm[1])
    )
    if np.any(band_mask):
        band_bpm = bpm[band_mask]
        band_power = spectrum[band_mask]
        peak_index = int(np.argmax(band_power))
        peak_bpm = float(band_bpm[peak_index])
        peak_power = float(band_power[peak_index])
        ax.axvspan(respiration_range_bpm[0], respiration_range_bpm[1], color="#2ca02c", alpha=0.08)
        ax.scatter([peak_bpm], [peak_power], color="#d62728", zorder=3)
        x_offset, y_offset, ha, va = _peak_label_offset(
            peak_bpm,
            peak_power,
            float(np.min(bpm[mask])) if np.any(mask) else 0.0,
            float(np.max(bpm[mask])) if np.any(mask) else max_bpm,
            y_top,
        )
        ax.annotate(
            f"peak {peak_bpm:.1f} bpm",
            xy=(peak_bpm, peak_power),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
            annotation_clip=False,
        )
    ax.set_xlabel("Frequency (bpm)")
    ax.set_ylabel("Power")
    ax.set_title("Processed respiration spectrum")
    ax.grid(alpha=0.25)
    _save_or_show(fig, output_path)
    return fig


def plot_time_frequency(
    frequencies_hz: np.ndarray,
    times_s: np.ndarray,
    time_frequency_map: np.ndarray,
    output_path: Optional[Path] = None,
    max_bpm: float = 60.0,
    respiration_range_bpm: tuple[float, float] = (10.0, 37.0),
):
    import matplotlib.pyplot as plt

    bpm = np.asarray(frequencies_hz) * 60.0
    times_s = np.asarray(times_s)
    tf = np.asarray(time_frequency_map)
    mask = (bpm >= 0.0) & (bpm <= max_bpm)
    if not np.any(mask):
        mask = np.ones_like(bpm, dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 4))
    x0 = float(times_s[0]) if len(times_s) else 0.0
    x1 = float(times_s[-1]) if len(times_s) else 1.0
    if x0 == x1:
        x1 = x0 + 1.0
    y_values = bpm[mask]
    y0 = float(y_values[0])
    y1 = float(y_values[-1])
    if y0 == y1:
        y1 = y0 + 1.0
    tf_display = tf[mask, :]
    if tf_display.size:
        upper = np.percentile(tf_display, 98)
        if upper > 0:
            tf_display = np.clip(tf_display, 0, upper)
    image = ax.imshow(
        tf_display,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, y0, y1],
    )
    ax.axhspan(respiration_range_bpm[0], respiration_range_bpm[1], color="#2ca02c", alpha=0.08)
    ax.axhline(respiration_range_bpm[0], color="#2ca02c", linestyle="--", linewidth=1)
    ax.axhline(respiration_range_bpm[1], color="#2ca02c", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (bpm)")
    ax.set_title("Respiration time-frequency map")
    fig.colorbar(image, ax=ax)
    _save_or_show(fig, output_path)
    return fig


def plot_delay_variance(
    delay_candidates: np.ndarray,
    variance_curve: np.ndarray,
    output_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    delay_candidates = np.asarray(delay_candidates)
    variance_curve = np.asarray(variance_curve)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(delay_candidates, variance_curve, linewidth=1.6)
    if len(variance_curve):
        peak_index = int(np.argmax(variance_curve))
        peak_x = float(delay_candidates[peak_index])
        peak_y = float(variance_curve[peak_index])
        finite_variance = variance_curve[np.isfinite(variance_curve)]
        y_top = float(np.max(finite_variance)) if finite_variance.size else 1.0
        y_bottom = float(np.min(finite_variance)) if finite_variance.size else 0.0
        if y_top <= y_bottom:
            y_top = y_bottom + 1.0
        ax.set_ylim(min(0.0, y_bottom), y_top * 1.18)
        ax.scatter([peak_x], [peak_y], color="#d62728", zorder=3)
        x_offset, y_offset, ha, va = _peak_label_offset(
            peak_x,
            peak_y,
            float(np.min(delay_candidates)),
            float(np.max(delay_candidates)),
            y_top,
        )
        ax.annotate(
            f"best {peak_x:.3f} taps",
            xy=(peak_x, peak_y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
            annotation_clip=False,
        )
    ax.set_xlabel("Delay candidate (tap)")
    ax.set_ylabel("Variance")
    ax.set_title("Dynamic path alignment variance")
    ax.grid(alpha=0.25)
    _save_or_show(fig, output_path)
    return fig


def plot_distance_profile(
    distance_axis_m: np.ndarray,
    mean_profile: np.ndarray,
    detection_score: np.ndarray,
    target_distances_m: np.ndarray,
    output_path: Optional[Path] = None,
    ground_truth_distances_m: Optional[np.ndarray] = None,
):
    import matplotlib.pyplot as plt

    distance_axis_m = np.asarray(distance_axis_m, dtype=float)
    mean_profile = _normalize_for_plot(mean_profile)
    detection_score = _normalize_for_plot(detection_score)
    target_distances_m = np.asarray(target_distances_m, dtype=float).reshape(-1)

    valid = distance_axis_m >= 0.0
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(
        distance_axis_m[valid],
        mean_profile[valid],
        label="mean CIR magnitude",
        linewidth=1.6,
    )
    ax.plot(
        distance_axis_m[valid],
        detection_score[valid],
        label="multi-target score",
        linewidth=1.4,
        linestyle="--",
    )

    for idx, distance in enumerate(target_distances_m, start=1):
        nearest = int(np.argmin(np.abs(distance_axis_m - distance)))
        ax.scatter([distance], [detection_score[nearest]], color="#d62728", zorder=4)
        y_offset = -22 if detection_score[nearest] > 0.85 else 10
        ax.annotate(
            f"target {idx}: {distance:.2f} m",
            xy=(distance, detection_score[nearest]),
            xytext=(8, y_offset),
            textcoords="offset points",
            fontsize=9,
            va="top" if y_offset < 0 else "bottom",
        )

    if ground_truth_distances_m is not None:
        ground_truth_distances_m = np.asarray(ground_truth_distances_m, dtype=float).reshape(-1)
        for idx, distance in enumerate(ground_truth_distances_m, start=1):
            ax.axvline(distance, color="#555555", linewidth=1, alpha=0.35)
            ax.text(
                distance,
                0.03,
                f"GT {idx}: {distance:.2f} m",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
                color="#555555",
            )
    else:
        ground_truth_distances_m = np.asarray([], dtype=float)

    marks = np.concatenate([target_distances_m, ground_truth_distances_m])
    if len(marks):
        x_max = max(12.0, float(np.nanmax(marks)) + 2.0)
    else:
        x_max = 12.0
    positive_axis = distance_axis_m[distance_axis_m >= 0.0]
    if len(positive_axis):
        x_max = min(x_max, float(np.nanmax(positive_axis)))
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.05, 1.12)

    ax.set_xlabel("Estimated distance (m)")
    ax.set_ylabel("Normalized profile")
    ax.set_title("Multi-target distance profile")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    _save_or_show(fig, output_path)
    return fig


def _normalize_for_plot(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.nanmin(values)
    peak = np.nanmax(values)
    if peak <= 1e-12:
        return np.zeros_like(values)
    return values / peak


def _peak_label_offset(
    peak_x: float,
    peak_y: float,
    x_min: float,
    x_max: float,
    y_top: float,
) -> tuple[int, int, str, str]:
    x_mid = (x_min + x_max) / 2.0
    x_offset = -10 if peak_x >= x_mid else 10
    ha = "right" if x_offset < 0 else "left"

    near_top = y_top > 0 and peak_y >= 0.78 * y_top
    y_offset = -14 if near_top else 12
    va = "top" if y_offset < 0 else "bottom"
    return x_offset, y_offset, ha, va


def _save_or_show(fig, output_path: Optional[Path]) -> None:
    if output_path is None:
        fig.show()
        return
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
