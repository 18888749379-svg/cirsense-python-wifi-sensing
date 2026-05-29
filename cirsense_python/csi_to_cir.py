from dataclasses import dataclass

import numpy as np

from config import SignalConfig


@dataclass
class CIRProjection:
    matrix: np.ndarray
    tap_values: np.ndarray
    used_subcarriers: np.ndarray
    data_subcarrier_indices: np.ndarray


def build_projection_matrix(signal_config: SignalConfig, ridge: float = 1e-8) -> CIRProjection:
    """Build the LS projection matrix from active CSI subcarriers to CIR taps.

    This mirrors the public MATLAB implementation:
    F = ifftshift(ifftshift(dftmtx(2048), 1), 2)
    FL = F(used_sc + 1024 + 1, 1024 + L + 1)
    A = (FL' * FL) \\ FL'
    """
    used_sc = signal_config.used_subcarriers.astype(float)
    taps = signal_config.tap_values.astype(float)
    dft_submatrix = np.exp(
        -2j * np.pi * np.outer(used_sc, taps) / signal_config.fft_size
    )

    gram = dft_submatrix.conj().T @ dft_submatrix
    regularized = gram + ridge * np.eye(gram.shape[0], dtype=np.complex128)
    projection = np.linalg.solve(regularized, dft_submatrix.conj().T)
    return CIRProjection(
        matrix=projection,
        tap_values=signal_config.tap_values,
        used_subcarriers=signal_config.used_subcarriers,
        data_subcarrier_indices=signal_config.used_subcarriers + 1013,
    )


def estimate_cir_from_csi(csi: np.ndarray, projection: CIRProjection) -> np.ndarray:
    """Estimate CIR from CSI.

    Input shape: [time, subcarrier, antenna].
    Output shape: [time, tap, antenna].
    """
    csi = _ensure_csi_shape(csi)
    if csi.shape[1] != projection.matrix.shape[1]:
        raise ValueError(
            f"CSI has {csi.shape[1]} subcarriers, projection expects {projection.matrix.shape[1]}"
        )
    return np.einsum("ls,tsa->tla", projection.matrix, csi)


def apply_fractional_delay_shift(
    csi: np.ndarray,
    delay_taps: float,
    used_subcarriers: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    """Shift CSI in the frequency domain by a fractional delay measured in taps."""
    csi = _ensure_csi_shape(csi)
    phase_shift = np.exp(2j * np.pi * used_subcarriers * delay_taps / fft_size)
    return csi * phase_shift[None, :, None]


def _ensure_csi_shape(csi: np.ndarray) -> np.ndarray:
    csi = np.asarray(csi, dtype=np.complex128)
    if csi.ndim == 2:
        return csi[:, :, None]
    if csi.ndim != 3:
        raise ValueError(f"CSI must have shape [time, subcarrier, antenna], got {csi.shape}")
    return csi
