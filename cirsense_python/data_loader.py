from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Iterable, Optional
import zlib

import numpy as np

from config import SignalConfig


_MISSING = object()
_MAT_NUMPY_DTYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    9: "<f8",
    12: "<i8",
    13: "<u8",
}


@dataclass
class CSISample:
    file_path: Path
    csi: np.ndarray
    time_s: Optional[np.ndarray] = None
    distance_m: Optional[float] = None
    target_distances_m: Optional[np.ndarray] = None
    bpm: Optional[float] = None
    ground_truth_signal: Optional[np.ndarray] = None
    ground_truth_time_s: Optional[np.ndarray] = None


def list_data_files(data_dir: Path, extensions: Iterable[str] = (".mat", ".npz")) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    files: list[Path] = []
    for extension in extensions:
        files.extend(data_dir.glob(f"*{extension}"))
    return sorted(files)


def load_csi_sample(file_path: Path, signal_config: Optional[SignalConfig] = None) -> CSISample:
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".mat":
        sample = _load_mat_sample(file_path)
    elif file_path.suffix.lower() == ".npz":
        sample = _load_npz_sample(file_path)
    else:
        raise ValueError(f"Unsupported data file type: {file_path.suffix}")

    csi = _as_complex_csi(sample.csi)
    if signal_config is not None:
        csi = select_configured_subcarriers(csi, signal_config)

    inferred_distance = sample.distance_m
    if inferred_distance is None:
        inferred_distance = infer_distance_from_filename(file_path.name)

    return CSISample(
        file_path=sample.file_path,
        csi=csi,
        time_s=sample.time_s,
        distance_m=inferred_distance,
        target_distances_m=sample.target_distances_m,
        bpm=sample.bpm,
        ground_truth_signal=sample.ground_truth_signal,
        ground_truth_time_s=sample.ground_truth_time_s,
    )


def select_configured_subcarriers(csi: np.ndarray, signal_config: SignalConfig) -> np.ndarray:
    """Select configured active subcarriers when raw data contains full FFT bins."""
    csi = _ensure_time_subcarrier_antenna(csi)
    n_subcarriers = csi.shape[1]
    expected = len(signal_config.used_subcarriers)
    if n_subcarriers == expected:
        return csi
    if n_subcarriers == 2025:
        indices = (signal_config.used_subcarriers + 1012).astype(int)
        return csi[:, indices, :]
    if n_subcarriers >= signal_config.fft_size:
        indices = (signal_config.used_subcarriers + signal_config.fft_size // 2).astype(int)
        return csi[:, indices, :]
    raise ValueError(
        "CSI subcarrier dimension does not match the configured active set. "
        f"Got {n_subcarriers}, expected {expected} or at least {signal_config.fft_size}."
    )


def _load_mat_sample(file_path: Path) -> CSISample:
    try:
        from scipy.io import loadmat

        raw = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except ImportError:
        raw = _load_mat_v5_numeric(file_path)

    csi = _first_existing(raw, ["csidata", "csi", "CSI", "csi_data"])
    time_s = _first_existing(raw, ["t_slice", "time", "timestamp", "timestamps"], default=None)
    distance = _first_existing(raw, ["ground_truth_distance", "distance", "distance_m"], default=None)
    bpm = _first_existing(raw, ["bpm", "respiration_rate", "respiration_rate_gt"], default=None)
    gt = _first_existing(raw, ["gt", "ground_truth_signal"], default=None)
    gt_t = _first_existing(raw, ["gt_t", "ground_truth_time", "gt_time"], default=None)

    return CSISample(
        file_path=file_path,
        csi=csi,
        time_s=_maybe_array(time_s),
        distance_m=_maybe_scalar(distance),
        target_distances_m=_maybe_vector(distance),
        bpm=_maybe_scalar(bpm),
        ground_truth_signal=_maybe_array(gt),
        ground_truth_time_s=_maybe_array(gt_t),
    )


def _load_npz_sample(file_path: Path) -> CSISample:
    raw = np.load(file_path, allow_pickle=True)
    csi = _first_existing(raw, ["csidata", "csi", "CSI", "csi_data"])
    time_s = _first_existing(raw, ["t_slice", "time", "timestamp", "timestamps"], default=None)
    distance = _first_existing(raw, ["ground_truth_distance", "distance", "distance_m"], default=None)
    bpm = _first_existing(raw, ["bpm", "respiration_rate", "respiration_rate_gt"], default=None)
    gt = _first_existing(raw, ["gt", "ground_truth_signal"], default=None)
    gt_t = _first_existing(raw, ["gt_t", "ground_truth_time", "gt_time"], default=None)
    return CSISample(
        file_path=file_path,
        csi=csi,
        time_s=_maybe_array(time_s),
        distance_m=_maybe_scalar(distance),
        target_distances_m=_maybe_vector(distance),
        bpm=_maybe_scalar(bpm),
        ground_truth_signal=_maybe_array(gt),
        ground_truth_time_s=_maybe_array(gt_t),
    )


def _first_existing(mapping, keys: list[str], default=_MISSING):
    for key in keys:
        if key in mapping:
            return mapping[key]
    if default is not _MISSING:
        return default
    raise KeyError(f"None of the expected keys were found: {keys}")


def infer_distance_from_filename(file_name: str) -> Optional[float]:
    """Infer distance labels from names like home_2_25m_1.mat or home__3m_1.mat."""
    match = re.search(r"(?<!\d)(\d+(?:_\d+)?)m", file_name)
    if not match:
        return None
    value = match.group(1).replace("_", ".")
    try:
        return float(value)
    except ValueError:
        return None


def _as_complex_csi(csi: np.ndarray) -> np.ndarray:
    csi = np.asarray(csi)
    if np.iscomplexobj(csi):
        return _ensure_time_subcarrier_antenna(csi.astype(np.complex128))

    if csi.shape[-1] == 2:
        complex_csi = csi[..., 0] + 1j * csi[..., 1]
        return _ensure_time_subcarrier_antenna(complex_csi.astype(np.complex128))

    raise ValueError(
        "CSI must be complex-valued or store real/imaginary parts in the last dimension."
    )


def _ensure_time_subcarrier_antenna(csi: np.ndarray) -> np.ndarray:
    csi = np.asarray(csi)
    if csi.ndim == 2:
        return csi[:, :, None]
    if csi.ndim != 3:
        raise ValueError(f"CSI must have 2 or 3 dimensions, got shape {csi.shape}")
    return csi


def _load_mat_v5_numeric(file_path: Path) -> dict[str, np.ndarray]:
    """Small MATLAB v5 reader for numeric arrays.

    This fallback keeps the project runnable in lightweight environments that
    do not have scipy installed. It supports the compressed complex-double
    arrays used by the CIRSense dataset.
    """
    data = Path(file_path).read_bytes()
    if not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise ImportError("Only MATLAB v5 .mat files are supported without scipy.")

    variables: dict[str, np.ndarray] = {}
    offset = 128
    while offset + 8 <= len(data):
        dtype, nbytes, data_offset, next_offset, _ = _read_mat_tag(data, offset)
        if dtype == 15:
            chunk = zlib.decompress(data[data_offset : data_offset + nbytes])
            variables.update(_parse_mat_chunk(chunk))
        elif dtype == 14:
            name, value = _parse_mat_matrix(data, data_offset, nbytes)
            variables[name] = value
        offset = next_offset
    return variables


def _parse_mat_chunk(chunk: bytes) -> dict[str, np.ndarray]:
    variables: dict[str, np.ndarray] = {}
    offset = 0
    while offset + 8 <= len(chunk):
        dtype, nbytes, data_offset, next_offset, _ = _read_mat_tag(chunk, offset)
        if dtype == 14:
            name, value = _parse_mat_matrix(chunk, data_offset, nbytes)
            variables[name] = value
        offset = next_offset
    return variables


def _parse_mat_matrix(buf: bytes, offset: int, nbytes: int) -> tuple[str, np.ndarray]:
    end = offset + nbytes
    dtype, size, data_offset, next_offset, _ = _read_mat_tag(buf, offset)
    flags = struct.unpack("<II", buf[data_offset : data_offset + size])
    is_complex = bool(flags[0] & 0x0800)
    offset = next_offset

    dtype, size, data_offset, next_offset, _ = _read_mat_tag(buf, offset)
    dims = tuple(struct.unpack("<" + "i" * (size // 4), buf[data_offset : data_offset + size]))
    offset = next_offset

    dtype, size, data_offset, next_offset, _ = _read_mat_tag(buf, offset)
    name = buf[data_offset : data_offset + size].decode("latin1")
    offset = next_offset

    real = None
    imag = None
    while offset + 8 <= end:
        dtype, size, data_offset, next_offset, _ = _read_mat_tag(buf, offset)
        if dtype in _MAT_NUMPY_DTYPES:
            np_dtype = np.dtype(_MAT_NUMPY_DTYPES[dtype])
            arr = np.frombuffer(buf, dtype=np_dtype, count=size // np_dtype.itemsize, offset=data_offset)
            arr = arr.reshape(dims, order="F").astype(np.float64, copy=True)
            if real is None:
                real = arr
            else:
                imag = arr
        offset = next_offset

    if real is None:
        raise ValueError(f"Variable {name} does not contain numeric double data.")
    if is_complex:
        if imag is None:
            imag = np.zeros_like(real)
        return name, real + 1j * imag
    return name, real


def _read_mat_tag(buf: bytes, offset: int) -> tuple[int, int, int, int, bool]:
    if offset + 4 > len(buf):
        raise ValueError("Incomplete MATLAB tag.")
    first, second = struct.unpack("<HH", buf[offset : offset + 4])
    if second != 0:
        dtype = first
        nbytes = second
        data_offset = offset + 4
        next_offset = offset + 8
        return dtype, nbytes, data_offset, next_offset, True
    dtype, nbytes = struct.unpack("<II", buf[offset : offset + 8])
    data_offset = offset + 8
    next_offset = data_offset + ((nbytes + 7) // 8) * 8
    return dtype, nbytes, data_offset, next_offset, False


def _maybe_array(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return None
    return arr


def _maybe_scalar(value) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _maybe_vector(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.size == 0:
        return None
    return arr.reshape(-1)
