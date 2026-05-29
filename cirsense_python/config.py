from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent


def default_used_subcarriers() -> np.ndarray:
    """Subcarrier indices used by the public CIRSense MATLAB scripts."""
    ranges = [
        np.arange(-984, -770),
        np.arange(-765, -514),
        np.arange(-509, -11),
        np.arange(12, 510),
        np.arange(515, 766),
        np.arange(771, 985),
    ]
    return np.concatenate(ranges).astype(int)


@dataclass
class SignalConfig:
    carrier_frequency_hz: float = 5.25e9
    bandwidth_hz: float = 160e6
    fft_size: int = 2048
    speed_of_light: float = 3e8
    los_distance_m: float = 0.6
    tap_min: int = -20
    tap_max: int = 49
    used_subcarriers: np.ndarray = field(default_factory=default_used_subcarriers)

    @property
    def tap_values(self) -> np.ndarray:
        return np.arange(self.tap_min, self.tap_max + 1)

    @property
    def center_tap_index(self) -> int:
        matches = np.where(self.tap_values == 0)[0]
        if len(matches) == 0:
            raise ValueError("tap range must include zero")
        return int(matches[0])


@dataclass
class ProcessingConfig:
    sampling_rate_hz: float = 200.0
    smoothing_window_s: float = 0.3
    respiration_bpm_min: float = 10.0
    respiration_bpm_max: float = 37.0
    n_fft: int = 8192
    n_delay_candidates: int = 200
    candidate_stride: int = 10


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class DataConfig:
    dataset_root: Path = (
        PROJECT_DIR
        / "cirsense-dataset-real-world-80211ax-csi-measurements-wireless-sensing"
        / "CIRSense_dataset"
    )
    output_root: Path = PROJECT_DIR / "outputs"
    test_output_root: Path = PROJECT_DIR / "output_test"
    distance_subdir: str = "distance"
    respiration_subdir: str = "breathe"
    nlos_respiration_subdir: str = "nlos_breathe"
    multitarget_subdir: str = "multitarget"


@dataclass
class ProjectConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def build_default_config() -> ProjectConfig:
    return ProjectConfig()
