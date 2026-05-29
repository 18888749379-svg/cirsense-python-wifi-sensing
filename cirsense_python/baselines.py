from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from cirsense_core import estimate_respiration_bpm
from config import ProcessingConfig


BaselineKind = Literal["svm", "knn", "random_forest"]


@dataclass
class PCABaseline:
    kind: BaselineKind = "svm"
    n_components: int = 20

    def fit(self, x: np.ndarray, y: np.ndarray):
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        if self.kind == "svm":
            estimator = SVR(kernel="rbf", C=10.0, gamma="scale")
        elif self.kind == "knn":
            estimator = KNeighborsRegressor(n_neighbors=max(1, min(5, len(x))))
        elif self.kind == "random_forest":
            estimator = RandomForestRegressor(n_estimators=200, random_state=42)
        else:
            raise ValueError(f"Unknown baseline kind: {self.kind}")

        self.pipeline_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", _pca_or_passthrough(PCA, x, self.n_components)),
                ("model", estimator),
            ]
        )
        self.pipeline_.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Call fit before predict.")
        return self.pipeline_.predict(x)


@dataclass
class PCAClassifier:
    kind: Literal["svm", "knn"] = "svm"
    n_components: int = 20

    def fit(self, x: np.ndarray, y: np.ndarray):
        from sklearn.decomposition import PCA
        from sklearn.dummy import DummyClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        if len(np.unique(y)) < 2:
            estimator = DummyClassifier(strategy="most_frequent")
        elif self.kind == "svm":
            estimator = SVC(kernel="rbf", C=10.0, gamma="scale")
        elif self.kind == "knn":
            estimator = KNeighborsClassifier(n_neighbors=max(1, min(5, len(x))))
        else:
            raise ValueError(f"Unknown classifier kind: {self.kind}")

        self.pipeline_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", _pca_or_passthrough(PCA, x, self.n_components)),
                ("model", estimator),
            ]
        )
        self.pipeline_.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Call fit before predict.")
        return self.pipeline_.predict(x)


def fft_respiration_baseline(
    signal: np.ndarray,
    time_s: Optional[np.ndarray],
    processing_config: ProcessingConfig,
) -> float:
    return estimate_respiration_bpm(signal, time_s, processing_config)


def _safe_pca_components(x: np.ndarray, requested: int) -> int:
    x = np.asarray(x)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("PCA baselines expect a non-empty 2D feature matrix.")
    return max(1, min(int(requested), int(x.shape[1]), int(x.shape[0])))


def _pca_or_passthrough(pca_cls, x: np.ndarray, requested: int):
    if len(x) < 2:
        return "passthrough"
    return pca_cls(n_components=_safe_pca_components(x, requested))
