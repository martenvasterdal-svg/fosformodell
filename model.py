"""
model.py

Fosformodell (log-transformerad RandomForest med mild viktning).
Exporterar funktionen `run_model` för enkel användning från andra script.

Kräver filen 'logrf_mild_model.pkl' i samma katalog.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np

_MODEL_FILENAME = "logrf_mild_model.pkl"
_DEFAULT_EPS = 0.01


# --------------------------------------------------
# Hjälpfunktioner
# --------------------------------------------------

def _check_0_1(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} måste vara ändligt")
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} måste ligga mellan 0 och 1, fick {v}")
    return v


def _normalize(values: np.ndarray) -> np.ndarray:
    s = values.sum()
    if s <= 0:
        raise ValueError("Summan av andelar är <= 0")
    return values / s


# --------------------------------------------------
# Modellklass
# --------------------------------------------------

class FosforModel:
    def __init__(self) -> None:
        model_path = Path(__file__).with_name(_MODEL_FILENAME)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saknar {_MODEL_FILENAME} i samma katalog som model.py"
            )

        with model_path.open("rb") as f:
            pkg = pickle.load(f)

        self.model = pkg["model"]
        self.eps = pkg.get("eps", _DEFAULT_EPS)
        self.feature_order = pkg["feature_order"]

    def predict(self, features: Dict[str, float]) -> float:
        X = np.array([[features[name] for name in self.feature_order]])
        y_log = self.model.predict(X)
        return float(np.exp(y_log) - self.eps)


# --------------------------------------------------
# Global modellinstans
# --------------------------------------------------

_fosfor_model: FosforModel | None = None


def _get_model() -> FosforModel:
    global _fosfor_model
    if _fosfor_model is None:
        _fosfor_model = FosforModel()
    return _fosfor_model


# --------------------------------------------------
# 🚀 Publikt API
# --------------------------------------------------

def run_model(
    *,
    area_ha: float,
    andel_akermark: float,
    andel_exploaterad: float,
    andel_skogsmark: float,
    andel_ovrig: float,
    andel_leriga: float,
    andel_medelfina: float,
    andel_grova: float,
    auto_normalize: bool = True,
) -> Dict[str, float]:
    """
    Kör fosformodellen.

    Returnerar en dict med:
    - p_kg_per_ha
    - total_kg_per_ar
    """

    # Validera area
    area = float(area_ha)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area_ha måste vara > 0")

    # Validera andelar
    land = np.array([
        _check_0_1("andel_akermark", andel_akermark),
        _check_0_1("andel_exploaterad", andel_exploaterad),
        _check_0_1("andel_skogsmark", andel_skogsmark),
        _check_0_1("andel_ovrig", andel_ovrig),
    ])

    soil = np.array([
        _check_0_1("andel_leriga", andel_leriga),
        _check_0_1("andel_medelfina", andel_medelfina),
        _check_0_1("andel_grova", andel_grova),
    ])

    if auto_normalize:
        land = _normalize(land)
        soil = _normalize(soil)
    else:
        if abs(land.sum() - 1.0) > 1e-6:
            raise ValueError("Markandelar summerar inte till 1")
        if abs(soil.sum() - 1.0) > 1e-6:
            raise ValueError("Jordartsandelar summerar inte till 1")

    features = {
        "f_coarse": soil[2],
        "f_medium": soil[1],
        "f_fine": soil[0],
        "f_field": land[0],
        "f_expl": land[1],
        "f_forest": land[2],
        "f_other": land[3],
    }

    model = _get_model()
    p_kg_per_ha = model.predict(features)
    total_kg = p_kg_per_ha * area

    return {
        "p_kg_per_ha": p_kg_per_ha,
        "total_kg_per_ar": total_kg,
    }
