"""
model.py

Fosformodell (log-transformerad RandomForest med mild viktning).
Laddar en sparad modell i filen 'logrf_mild_model.pkl' i samma mapp.

Indata (per område):
- area_ha            : total area (ha)
- andel_akermark     : fraktion åkermark (0–1)
- andel_exploaterad  : fraktion exploaterad/urban mark (0–1)
- andel_skogsmark    : fraktion skogsmark (0–1)
- andel_ovrig        : fraktion övriga markslag (0–1)
- andel_leriga       : fraktion finkorniga jordar (0–1)
- andel_medelfina    : fraktion mellangrova jordar (0–1)
- andel_grova        : fraktion grova jordar (0–1)

Utdata:
- genomsnittlig P-belastning (kg/ha och år)
- total P-belastning (kg/år)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

_DEFAULT_EPS = 0.01
_MODEL_FILENAME = "logrf_mild_model.pkl"


def _check_0_1(name: str, value: float) -> float:
    try:
        v = float(value)
    except Exception as e:
        raise ValueError(f"{name} måste vara ett tal, fick {value!r}") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} måste vara ändligt (inte NaN/inf), fick {v}")
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} måste ligga mellan 0 och 1, fick {v}")
    return v


def _normalize_group(
    values: np.ndarray,
    group_name: str,
    auto_normalize: bool,
    tol: float = 1e-6,
) -> np.ndarray:
    s = float(values.sum())
    if abs(s - 1.0) <= tol:
        return values

    if not auto_normalize:
        raise ValueError(
            f"{group_name} summerar till {s:.6f} (ska vara 1.0). "
            f"Sätt auto_normalize=True eller justera andelarna."
        )

    if s <= 0:
        raise ValueError(
            f"{group_name} summerar till {s:.6f} (<=0). Kan inte normalisera."
        )

    return values / s


class FosforModel:
    """
    Inkapslar den sparade RandomForest-modellen och ger prediktionsfunktioner.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name(_MODEL_FILENAME)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hittar inte modellen '{model_path}'. "
                f"Lägg '{_MODEL_FILENAME}' i samma mapp som model.py."
            )

        with model_path.open("rb") as f:
            pkg = pickle.load(f)

        self._rf = pkg["model"]
        self._eps: float = float(pkg.get("eps", _DEFAULT_EPS))
        self._feature_order = pkg.get(
            "feature_order",
            ["f_coarse", "f_medium", "f_fine", "f_field", "f_forest", "f_expl", "f_other"],
        )

    def predict_p_kg_per_ha(
        self,
        *,
        andel_akermark: float,
        andel_exploaterad: float,
        andel_skogsmark: float,
        andel_ovrig: float,
        andel_leriga: float,
        andel_medelfina: float,
        andel_grova: float,
        auto_normalize: bool = True,
    ) -> float:
        """
        Predikterar genomsnittlig fosforbelastning (kg/ha och år).

        auto_normalize=True:
          - normaliserar markandelar och jordartsandelar så att de summerar till 1
            om de ligger nära men inte exakt 1 (t.ex. p.g.a. avrundning).
        """

        # Validera 0..1
        a_field = _check_0_1("andel_akermark", andel_akermark)
        a_expl = _check_0_1("andel_exploaterad", andel_exploaterad)
        a_forest = _check_0_1("andel_skogsmark", andel_skogsmark)
        a_other = _check_0_1("andel_ovrig", andel_ovrig)

        s_fine = _check_0_1("andel_leriga", andel_leriga)
        s_med = _check_0_1("andel_medelfina", andel_medelfina)
        s_coarse = _check_0_1("andel_grova", andel_grova)

        # Normalisera grupper
        land = np.array([a_field, a_expl, a_forest, a_other], dtype=float)
        soil = np.array([s_fine, s_med, s_coarse], dtype=float)

        land = _normalize_group(land, "Markandelar (åker+exploaterad+skog+övrig)", auto_normalize)
        soil = _normalize_group(soil, "Jordartsandelar (leriga+medelfina+grova)", auto_normalize)

        # Mappa till modellens feature-namn
        feat_dict = {
            "f_coarse": soil[2],
            "f_medium": soil[1],
            "f_fine": soil[0],
            "f_field": land[0],
            "f_expl": land[1],
            "f_forest": land[2],
            "f_other": land[3],
        }

        X = np.array([[feat_dict[name] for name in self._feature_order]], dtype=float)

        # Modellen är tränad på log(P + eps)
        y_log = self._rf.predict(X)
        p_kg_per_ha = float(np.exp(y_log) - self._eps)
        return p_kg_per_ha

    def predict_total_load_kg(
        self,
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
    ) -> Tuple[float, float]:
        """
        Returnerar (kg/ha·år, kg/år) för området.
        """

        try:
            area = float(area_ha)
        except Exception as e:
            raise ValueError(f"area_ha måste vara ett tal, fick {area_ha!r}") from e
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"area_ha måste vara > 0 och ändligt, fick {area}")

        p_kg_per_ha = self.predict_p_kg_per_ha(
            andel_akermark=andel_akermark,
            andel_exploaterad=andel_exploaterad,
            andel_skogsmark=andel_skogsmark,
            andel_ovrig=andel_ovrig,
            andel_leriga=andel_leriga,
            andel_medelfina=andel_medelfina,
            andel_grova=andel_grova,
            auto_normalize=auto_normalize,
        )
        total_kg_per_year = area * p_kg_per_ha
        return p_kg_per_ha, total_kg_per_year


# Bekväm modulnivå-API
try:
    fosfor_model = FosforModel()
except FileNotFoundError:
    fosfor_model = None


def predict_p_kg_per_ha(
    *,
    area_ha: float = 1.0,  # finns för att matcha ditt indata-schema, men används inte här
    andel_akermark: float,
    andel_exploaterad: float,
    andel_skogsmark: float,
    andel_ovrig: float,
    andel_leriga: float,
    andel_medelfina: float,
    andel_grova: float,
    auto_normalize: bool = True,
) -> float:
    if fosfor_model is None:
        raise RuntimeError(
            f"Modellen är inte laddad. Kontrollera att '{_MODEL_FILENAME}' finns i samma mapp som model.py."
        )
    return fosfor_model.predict_p_kg_per_ha(
        andel_akermark=andel_akermark,
        andel_exploaterad=andel_exploaterad,
        andel_skogsmark=andel_skogsmark,
        andel_ovrig=andel_ovrig,
        andel_leriga=andel_leriga,
        andel_medelfina=andel_medelfina,
        andel_grova=andel_grova,
        auto_normalize=auto_normalize,
    )


def predict_total_load_kg(
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
) -> Tuple[float, float]:
    if fosfor_model is None:
        raise RuntimeError(
            f"Modellen är inte laddad. Kontrollera att '{_MODEL_FILENAME}' finns i samma mapp som model.py."
        )
    return fosfor_model.predict_total_load_kg(
        area_ha=area_ha,
        andel_akermark=andel_akermark,
        andel_exploaterad=andel_exploaterad,
        andel_skogsmark=andel_skogsmark,
        andel_ovrig=andel_ovrig,
        andel_leriga=andel_leriga,
        andel_medelfina=andel_medelfina,
        andel_grova=andel_grova,
        auto_normalize=auto_normalize,
    )


