"""
model.py

Fosformodell baserad på log-transformerad RandomForest med mild viktning.
Modellen förväntas vara sparad i filen 'logrf_mild_model.pkl' i samma mapp.

Indata (per område):
- area_ha            : total area (ha)
- andel_akermark     : fraktion åkermark (0–1)
- andel_exploaterad  : fraktion exploaterad/urban mark (0–1)
- andel_skogsmark    : fraktion skogsmark (0–1)
- andel_ovrig        : fraktion övriga markslag (0–1)
- andel_leriga       : fraktion finkorniga jordar (lera m.m.) (0–1)
- andel_medelfina    : fraktion mellangrova jordar (0–1)
- andel_grova        : fraktion grova jordar (0–1)

Utdata:
- genomsnittlig P-belastning i kg/ha och år
- total P-belastning i kg/år
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

# Standardvärde för log-transformeringen i modellen
_DEFAULT_EPS = 0.01

# Filnamn för den sparade modellen (RandomForest)
_MODEL_FILENAME = "logrf_mild_model.pkl"


class FosforModel:
    """
    Inkapslar den tränade RandomForest-modellen och ger
    enkla gränssnittsfunktioner för prediktion.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).with_name(_MODEL_FILENAME)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Hittar inte modellen '{model_path}'. "
                f"Säkerställ att du har sparat logrf_mild_model.pkl i samma mapp som model.py."
            )

        with model_path.open("rb") as f:
            pkg = pickle.load(f)

        self._rf = pkg["model"]
        self._eps: float = pkg.get("eps", _DEFAULT_EPS)

        # Ordning på features i modellen
        # Dessa måste matcha vad modellen tränades med.
        self._feature_order = pkg.get(
            "feature_order",
            [
                "f_coarse",
                "f_medium",
                "f_fine",
                "f_field",
                "f_forest",
                "f_expl",
                "f_other",
            ],
        )

    def _build_feature_vector(
        self,
        andel_akermark: float,
        andel_exploaterad: float,
        andel_skogsmark: float,
        andel_ovrig: float,
        andel_leriga: float,
        andel_medelfina: float,
        andel_grova: float,
    ) -> np.ndarray:
        """
        Bygger upp en feature-vektor i samma ordning som modellen tränades på.

        Mapping:
        - f_field  <- andel_akermark
        - f_expl   <- andel_exploaterad
        - f_forest <- andel_skogsmark
        - f_other  <- andel_ovrig
        - f_fine   <- andel_leriga
        - f_medium <- andel_medelfina
        - f_coarse <- andel_grova
        """

        # Jordarter
        f_coarse = float(andel_grova)
        f_medium = float(andel_medelfina)
        f_fine = float(andel_leriga)

        # Markanvändning
        f_field = float(andel_akermark)
        f_expl = float(andel_exploaterad)
        f_forest = float(andel_skogsmark)
        f_other = float(andel_ovrig)

        # Lägg i en dict enligt feature-namnen
        feat_dict = {
            "f_coarse": f_coarse,
            "f_medium": f_medium,
            "f_fine": f_fine,
            "f_field": f_field,
            "f_forest": f_forest,
            "f_expl": f_expl,
            "f_other": f_other,
        }

        # Bygg en array i exakt samma ordning som modellen förväntar sig
        x = np.array([[feat_dict[name] for name in self._feature_order]], dtype=float)

        return x

    def predict_p_kg_per_ha(
        self,
        andel_akermark: float,
        andel_exploaterad: float,
        andel_skogsmark: float,
        andel_ovrig: float,
        andel_leriga: float,
        andel_medelfina: float,
        andel_grova: float,
    ) -> float:
        """
        Predikterar genomsnittlig fosforbelastning (kg/ha och år)
        för ett område givet markanvändning och jordarter.
        """

        X = self._build_feature_vector(
            andel_akermark=andel_akermark,
            andel_exploaterad=andel_exploaterad,
            andel_skogsmark=andel_skogsmark,
            andel_ovrig=andel_ovrig,
            andel_leriga=andel_leriga,
            andel_medelfina=andel_medelfina,
            andel_grova=andel_grova,
        )

        # Modellen är tränad på log(P + eps)
        y_log = self._rf.predict(X)
        p_kg_per_ha = float(np.exp(y_log) - self._eps)

        return p_kg_per_ha

    def predict_total_load_kg(
        self,
        area_ha: float,
        andel_akermark: float,
        andel_exploaterad: float,
        andel_skogsmark: float,
        andel_ovrig: float,
        andel_leriga: float,
        andel_medelfina: float,
        andel_grova: float,
    ) -> Tuple[float, float]:
        """
        Predikterar både:
        - genomsnittlig belastning (kg/ha och år)
        - total belastning (kg/år) för området.
        """

        p_kg_per_ha = self.predict_p_kg_per_ha(
            andel_akermark=andel_akermark,
            andel_exploaterad=andel_exploaterad,
            andel_skogsmark=andel_skogsmark,
            andel_ovrig=andel_ovrig,
            andel_leriga=andel_leriga,
            andel_medelfina=andel_medelfina,
            andel_grova=andel_grova,
        )

        total_kg_per_year = float(area_ha) * p_kg_per_ha
        return p_kg_per_ha, total_kg_per_year


# Skapa en standardinstans som kan importeras direkt:
# from model import fosfor_model, predict_p_kg_per_ha, predict_total_load_kg
try:
    fosfor_model = FosforModel()
except FileNotFoundError:
    # Om modellen inte finns ännu vill vi inte krascha direkt vid import i alla sammanhang
    fosfor_model = None


def predict_p_kg_per_ha(
    area_ha: float,  # tas med för konsistent signatur, används ej i denna funktion
    andel_akermark: float,
    andel_exploaterad: float,
    andel_skogsmark: float,
    andel_ovrig: float,
    andel_leriga: float,
    andel_medelfina: float,
    andel_grova: float,
) -> float:
    """
    Hjälpfunktion på modulpnivå för att enklare kunna anropa modellen.

    Returnerar genomsnittlig fosforbelastning i kg/ha och år.
    """

    if fosfor_model is None:
        raise RuntimeError(
            "fosfor_model är inte initierad. Kontrollera att 'logrf_mild_model.pkl' "
            "finns i samma mapp som model.py."
        )

    return fosfor_model.predict_p_kg_per_ha(
        andel_akermark=andel_akermark,
        andel_exploaterad=andel_exploaterad,
        andel_skogsmark=andel_skogsmark,
        andel_ovrig=andel_ovrig,
        andel_leriga=andel_leriga,
        andel_medelfina=andel_medelfina,
        andel_grova=andel_grova,
    )


def predict_total_load_kg(
    area_ha: float,
    andel_akermark: float,
    andel_exploaterad: float,
    andel_skogsmark: float,
    andel_ovrig: float,
    andel_leriga: float,
    andel_medelfina: float,
    andel_grova: float,
) -> Tuple[float, float]:
    """
    Hjälpfunktion på modulpnivå som returnerar:
    (genomsnittlig belastning kg/ha och år, total belastning kg/år)
    """

    if fosfor_model is None:
        raise RuntimeError(
            "fosfor_model är inte initierad. Kontrollera att 'logrf_mild_model.pkl' "
            "finns i samma mapp som model.py."
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
    )

