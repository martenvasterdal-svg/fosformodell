
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Importera DataFrame-varianten från din modell (alternativ B)
from model import run_model_df

# ----------------------- Sidinställningar -----------------------
st.set_page_config(page_title="Fosforbelastning – andelar & area", layout="wide")
st.title("🧮 Fosforbelastning per polygon – andelar & area (ha)")
st.caption(
    "Ange area (ha) och andelar för markanvändning och jordarter per polygon. "
    "Koefficienterna är låsta i din tränade modell (model.py). "
    "Beräkna fosforbelastning i kg/ha/år och kg/år."
)

# ----------------------- Konfiguration -------------------------
LAND_COLS = ["andel_akermark", "andel_exploaterad", "andel_skogsmark", "andel_ovrig"]
SOIL_COLS = ["andel_leriga", "andel_medelfina", "andel_grova"]

# ----------------------- Hjälpfunktioner ------------------------
def make_empty_table(n: int) -> pd.DataFrame:
    """Skapar en startmall med polygon_id, area_ha och standardandelar."""
    data = {
        "polygon_id": [f"P{i+1}" for i in range(n)],
        "area_ha": [10.0] * n,  # default 10 ha – kan ändras i tabellen
        # Markandelar (summa 1)
        "andel_akermark":   [0.25] * n,
        "andel_exploaterad":[0.10] * n,
        "andel_skogsmark":  [0.50] * n,
        "andel_ovrig":      [0.15] * n,
        # Jordartsandelar (summa 1)
        "andel_leriga":     [0.40] * n,
        "andel_medelfina":  [0.40] * n,
        "andel_grova":      [0.20] * n,
    }
    cols = ["polygon_id", "area_ha"] + LAND_COLS + SOIL_COLS
    return pd.DataFrame(data, columns=cols)

def clamp01(x):
    """Klipp värden till [0,1] och hantera icke-numeriska som 0.0."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    if np.isnan(x):
        return 0.0
    return min(max(x, 0.0), 1.0)


def validate_and_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Klipper andelar till [0,1], säkrar area_ha >= 0,
    och returnerar ev. varningar om summor inte är exakt 1.
    (Ingen normalisering görs här – modellen kan normalisera om användaren valt det.)
    """
    df2 = df.copy()

    # Säkerställ kolumner och klipp andelar
    for c in LAND_COLS + SOIL_COLS:
        if c not in df2.columns:
            df2[c] = 0.0
        df2[c] = df2[c].apply(clamp01)

    # Area
    if "area_ha" not in df2.columns:
        df2["area_ha"] = 0.0
    df2["area_ha"] = pd.to_numeric(df2["area_ha"], errors="coerce").fillna(0.0)

    # Varning: negativa area → sätt till 0
    if (df2["area_ha"] < 0).any():
        df2.loc[df2["area_ha"] < 0, "area_ha"] = 0.0

    warnings = []

    # Summeringar (varning om ej exakt 1.0)
    land_sum = df2[LAND_COLS].sum(axis=1)
    soil_sum = df2[SOIL_COLS].sum(axis=1)

    bad_land = ~np.isclose(land_sum, 1.0)
    bad_soil = ~np.isclose(soil_sum, 1.0)

    if bad_land.any():
        ids = df2.loc[bad_land, "polygon_id"].astype(str).tolist()
        warnings.append(f"Markandelar summerar inte till 1 för: {', '.join(ids)}.")

    if bad_soil.any():
        ids = df2.loc[bad_soil, "polygon_id"].astype(str).tolist()
        warnings.append(f"Jordartsandelar summerar inte till 1 för: {', '.join(ids)}.")

    # Area 0 (ger Tot P (kg/år) = 0 – OK men informativt)
    zero_area = df2["area_ha"] <= 0
    if zero_area.any():
        ids = df2.loc[zero_area, "polygon_id"].astype(str).tolist()
        warnings.append(f"Area (ha) är 0 för: {', '.join(ids)} (total belastning blir 0).")

    return df2, warnings

