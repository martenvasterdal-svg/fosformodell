
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Importera DataFrame-varianten från din modell
# (wrapper som tar hela tabellen och returnerar resultatkolumnerna)
from model import run_model_df

# ----------------------- Sidinställningar -----------------------
st.set_page_config(page_title="Fosforbelastning – andelar & area", layout="wide")
st.title("🧮 Fosforbelastning per polygon – andelar & area (ha)")
st.caption(
    "Ange area (ha) och andelar för markanvändning och jordarter per polygon. "
    "Koefficienterna styrs av din modellfil (logrf_mild_model.pkl via model.py). "
    "Beräkna fosforbelastning i kg/ha/år och kg/år."
)

# ----------------------- Konfiguration -------------------------
LAND_COLS = ["andel_akermark", "andel_exploaterad", "andel_skogsmark", "andel_ovrig"]
SOIL_COLS = ["andel_leriga", "andel_medelfina", "andel_grova"]

# ----------------------- Hjälpfunktioner ------------------------
def make_empty_table(n: int) -> pd.DataFrame:
    """Skapar en startmall med polygon_id, area_ha och standardandelar."""
    df = pd.DataFrame({
        "polygon_id": [f"P{i+1}" for i in range(n)],
        "area_ha": [10.0] * n,  # default 10 ha – kan ändras i tabellen
        # Markandelar (summa 1)
