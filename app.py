
# app.py
import numpy as np
import streamlit as st

# Importera din modellfunktion
from model import run_model

st.set_page_config(
    page_title="Fosformodell – P-belastning",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 Fosformodell (log-transformerad RandomForest)")
st.caption("Beräknar P-belastning baserat på area, mark- och jordartsandelar.")

with st.expander("ℹ️ Om modellen", expanded=False):
    st.markdown(
        """
        Den här appen använder funktionen `run_model(...)` från `model.py` och kräver
        att filen **`logrf_mild_model.pkl`** ligger i samma katalog.

        **Utdata**:
        - `p_kg_per_ha` – beräknad fosforbelastning per hektar (kg/ha·år)
        - `total_kg_per_ar` – total belastning per år (kg/år)
        """
    )

# --------------- Inmatning ---------------

st.header("1) Inmatning")

col_area, col_norm = st.columns([2, 1])
with col_area:
    area_ha = st.number_input(
        "Area (ha)",
        min_value=0.0,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Total area i hektar (> 0).",
    )
with col_norm:
    auto_normalize = st.toggle(
        "Auto-normalisera andelar",
        value=True,
        help="Om på, normaliseras andelarna automatiskt till att summera till 1.",
    )

st.subheader("Markanvändning – andelar")
col1, col2 = st.columns(2)
with col1:
    andel_akermark = st.number_input("Andel åkermark", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    andel_skogsmark = st.number_input("Andel skogsmark", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
with col2:
    andel_exploaterad = st.number_input("Andel exploaterad", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    andel_ovrig = st.number_input("Andel övrig", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

if auto_normalize:
    land_raw = np.array([andel_akermark, andel_exploaterad, andel_skogsmark, andel_ovrig], dtype=float)
    land_sum = land_raw.sum()
    if land_sum > 0:
        land_norm = land_raw / land_sum
        st.caption(
            f"Normaliserade markandelar (summa {land_sum:.3f} → 1.000): "
            f"Åker={land_norm[0]:.3f}, Exploaterad={land_norm[1]:.3f}, Skog={land_norm[2]:.3f}, Övrig={land_norm[3]:.3f}"
        )
    else:
        st.warning("Summan av markandelar är 0. Öka minst en av andelarna.", icon="⚠️")

st.subheader("Jordarter – andelar")
col3, col4, col5 = st.columns(3)
with col3:
    andel_leriga = st.number_input("Andel leriga", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
with col4:
    andel_medelfina = st.number_input("Andel medelfina", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
with col5:
    andel_grova = st.number_input("Andel grova", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

if auto_normalize:
    soil_raw = np.array([andel_leriga, andel_medelfina, andel_grova], dtype=float)
    soil_sum = soil_raw.sum()
    if soil_sum > 0:
        soil_norm = soil_raw / soil_sum
        st.caption(
            f"Normaliserade jordartsandelar (summa {soil_sum:.3f} → 1.000): "
            f"Leriga={soil_norm[0]:.3f}, Medelfina={soil_norm[1]:.3f}, Grova={soil_norm[2]:.3f}"
        )
    else:
        st.warning("Summan av jordartsandelar är 0. Öka minst en av andelarna.", icon="⚠️")

# --------------- Kör modell ---------------

st.header("2) Beräkna")
run = st.button("Kör modellen", type="primary")

if run:
    try:
        result = run_model(
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

        p_kg_per_ha = result["p_kg_per_ha"]
        total_kg_per_ar = result["total_kg_per_ar"]

        st.success("Beräkning klar ✅")
        st.metric(label="P-belastning (kg/ha·år)", value=f"{p_kg_per_ha:,.3f}")
        st.metric(label="Total belastning (kg/år)", value=f"{total_kg_per_ar:,.1f}")

        with st.expander("Visa råutdata", expanded=False):
            st.json(result)

    except FileNotFoundError as e:
        st.error(
            "Kunde inte hitta modellen (`logrf_mild_model.pkl`).\n\n"
            "Lägg filen i samma katalog som `app.py` och `model.py`.",
            icon="❌",
        )
        st.exception(e)
    except ValueError as e:
        st.error("Ogiltiga indata. Kontrollera area och andelar.", icon="⚠️")
        st.exception(e)
    except Exception as e:
        st.error("Ett oväntat fel inträffade vid körning av modellen.", icon="💥")
        st.exception(e)

st.divider()
st.caption("© Fosformodell – log-transformerad RandomForest med mild viktning.")
