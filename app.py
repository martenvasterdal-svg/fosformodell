
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ----------------------- Sidinställningar -----------------------
st.set_page_config(page_title="Fosforbelastning – andelar & area", layout="wide")
st.title("🧮 Fosforbelastning per polygon – andelar & area (ha)")
st.caption("Ange area (ha) och andelar för markanvändning och jordarter per polygon. Koefﬁcienterna styrs av modell.py. Beräkna fosforbelastning i kg/ha/år och kg/år.")

# ----------------------- Konfiguration -------------------------
LAND_COLS = ["andel_akermark", "andel_exploaterad", "andel_skogsmark", "andel_ovrig"]
SOIL_COLS = ["andel_leriga", "andel_medelfina", "andel_grova"]

# Importera modellen (koefficienter och beräkningar styrs här)
from model import run_model  # <- viktiga ändringen: inga koeff-inmatningar i UI

# ----------------------- Hjälpfunktioner ------------------------
def make_empty_table(n: int):
    """Skapar en startmall med polygon_id, area_ha och standardandelar."""
    df = pd.DataFrame({
        "polygon_id": [f"P{i+1}" for i in range(n)],
        "area_ha": [10.0]*n,  # default 10 ha – kan ändras i tabellen
        # Markandelar (summa 1)
        "andel_akermark":   [0.25]*n,
        "andel_exploaterad":[0.10]*n,
        "andel_skogsmark":  [0.50]*n,
        "andel_ovrig":      [0.15]*n,
        # Jordartsandelar (summa 1)
        "andel_leriga":     [0.40]*n,
        "andel_medelfina":  [0.40]*n,
        "andel_grova":      [0.20]*n,
    })
    return df

def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if np.isnan(x):
        return 0.0
    return min(max(x, 0.0), 1.0)

def validate_and_normalize_groups(df: pd.DataFrame, auto_normalize: bool):
    """
    Validerar: andelar i [0,1], summor per grupp (mark/jordarter) ≈ 1.
    Om auto_normalize=True: normalisera grupper (där summan > 0).
    Returnerar (df_fix, warnings).
    """
    df = df.copy()

    # Säkerställ och klipp andelar
    for c in LAND_COLS + SOIL_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].apply(clamp01)

    warnings = []

    # Validera area
    if "area_ha" not in df.columns:
        df["area_ha"] = 0.0
    df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce").fillna(0.0)
    if (df["area_ha"] < 0).any():
        neg_ids = df.loc[df["area_ha"] < 0, "polygon_id"].astype(str).tolist()
        warnings.append(f"Area (ha) är negativ för polygoner: {', '.join(neg_ids)}. Värden < 0 sätts till 0.")
        df.loc[df["area_ha"] < 0, "area_ha"] = 0.0

    # Markgrupp
    land_sum = df[LAND_COLS].sum(axis=1)
    mask_zero_land = np.isclose(land_sum, 0.0)
    mask_ne1_land = ~np.isclose(land_sum, 1.0)
    if auto_normalize:
        idx = (~mask_zero_land) & mask_ne1_land
        df.loc[idx, LAND_COLS] = df.loc[idx, LAND_COLS].div(land_sum[idx], axis=0)
    else:
        bad = (~mask_zero_land) & mask_ne1_land
        if bad.any():
            bad_ids = df.loc[bad, "polygon_id"].astype(str).tolist()
            warnings.append(f"Markandelar summerar inte till 1 för polygoner: {', '.join(bad_ids)}.")

    # Jordartsgrupp
    soil_sum = df[SOIL_COLS].sum(axis=1)
    mask_zero_soil = np.isclose(soil_sum, 0.0)
    mask_ne1_soil = ~np.isclose(soil_sum, 1.0)
    if auto_normalize:
        idx = (~mask_zero_soil) & mask_ne1_soil
        df.loc[idx, SOIL_COLS] = df.loc[idx, SOIL_COLS].div(soil_sum[idx], axis=0)
    else:
        bad = (~mask_zero_soil) & mask_ne1_soil
        if bad.any():
            bad_ids = df.loc[bad, "polygon_id"].astype(str).tolist()
            warnings.append(f"Jordartsandelar summerar inte till 1 för polygoner: {', '.join(bad_ids)}.")

    # Noll-summor
    if mask_zero_land.any():
        zero_ids = df.loc[mask_zero_land, "polygon_id"].astype(str).tolist()
        warnings.append(f"Markandelar är alla noll för polygoner: {', '.join(zero_ids)} (tolkas som 0 för alla kategorier).")
    if mask_zero_soil.any():
        zero_ids = df.loc[mask_zero_soil, "polygon_id"].astype(str).tolist()
        warnings.append(f"Jordartsandelar är alla noll för polygoner: {', '.join(zero_ids)} (tolkas som 0 för alla kategorier).")

    return df, warnings

def append_rows(df: pd.DataFrame, n_new: int) -> pd.DataFrame:
    """Lägg till n_new nya rader med defaultvärden och unika polygon_id."""
    df = df.copy()
    start_idx = len(df)
    new_df = make_empty_table(n_new)
    # Gör unika ID som fortsätter befintlig numrering
    new_df["polygon_id"] = [f"P{start_idx + i + 1}" for i in range(n_new)]
    return pd.concat([df, new_df], ignore_index=True)

# ----------------------- Sidebar (bara validering) -------------------------------
with st.sidebar:
    st.header("🧰 Inställningar")
    st.write("Koefficienter styrs av **modell.py** och kan inte ändras här.")
    auto_norm = st.checkbox("Normalisera andelar automatiskt till 1 per grupp", value=True)

# ----------------------- Dataingång -----------------------------
st.subheader("1) Ladda upp tabell eller starta från mall")
uploaded = st.file_uploader(
    "CSV eller Excel (XLSX) med kolumner: polygon_id, area_ha + andelar för mark och jordarter",
    type=["csv", "xlsx"]
)

# Initiera session-state för arbets-DF
if "work_df" not in st.session_state:
    st.session_state["work_df"] = make_empty_table(5)  # default 5 rader

# Skapa ny mall med valfritt antal rader
with st.expander("Starta från mall (valfritt)"):
    antal_mall = st.number_input("Antal områden i ny mall", min_value=1, max_value=5000, value=5, step=1)
    if st.button("🧩 Skapa ny mall"):
        st.session_state["work_df"] = make_empty_table(int(antal_mall))
        st.success(f"Skapade ny mall med {antal_mall} områden.")

# Läs uppladdad fil (om finns)
if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        df_in = pd.read_csv(uploaded)
    else:
        df_in = pd.read_excel(uploaded, engine="openpyxl")

    # Säkerställ kolumner
    if "polygon_id" not in df_in.columns:
        df_in.insert(0, "polygon_id", [f"P{i+1}" for i in range(len(df_in))])
    if "area_ha" not in df_in.columns:
        df_in.insert(1, "area_ha", 10.0)
    for c in LAND_COLS + SOIL_COLS:
        if c not in df_in.columns:
            df_in[c] = 0.0

    st.session_state["work_df"] = df_in
    st.success("Fil inläst och inlagd som arbetsdata.")

# ----------------------- Lägg till rader manuellt ----------------------
st.subheader("2) Ange area (ha) och andelar per polygon")
st.caption("Du kan lägga till valfritt antal områden manuellt samt redigera tabellen nedan.")

col_add1, col_add2 = st.columns([1, 1])
with col_add1:
    antal_nya = st.number_input("Antal nya områden att lägga till", min_value=1, max_value=5000, value=1, step=1)
with col_add2:
    if st.button("➕ Lägg till rader"):
        st.session_state["work_df"] = append_rows(st.session_state["work_df"], int(antal_nya))
        st.success(f"La till {antal_nya} nya områden.")

# Redigerbar tabell
column_config = {
    "polygon_id": st.column_config.TextColumn("Polygon-ID"),
    "area_ha": st.column_config.NumberColumn("Area (ha)", min_value=0.0, step=0.1),
}
for c in LAND_COLS + SOIL_COLS:
    column_config[c] = st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0, step=0.01)

edited = st.data_editor(
    st.session_state["work_df"],
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config=column_config,
    key="andels_area_editor",
)
# Uppdatera state med redigerad tabell
st.session_state["work_df"] = edited

# Validera/normalisera
edited_norm, warns = validate_and_normalize_groups(st.session_state["work_df"], auto_normalize=auto_norm)
if warns:
    for w in warns:
        st.warning(w)

st.divider()

# ----------------------- Kör modellen ---------------------------
st.subheader("3) Kör beräkning (koefficienter från modell.py)")
if st.button("🧪 Beräkna fosforbelastning", type="primary"):
    with st.spinner("Beräknar..."):
        # Viktigt: modellen styr koefficienter — vi skickar endast datatabellen
        out = run_model(df=edited_norm)

    st.success("Klar!")
    st.subheader("Resultat per polygon")
    st.dataframe(out, use_container_width=True)

    # Summering
    total_area = out["area_ha"].sum()
    total_p_kgyr = out["Tot P (kg/år)"].sum()
    mean_p_kghayr = out["Tot P bel. (kg/ha och år)"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total area (ha)", f"{total_area:,.2f}")
    c2.metric("Total fosfor (kg/år)", f"{total_p_kgyr:,.2f}")
    c3.metric("Medel specifik belastning (kg/ha/år)", f"{mean_p_kghayr:,.2f}")

    # Diagram – kg/år per polygon (Altair)
    chart_tot = alt.Chart(out).mark_bar().encode(
        x=alt.X("polygon_id:N", title="Polygon-ID", sort=None),
        y=alt.Y("Tot P (kg/år):Q", title="kg/år"),
        tooltip=["polygon_id", "Tot P (kg/år)", "Tot P bel. (kg/ha och år)", "area_ha"]
    ).properties(title="Fosforbelastning (kg/år) per polygon")
    st.altair_chart(chart_tot, use_container_width=True)

    # Diagram – kg/ha/år per polygon (Altair)
    chart_spec = alt.Chart(out).mark_bar(color="#3b82f6").encode(
        x=alt.X("polygon_id:N", title="Polygon-ID", sort=None),
        y=alt.Y("Tot P bel. (kg/ha och år):Q", title="kg/ha/år"),
        tooltip=["polygon_id", "Tot P bel. (kg/ha och år)", "area_ha"]
    ).properties(title="Specifik fosforbelastning (kg/ha/år) per polygon")
    st.altair_chart(chart_spec, use_container_width=True)

    # Export
    st.download_button(
        "⤓ Ladda ned resultat (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="fosfor_resultat.csv",
        mime="text/csv"
    )
else:
