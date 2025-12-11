
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------- Sidinställningar -----------------------
st.set_page_config(page_title="Fosforbelastning – andelar & area", layout="wide")
st.title("🧮 Fosforbelastning per polygon – andelar & area (ha)")
st.caption("Ange area (ha) och andelar för markanvändning och jordarter per polygon. Justera koefficienter och beräkna fosforbelastning i kg/ha/år och kg/år.")

# ----------------------- Konfiguration -------------------------
LAND_COLS = ["andel_akermark", "andel_exploaterad", "andel_skogsmark", "andel_ovrig"]
SOIL_COLS = ["andel_leriga", "andel_medelfina", "andel_grova"]
DEFAULT_ROWS = 5

# ----------------------- Hjälpfunktioner ------------------------
def make_empty_table(n=DEFAULT_ROWS):
    """Skapar en startmall med polygon_id, area_ha och standardandelar."""
    df = pd.DataFrame({
        "polygon_id": [f"P{i+1}" for i in range(n)],
        "area_ha": [10.0]*n,  # default 10 ha – går att ändra i tabellen
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
    Om auto_normalize=True normaliseras grupper (där summan > 0).
    Returnerar (df_fix, warnings)
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

def run_model(df: pd.DataFrame,
              koef_land: dict,
              koef_soil: dict,
              baslinje_p: float = 0.0) -> pd.DataFrame:
    """
    Exempelmodell:
      Tot P bel. (kg/ha/år) = baslinje
                              + sum(mark_andel * koef_land)
                              + sum(jord_andel * koef_soil)

    Tot P (kg/år) = Tot P bel. (kg/ha/år) * area_ha

    Byt gärna ut mot din faktiska statistiska modell.
    """
    df = df.copy()

    land_index = (
        df["andel_akermark"]    * koef_land.get("akermark", 0.0) +
        df["andel_exploaterad"] * koef_land.get("exploaterad", 0.0) +
        df["andel_skogsmark"]   * koef_land.get("skogsmark", 0.0) +
        df["andel_ovrig"]       * koef_land.get("ovrig", 0.0)
    )

    soil_index = (
        df["andel_leriga"]      * koef_soil.get("leriga", 0.0) +
        df["andel_medelfina"]   * koef_soil.get("medelfina", 0.0) +
        df["andel_grova"]       * koef_soil.get("grova", 0.0)
    )

    df["Tot P bel. (kg/ha och år)"] = baslinje_p + land_index + soil_index
    df["Tot P (kg/år)"] = df["Tot P bel. (kg/ha och år)"] * df["area_ha"]
    return df

# ----------------------- Sidofält -------------------------------
with st.sidebar:
    st.header("🧰 Koefficienter & validering")

    st.subheader("Koefficienter – mark (kg P/ha/år per andel)")
    akermark_k = st.number_input("Åkermark", value=0.8, step=0.1)
    explo_k    = st.number_input("Exploaterad mark", value=0.4, step=0.1)
    skog_k     = st.number_input("Skog", value=0.2, step=0.1)
    ovrig_k    = st.number_input("Övrig mark", value=0.1, step=0.1)

    st.subheader("Koefficienter – jordarter (modifier/addering per andel)")
    ler_k   = st.number_input("Leriga jordar", value=0.2, step=0.1)
    medel_k = st.number_input("Medelfina jordar", value=0.1, step=0.1)
    grov_k  = st.number_input("Grova jordar", value=0.0, step=0.1)

    baslinje_p = st.number_input("Baslinje (kg P/ha/år)", value=0.0, step=0.1)

    st.subheader("Validering")
    auto_norm = st.checkbox("Normalisera andelar automatiskt till 1 per grupp", value=True)

    koef_land = {"akermark": akermark_k, "exploaterad": explo_k, "skogsmark": skog_k, "ovrig": ovrig_k}
    koef_soil = {"leriga": ler_k, "medelfina": medel_k, "grova": grov_k}

# ----------------------- Dataingång -----------------------------
st.subheader("1) Ladda upp tabell eller starta från mall")
uploaded = st.file_uploader(
    "CSV eller Excel (XLSX) med kolumner: polygon_id, area_ha + andelar för mark och jordarter",
    type=["csv", "xlsx"]
)

if uploaded:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
    st.success("Fil inläst.")
else:
    df = make_empty_table()
    st.info("Ingen fil uppladdad – startar från en mall med 5 rader.")

# Se till att nödvändiga kolumner finns
if "polygon_id" not in df.columns:
    df.insert(0, "polygon_id", [f"P{i+1}" for i in range(len(df))])
if "area_ha" not in df.columns:
    df.insert(1, "area_ha", 10.0)

for c in LAND_COLS + SOIL_COLS:
    if c not in df.columns:
        df[c] = 0.0

# ----------------------- Redigerbar tabell ----------------------
st.subheader("2) Ange area (ha) och andelar per polygon")
st.caption("Area (ha) används för att beräkna total belastning (kg/år). Andelar ska summera till 1 inom Mark-gruppen respektive Jordarts-gruppen.")
column_config = {
    "polygon_id": st.column_config.TextColumn("Polygon-ID"),
    "area_ha": st.column_config.NumberColumn("Area (ha)", min_value=0.0, step=0.1),
}
for c in LAND_COLS + SOIL_COLS:
    column_config[c] = st.column_config.NumberColumn(c, min_value=0.0, max_value=1.0, step=0.01)

edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config=column_config,
    key="andels_area_editor",
)

# Validera/normalisera
edited_norm, warns = validate_and_normalize_groups(edited, auto_normalize=auto_norm)
if warns:
    for w in warns:
        st.warning(w)

st.divider()

# ----------------------- Kör modellen ---------------------------
st.subheader("3) Kör beräkning")
if st.button("🧪 Beräkna fosforbelastning", type="primary"):
    with st.spinner("Beräknar..."):
        out = run_model(
            df=edited_norm,
            koef_land=koef_land,
            koef_soil=koef_soil,
            baslinje_p=baslinje_p
        )

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

    # Diagram – kg/år per polygon
    fig_tot = px.bar(
        out,
        x="polygon_id",
        y="Tot P (kg/år)",
        title="Fosforbelastning (kg/år) per polygon",
        labels={"polygon_id": "Polygon-ID", "Tot P (kg/år)": "kg/år"},
        text="Tot P (kg/år)"
    )
    fig_tot.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_tot.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_tot, use_container_width=True)

    # Diagram – kg/ha/år per polygon
    fig_spec = px.bar(
        out,
        x="polygon_id",
        y="Tot P bel. (kg/ha och år)",
        title="Specifik fosforbelastning (kg/ha/år) per polygon",
        labels={"polygon_id": "Polygon-ID", "Tot P bel. (kg/ha och år)": "kg/ha/år"},
        text="Tot P bel. (kg/ha och år)"
    )
    fig_spec.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_spec.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_spec, use_container_width=True)

    # Export
    st.download_button(
        "⤓ Ladda ned resultat (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="fosfor_resultat.csv",
        mime="text/csv"
    )
else:
    st.info("Sätt area (ha) och andelar i tabellen och klicka **Beräkna fosforbelastning**.")
