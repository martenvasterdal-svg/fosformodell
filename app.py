import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Tunga GIS-beroenden importeras efter att vi säkrat att environment är ok
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats


st.set_page_config(page_title="Fosfor-Features: lera & markslag", layout="wide")

# ----------------------------
# Datafiler (laddas ner vid behov)
# ----------------------------
MARKTACKE_URL = "https://github.com/martenvasterdal-svg/fosformodell/releases/download/data-v1/marktacke.tif"
LERA_URL = "https://github.com/martenvasterdal-svg/fosformodell/releases/download/data-v1/finkorniga.jordarter.tif"

MARKTACKE_PATH = Path("marktacke.tif")
LERA_PATH = Path("finkorniga.jordarter.tif")

# Klassning enligt din definition
LANDCOVER_MAP = {
    1: "ovrig_mark",
    2: "akermark",
    3: "ovrig_mark",
    4: "exploaterad_mark",
    5: "vatten",
    6: "skog",
}
CLAY_CODE = 2  # 2 = lerig jord


def ensure_file(path, url: str):
    """Ladda ner fil om den saknas. Robust mot str/Path."""
    path = Path(path)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    import requests
    st.info(f"Laddar ner {path.name} (första körningen)...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


@st.cache_resource
def open_raster(path):
    """Öppna raster (cachas). Robust mot str/Path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Hittar inte rasterfilen: {path.resolve()}")
    return rasterio.open(path)


def read_uploaded_vector(uploaded_file) -> gpd.GeoDataFrame:
    """Läs GeoJSON, GPKG eller ZIP-shapefile."""
    name = uploaded_file.name.lower()

    if name.endswith(".geojson") or name.endswith(".json"):
        return gpd.read_file(uploaded_file)

    if name.endswith(".gpkg"):
        data = uploaded_file.getvalue()
        tmp = Path(st.session_state.get("tmp_gpkg_path", "tmp_upload.gpkg"))
        tmp.write_bytes(data)
        st.session_state["tmp_gpkg_path"] = str(tmp)
        return gpd.read_file(tmp)

    if name.endswith(".zip"):
        zdata = uploaded_file.getvalue()
        z = zipfile.ZipFile(io.BytesIO(zdata))

        extract_dir = Path(st.session_state.get("tmp_shp_dir", "tmp_shp"))
        extract_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(extract_dir)
        st.session_state["tmp_shp_dir"] = str(extract_dir)

        shp_files = list(extract_dir.glob("*.shp")) or list(extract_dir.rglob("*.shp"))
        if not shp_files:
            raise ValueError("Kunde inte hitta någon .shp i ZIP-filen.")
        return gpd.read_file(shp_files[0])

    raise ValueError("Stöds ej. Ladda upp GeoJSON, GPKG eller ZIP (shapefile).")


def ensure_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("Vektorfiler saknar CRS. Sätt CRS innan uppladdning (t.ex. EPSG:3006).")


def reproject_to_match(gdf: gpd.GeoDataFrame, raster) -> gpd.GeoDataFrame:
    if gdf.crs != raster.crs:
        return gdf.to_crs(raster.crs)
    return gdf


def zonal_counts_categorical(gdf: gpd.GeoDataFrame, raster, all_touched: bool) -> pd.DataFrame:
    """Returnerar DataFrame med pixelräkningar per klasskod + __total__."""
    stats = zonal_stats(
        gdf,
        raster.read(1),
        affine=raster.transform,
        nodata=raster.nodata,
        categorical=True,
        all_touched=all_touched,
        geojson_out=False,
    )
    df = pd.DataFrame(stats).fillna(0)
    df["__total__"] = df.sum(axis=1)
    return df


def compute_clay_share(gdf: gpd.GeoDataFrame, raster, id_series: pd.Series, all_touched: bool) -> pd.DataFrame:
    df = zonal_counts_categorical(gdf, raster, all_touched=all_touched)
    total = df["__total__"].replace(0, np.nan)

    if CLAY_CODE in df.columns:
        clay_px = df[CLAY_CODE]
    else:
        clay_px = 0.0

    return pd.DataFrame(
        {
            "id": id_series.to_numpy(),
            "andel_lerjord": (clay_px / total).to_numpy(),
        }
    )


def compute_landcover_shares(gdf: gpd.GeoDataFrame, raster, id_series: pd.Series, all_touched: bool) -> pd.DataFrame:
    df = zonal_counts_categorical(gdf, raster, all_touched=all_touched)

    total = df["__total__"].replace(0, np.nan)
    df = df.drop(columns=["__total__"])
    prop = df.div(total, axis=0)

    out = pd.DataFrame({"id": id_series.to_numpy()})

    # Summera ihop koder som mappar till samma namn (1 och 3 -> ovrig_mark)
    for code, name in LANDCOVER_MAP.items():
        colname = f"andel_{name}"
        if colname not in out.columns:
            out[colname] = 0.0
        if code in prop.columns:
            out[colname] = out[colname] + prop[code]

    # Säkerställ att alla förväntade kolumner finns
    for name in set(LANDCOVER_MAP.values()):
        cname = f"andel_{name}"
        if cname not in out.columns:
            out[cname] = 0.0

    return out


# ----------------------------
# UI
# ----------------------------
st.title("Features per avrinningsområde: andel lerjord + markslag")

st.write(
    "Appen beräknar per avrinningsområde:\n"
    "- **andel_lerjord** (jordartsklass 2)\n"
    "- andelar av markslag (1/3 övrig, 2 åker, 4 exploaterad, 5 vatten, 6 skog)\n"
)

with st.sidebar:
    st.header("Inställningar")
    id_field = st.text_input("ID-fält i vektordata", value="id")
    all_touched = st.checkbox("All_touched (räkna pixlar som berör polygon)", value=True)

# Se till att raster finns lokalt (om inte, ladda ner)
try:
    ensure_file(MARKTACKE_PATH, MARKTACKE_URL)
    ensure_file(LERA_PATH, LERA_URL)
except Exception as e:
    st.error(f"Kunde inte ladda ner rasterfiler: {e}")
    st.stop()

# Öppna raster (cachat)
try:
    markt_r = open_raster(MARKTACKE_PATH)
    lera_r = open_raster(LERA_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader(
    "Ladda upp avrinningsområden (GeoJSON, GPKG eller ZIP-shapefile)",
    type=["geojson", "json", "gpkg", "zip"],
)

if not uploaded:
    st.stop()

try:
    gdf = read_uploaded_vector(uploaded)
    ensure_crs(gdf)
except Exception as e:
    st.error(f"Kunde inte läsa vektordata: {e}")
    st.stop()

# Grundstädning av geometrier
gdf = gdf[gdf.geometry.notna()].copy()
gdf = gdf[gdf.is_valid].copy()
if gdf.empty:
    st.error("Inga giltiga geometrier efter filtrering.")
    st.stop()

if id_field not in gdf.columns:
    st.warning(f"ID-fältet '{id_field}' hittades inte. Skapar ett löpnummer som id.")
    gdf["id"] = np.arange(1, len(gdf) + 1)
    id_field = "id"

# Reprojicera mot respektive raster-CRS
gdf_m = reproject_to_match(gdf, markt_r)
gdf_l = reproject_to_match(gdf, lera_r)

with st.spinner("Beräknar andelar per avrinningsområde..."):
    clay_df = compute_clay_share(gdf_l, lera_r, gdf_l[id_field], all_touched=all_touched)
    lc_df = compute_landcover_shares(gdf_m, markt_r, gdf_m[id_field], all_touched=all_touched)
    out = clay_df.merge(lc_df, on="id", how="left")

st.subheader("Resultattabell")
st.caption("Andelar är mellan 0 och 1.")
st.dataframe(out, use_container_width=True)

csv = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Ladda ner som CSV",
    data=csv,
    file_name="andelar_lera_markslag.csv",
    mime="text/csv",
)

st.subheader("Snabb kontroll")
mark_cols = [c for c in out.columns if c.startswith("andel_") and c != "andel_lerjord"]
if mark_cols:
    sums = out[mark_cols].sum(axis=1)
    st.write("Summan av markslagsandelar (bör vara nära 1 om polygonerna täcks av raster och lite nodata):")
    st.dataframe(pd.DataFrame({"id": out["id"], "sum_markslag": sums}), use_container_width=True)
