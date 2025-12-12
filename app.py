import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats


st.set_page_config(page_title="Fosfor-Features: lera & markslag", layout="wide")

MARKTACKE_PATH = Path("marktacke.tif")
LERA_PATH = Path("finkorniga jordarter.tif")

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


@st.cache_resource
def open_raster(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Hittar inte rasterfilen: {path.resolve()}")
    return rasterio.open(path)


def read_uploaded_vector(uploaded_file) -> gpd.GeoDataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".geojson") or name.endswith(".json"):
        return gpd.read_file(uploaded_file)

    if name.endswith(".gpkg"):
        data = uploaded_file.getvalue()
        tmp = Path(st.session_state.get("tmp_gpkg", "tmp_upload.gpkg"))
        tmp.write_bytes(data)
        st.session_state["tmp_gpkg"] = str(tmp)
        return gpd.read_file(tmp)

    if name.endswith(".zip"):
        zdata = uploaded_file.getvalue()
        z = zipfile.ZipFile(io.BytesIO(zdata))
        extract_dir = Path(st.session_state.get("tmp_shp_dir", "tmp_shp"))
        extract_dir.mkdir(exist_ok=True)
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


def zonal_counts_categorical(gdf: gpd.GeoDataFrame, raster, all_touched: bool):
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


def compute_clay_share(gdf: gpd.GeoDataFrame, raster, id_series: pd.Series, all_touched: bool):
    df = zonal_counts_categorical(gdf, raster, all_touched=all_touched)
    total = df["__total__"].replace(0, np.nan)
    clay_px = df[CLAY_CODE] if CLAY_CODE in df.columns else 0
    return pd.DataFrame({
        "id": id_series.to_numpy(),
        "andel_lerjord": (clay_px / total).to_numpy()
    })


def compute_landcover_shares(gdf: gpd.GeoDataFrame, raster, id_series: pd.Series, all_touched: bool):
    df = zonal_counts_categorical(gdf, raster, all_touched=all_touched)
    total = df["__tota]()
