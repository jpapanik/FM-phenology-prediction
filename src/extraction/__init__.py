"""Data curation, weather/GDD/AGDD, Sentinel-2 NDVI, and foundation model embedding extraction."""

from src.extraction.embeddings import (
    dequantize_olmo,
    extract_alphaearth_embeddings,
    extract_olmoearth_embeddings,
    extract_presto_embeddings,
    extract_tessera_embeddings,
    lonlat_to_utm_zone,
)
from src.extraction.pep725_weather import (
    CROP_TBASE_MAP,
    build_curated_modeling_table,
    combine_raw_agera5_weather,
    compute_daily_gdd_agdd_pivots,
    crop_lookup,
    curate_pep725_observations,
    download_agera5_temperature,
    ensure_utf8_cdsapirc,
)
from src.extraction.sentinel2_ndvi import (
    compute_daily_ndvi_pivots,
    extract_sentinel2_ndvi,
    merge_ndvi_into_modeling_csv,
)

__all__ = [
    "CROP_TBASE_MAP",
    "crop_lookup",
    "curate_pep725_observations",
    "download_agera5_temperature",
    "combine_raw_agera5_weather",
    "compute_daily_gdd_agdd_pivots",
    "build_curated_modeling_table",
    "ensure_utf8_cdsapirc",
    "extract_sentinel2_ndvi",
    "compute_daily_ndvi_pivots",
    "merge_ndvi_into_modeling_csv",
    "dequantize_olmo",
    "lonlat_to_utm_zone",
    "extract_presto_embeddings",
    "extract_tessera_embeddings",
    "extract_alphaearth_embeddings",
    "extract_olmoearth_embeddings",
]
