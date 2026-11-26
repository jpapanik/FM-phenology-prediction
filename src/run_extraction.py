"""CLI entrypoint for PEP725/AgERA5 curation, GDD/AGDD calculation, Sentinel-2 NDVI, and FM embeddings."""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.extraction.embeddings import (
    extract_alphaearth_embeddings,
    extract_olmoearth_embeddings,
    extract_presto_embeddings,
    extract_tessera_embeddings,
)
from src.extraction.pep725_weather import (
    build_curated_modeling_table,
    combine_raw_agera5_weather,
    curate_pep725_observations,
    download_agera5_temperature,
    ensure_utf8_cdsapirc,
)
from src.extraction.sentinel2_ndvi import extract_sentinel2_ndvi, merge_ndvi_into_modeling_csv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run data & embedding extraction stages.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["pep725", "agera5", "gdd_agdd", "s2_ndvi", "presto", "tessera", "alphaearth", "olmoearth"],
        help="Extraction stage to execute.",
    )
    parser.add_argument("--study-folder", type=Path, required=True, help="Target study output directory.")
    parser.add_argument("--pep-csv", type=Path, default=Path("pep725_data.csv"), help="Path to PEP725 CSV.")
    parser.add_argument("--country", type=str, default="Germany")
    parser.add_argument("--crop-code", type=str, default="BARW")
    parser.add_argument("--target-bbch", type=float, default=51.0)
    parser.add_argument("--start-date", type=str, default="2017-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--gee-project", type=str, default="thesis-phenology")
    parser.add_argument("--skip-cds-download", action="store_true")
    args = parser.parse_args(argv)

    args.study_folder.mkdir(parents=True, exist_ok=True)
    obs_csv = args.study_folder / "AgERA5_all_observations.csv"
    combined_weather_csv = args.study_folder / "AgERA5_observations_combined_with_weather_data.csv"
    modeling_csv = args.study_folder / "curated_for_modeling.csv"
    agera5_raw_dir = args.raw_data_dir / f"agERA5_{args.study_folder.name}_raw"

    if args.stage == "pep725":
        filtered_obs, grid_cells = curate_pep725_observations(
            pep_csv_path=args.pep_csv,
            country=args.country,
            crop_code=args.crop_code,
            target_bbch=args.target_bbch,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.raw_data_dir,
        )
        filtered_obs.to_csv(obs_csv, sep=";", index=False)
        print(f"Saved {len(filtered_obs)} curated PEP725 observations to {obs_csv} ({len(grid_cells)} AgERA5 grid cells).")

    elif args.stage in ("agera5", "gdd_agdd"):
        if obs_csv.exists():
            filtered_obs = pd.read_csv(obs_csv, sep=";")
            grid_cells = (
                filtered_obs[["AgERA5_Longitude", "AgERA5_Latitude"]]
                .drop_duplicates()
                .rename(columns={"AgERA5_Longitude": "Longitude", "AgERA5_Latitude": "Latitude"})
                .reset_index(drop=True)
            )
        else:
            filtered_obs, grid_cells = curate_pep725_observations(
                pep_csv_path=args.pep_csv,
                country=args.country,
                crop_code=args.crop_code,
                target_bbch=args.target_bbch,
                start_date=args.start_date,
                end_date=args.end_date,
                data_dir=args.raw_data_dir,
            )
            filtered_obs.to_csv(obs_csv, sep=";", index=False)

        if args.stage == "agera5" and not args.skip_cds_download:
            ensure_utf8_cdsapirc()
            download_agera5_temperature(
                grid_cells=grid_cells,
                start_date=args.start_date,
                end_date=args.end_date,
                download_dir=agera5_raw_dir,
            )

        weather_df = combine_raw_agera5_weather(
            raw_dir=agera5_raw_dir,
            combined_weather_file=combined_weather_csv,
        )
        df_final = build_curated_modeling_table(
            observations_df=filtered_obs,
            weather_df=weather_df,
            crop_code=args.crop_code,
            output_csv_path=modeling_csv,
        )
        print(f"Saved {len(df_final)} rows with GDD_doy_1..365 and AGDD_doy_1..365 to {modeling_csv}")

    elif args.stage == "s2_ndvi":
        if not modeling_csv.exists():
            raise FileNotFoundError(f"Expected {modeling_csv} before running s2_ndvi.")
        s2_raw_dir = args.raw_data_dir / f"s2_ndvi_raw_{args.study_folder.name}"
        extract_sentinel2_ndvi(df_observations=pd.read_csv(modeling_csv, sep=";"), s2_raw_dir=s2_raw_dir)
        merged = merge_ndvi_into_modeling_csv(modeling_csv_path=modeling_csv, s2_raw_dir=s2_raw_dir)
        print(f"Updated {modeling_csv} with 365-day NDVI features ({merged['NDVI_doy_180'].notna().sum()}/{len(merged)} non-null).")

    elif args.stage == "presto":
        out = extract_presto_embeddings(
            study_folder=args.study_folder,
            modeling_csv_path=modeling_csv if modeling_csv.exists() else None,
            gee_project=args.gee_project,
        )
        print(f"Saved PRESTO embeddings to {out}")

    elif args.stage == "tessera":
        out = extract_tessera_embeddings(study_folder=args.study_folder, modeling_csv_path=modeling_csv)
        print(f"Saved TESSERA embeddings to {out}")

    elif args.stage == "alphaearth":
        out = extract_alphaearth_embeddings(study_folder=args.study_folder, modeling_csv_path=modeling_csv)
        print(f"Saved AlphaEarth embeddings to {out}")

    elif args.stage == "olmoearth":
        out = extract_olmoearth_embeddings(study_folder=args.study_folder, modeling_csv_path=modeling_csv)
        print(f"Saved OlmoEarth embeddings to {out}")


if __name__ == "__main__":
    main()
