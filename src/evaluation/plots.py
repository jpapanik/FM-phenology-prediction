from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.evaluation.metrics import compute_r2_decomposition


def plot_scatter_grid(
    model_entries: List[Tuple[str, str, np.ndarray]],
    actual_doy: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    colors = ["steelblue", "darkorange", "forestgreen", "purple", "brown", "crimson"]
    n_models = len(model_entries)
    fig, axes = plt.subplots(1, n_models, figsize=(4.4 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    min_val = float(np.min(actual_doy)) - 5.0
    max_val = float(np.max(actual_doy)) + 5.0

    for ax, (_, label, preds), color in zip(axes, model_entries, colors * 2):
        mae = mean_absolute_error(actual_doy, preds)
        rmse = np.sqrt(mean_squared_error(actual_doy, preds))
        ax.scatter(actual_doy, preds, alpha=0.35, color=color, s=22)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="1:1 Fit")
        ax.set_title(f"{label}\nMAE: {mae:.2f} d | RMSE: {rmse:.2f} d")
        ax.set_xlabel("Actual DOY")
        ax.set_ylabel("Predicted DOY")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_r2_decomposition_and_trajectory(
    df_decomp: pd.DataFrame,
    df_per_year: pd.DataFrame,
    title_prefix: str,
    out_dir: Path,
    file_suffix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_ticks = np.arange(-1.0, 1.1, 0.1)

    # 1. R² Decomposition Bar Chart
    x_pos = np.arange(len(df_decomp))
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_pos - bar_width, df_decomp["Global_Pooled_R2"], width=bar_width, label="Global Pooled R²", color="steelblue")
    ax.bar(x_pos, df_decomp["Spatial_Station_Mean_R2"], width=bar_width, label="Spatial Station-Mean R²", color="darkorange")
    ax.bar(x_pos + bar_width, df_decomp["Temporal_Anomaly_R2"], width=bar_width, label="Temporal Anomaly R²", color="forestgreen")

    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_decomp["Model_Label"], rotation=15, ha="right")
    ax.set_yticks(y_ticks)
    ax.set_ylabel("Coefficient of Determination (R²)")
    ax.set_title(f"{title_prefix} — R² Decomposition")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_dir / f"r2_decomposition_{file_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Per-Year Trajectory Line Plot
    colors = ["steelblue", "darkorange", "forestgreen", "purple", "brown", "crimson"]
    fig, ax = plt.subplots(figsize=(14, 6))
    years = df_per_year["Test_Year"].values

    for (_, row), color in zip(df_decomp.iterrows(), colors * 2):
        key = row["Model_Key"]
        ax.plot(years, df_per_year[f"{key}_R2"], marker="o", lw=2, color=color, label=row["Model_Label"])

    ax.axhline(0, color="red", linestyle="--", lw=1.2, alpha=0.7)
    ax.set_title(f"{title_prefix} — Temporal Transferability Across Unseen Years")
    ax.set_xlabel("Held-Out Test Year")
    ax.set_ylabel("Held-Out Year R²")
    ax.set_xticks(years)
    ax.set_yticks(y_ticks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(out_dir / f"r2_per_year_{file_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_c22_vs_dl_comparison(
    algo_preds: Dict[str, Dict[str, np.ndarray]],
    model_keys: List[str],
    actual_doy: np.ndarray,
    station_ids: np.ndarray,
    emb_upper: str,
    dataset_display_name: str,
    save_path: Path,
    n_baselines: int = 5,
) -> None:
    """Plot 2-panel (Random Forest & XGBoost) R² decomposition for catch22 vs. Double Logistic & FM fusion."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    x_pos = np.arange(len(model_keys))
    bar_width = 0.26

    for ax, algo_name in [(axes[0], "Random Forest"), (axes[1], "XGBoost")]:
        preds_dict = algo_preds.get(algo_name, {})
        g_list, s_list, t_list = [], [], []
        for k in model_keys:
            if k in preds_dict:
                g_r2, s_r2, t_r2 = compute_r2_decomposition(actual_doy, preds_dict[k], station_ids)
            else:
                g_r2, s_r2, t_r2 = np.nan, np.nan, np.nan
            g_list.append(g_r2)
            s_list.append(s_r2)
            t_list.append(t_r2)

        ax.bar(x_pos - bar_width, g_list, width=bar_width, label="Global Pooled R²", color="steelblue")
        ax.bar(x_pos, s_list, width=bar_width, label="Spatial Station-Mean R²", color="darkorange")
        ax.bar(x_pos + bar_width, t_list, width=bar_width, label="Temporal Anomaly R² (ΔDOY)", color="forestgreen")

        if len(model_keys) > n_baselines:
            ax.axvline(n_baselines - 0.5, color="gray", linestyle="--", lw=1.0)

        ax.axhline(0, color="black", lw=1.0)
        ax.set_ylabel("R² Score")
        ax.set_title(f"{algo_name}: Standalone Baselines (Left) vs. {emb_upper} Fusion (Right)")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", frameon=True)

    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(model_keys, rotation=20, ha="right")
    fig.suptitle(f"catch22 vs. Double Logistic Phenology Curves — {emb_upper}\n{dataset_display_name}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
