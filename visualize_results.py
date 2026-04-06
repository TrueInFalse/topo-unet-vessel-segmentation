#!/usr/bin/env python3
"""Mainline training log visualization tool.

This script focuses on CSV training-log curve plotting for Baseline-ROI and
Topo-ROI workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODE_CHOICES = ("auto", "baseline", "topo")


COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "epoch": ["epoch"],
    "primary_loss": ["train_loss", "train_dice_loss", "train_dice_loss_roi"],
    "dice_loss": ["train_dice_loss", "train_dice_loss_roi"],
    "topology_loss": ["train_loss_topo", "topo_loss_scaled", "topo_loss_raw"],
    "train_dice": ["train_dice"],
    "val_dice": ["val_dice"],
    "cl_break": ["val_cl_break", "cl_break"],
    "betti_error": ["val_delta_beta0", "delta_beta0"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2x3 training curves from a CSV log (Baseline-ROI / Topo-ROI)."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Input CSV log file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG file path. Default: results/<log_stem>.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure-level title.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODE_CHOICES,
        default="auto",
        help="Log mode: auto / baseline / topo. Default: auto.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output file already exists.",
    )
    return parser.parse_args()


def infer_mode(columns: Iterable[str]) -> str:
    cols = set(columns)
    topo_markers = {
        "train_loss_topo",
        "topo_loss_raw",
        "topo_loss_scaled",
        "ratio",
        "train_dice_loss_roi",
    }
    baseline_markers = {
        "train_dice_loss",
        "val_cl_break",
        "val_delta_beta0",
    }

    if cols & topo_markers:
        return "topo"
    if cols & baseline_markers:
        return "baseline"
    return "baseline"


def resolve_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for name in candidates:
        if name in colset:
            return name
    return None


def standardize_columns(
    df: pd.DataFrame, mode: str
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    source_map: Dict[str, Optional[str]] = {}
    warnings: List[str] = []

    for canonical, candidates in COLUMN_CANDIDATES.items():
        src = resolve_column(df.columns, candidates)
        source_map[canonical] = src

    if source_map["epoch"] is None:
        raise ValueError("Missing required epoch column. Expected one of: epoch")

    expected_by_mode = {
        "baseline": [
            "primary_loss",
            "dice_loss",
            "train_dice",
            "val_dice",
            "cl_break",
            "betti_error",
        ],
        "topo": [
            "primary_loss",
            "dice_loss",
            "topology_loss",
            "train_dice",
            "val_dice",
            "cl_break",
            "betti_error",
        ],
    }

    for key in expected_by_mode.get(mode, []):
        if source_map.get(key) is None:
            warnings.append(
                f"Missing expected field '{key}'. Candidates: {COLUMN_CANDIDATES[key]}"
            )

    std = pd.DataFrame()
    for canonical, src in source_map.items():
        if src is None:
            continue
        std[canonical] = pd.to_numeric(df[src], errors="coerce")

    std = std.dropna(subset=["epoch"]).copy()
    std["epoch"] = std["epoch"].astype(int)

    if std["epoch"].duplicated().any():
        dup_count = int(std["epoch"].duplicated().sum())
        warnings.append(
            f"Detected {dup_count} duplicated epoch rows. Keeping the last record per epoch."
        )
        std = std.drop_duplicates(subset=["epoch"], keep="last")

    std = std.sort_values("epoch").reset_index(drop=True)
    return std, source_map, warnings


def resolve_output_path(log_file: Path, output: Optional[Path]) -> Path:
    if output is not None:
        return output
    return Path("results") / f"{log_file.stem}.png"


def plot_series_or_na(
    ax: plt.Axes,
    epochs: pd.Series,
    series: Optional[pd.Series],
    title: str,
    ylabel: str,
    label: Optional[str] = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if series is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return

    valid = (~series.isna()) & (~epochs.isna())
    if valid.sum() == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(epochs[valid], series[valid], linewidth=1.8, label=label)
    if label:
        ax.legend()


def plot_dice_curve(
    ax: plt.Axes,
    epochs: pd.Series,
    train_dice: Optional[pd.Series],
    val_dice: Optional[pd.Series],
) -> None:
    ax.set_title("Dice Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.grid(True, alpha=0.3)

    plotted = 0
    if train_dice is not None:
        valid = (~train_dice.isna()) & (~epochs.isna())
        if valid.sum() > 0:
            ax.plot(epochs[valid], train_dice[valid], linewidth=1.8, label="Train Dice")
            plotted += 1

    if val_dice is not None:
        valid = (~val_dice.isna()) & (~epochs.isna())
        if valid.sum() > 0:
            ax.plot(epochs[valid], val_dice[valid], linewidth=1.8, label="Val Dice")
            plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend()


def plot_curves(
    std_df: pd.DataFrame,
    mode: str,
    output_path: Path,
    title: Optional[str],
) -> None:
    epochs = std_df["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_series_or_na(
        axes[0, 0],
        epochs,
        std_df.get("primary_loss"),
        "Total / Primary Loss",
        "Loss",
        label="Primary Loss",
    )
    plot_series_or_na(
        axes[0, 1],
        epochs,
        std_df.get("dice_loss"),
        "Dice Loss",
        "Loss",
        label="Dice Loss",
    )
    plot_series_or_na(
        axes[0, 2],
        epochs,
        std_df.get("topology_loss"),
        "Topology Loss",
        "Loss",
        label="Topology Loss" if mode == "topo" else None,
    )

    plot_dice_curve(
        axes[1, 0],
        epochs,
        std_df.get("train_dice"),
        std_df.get("val_dice"),
    )
    plot_series_or_na(
        axes[1, 1],
        epochs,
        std_df.get("cl_break"),
        "CL-Break",
        "Count",
        label="CL-Break",
    )
    plot_series_or_na(
        axes[1, 2],
        epochs,
        std_df.get("betti_error"),
        "Betti Error",
        "Error",
        label="Betti Error",
    )

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    log_file = args.log_file
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return 1

    output_path = resolve_output_path(log_file=log_file, output=args.output)
    if args.no_overwrite and output_path.exists():
        print(f"[ERROR] Output file already exists: {output_path}")
        print("[ERROR] Use a different --output or remove --no-overwrite.")
        return 1

    df = pd.read_csv(log_file)
    if df.empty:
        print(f"[ERROR] CSV is empty: {log_file}")
        return 1

    if args.mode == "auto":
        mode = infer_mode(df.columns)
        print(f"[INFO] Auto-detected mode: {mode}")
    else:
        mode = args.mode
        print(f"[INFO] Mode from CLI: {mode}")

    try:
        std_df, source_map, warnings = standardize_columns(df, mode)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("[INFO] Column mapping:")
    for key in [
        "epoch",
        "primary_loss",
        "dice_loss",
        "topology_loss",
        "train_dice",
        "val_dice",
        "cl_break",
        "betti_error",
    ]:
        print(f"  - {key}: {source_map.get(key)}")

    if warnings:
        print("[WARN] Missing or normalized fields:")
        for msg in warnings:
            print(f"  - {msg}")

    plot_curves(std_df=std_df, mode=mode, output_path=output_path, title=args.title)

    print(f"[OK] Saved figure: {output_path}")
    print(f"[OK] Rows plotted: {len(std_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
