#!/usr/bin/env python3
"""Сводные графики сравнения результатов по нескольким датасетам."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load_summary() -> pd.DataFrame:
    path = RESULTS / "experiment_results.json"
    with open(RESULTS / "experiment_results.json", encoding="utf-8") as f:
        payload = json.load(f)

    if "datasets" in payload:
        rows = []
        for key, res in payload["datasets"].items():
            s = res["summary"]
            rows.append({
                "dataset": key,
                "title": res["dataset"]["title"],
                "rmse_improvement_pct": s["rmse_improvement_pct"],
                "speedup_pct": s["speedup_pct"],
                "ctr_reduction_pct": s["ctr_reduction_pct"],
                "baseline_rmse": res["baseline"]["rmse"],
                "best_rmse": res["optimizer_best"]["rmse"],
                "baseline_inference_ms": res["baseline"]["inference_ms"],
                "best_inference_ms": res["optimizer_best"]["inference_ms"],
                "baseline_ctr": res["baseline"]["ctr_tables"],
                "best_ctr": res["optimizer_best"]["ctr_tables"],
            })
        return pd.DataFrame(rows)

    b, o = payload["baseline"], payload["optimizer_best"]
    return pd.DataFrame([{
        "dataset": "amazon",
        "title": "Cars",
        "rmse_improvement_pct": (b["rmse"] - o["rmse"]) / b["rmse"] * 100,
        "speedup_pct": (b["inference_ms"] - o["inference_ms"]) / b["inference_ms"] * 100,
        "ctr_reduction_pct": (
            (b["ctr_tables"] - o["ctr_tables"]) / max(b["ctr_tables"], 1) * 100
        ),
        "baseline_rmse": b["rmse"],
        "best_rmse": o["rmse"],
        "baseline_inference_ms": b["inference_ms"],
        "best_inference_ms": o["inference_ms"],
        "baseline_ctr": b["ctr_tables"],
        "best_ctr": o["ctr_tables"],
    }])


def plot_quality_and_latency(df: pd.DataFrame) -> None:
    """Рисунок 1: RMSE и latency — одна шкала %, сопоставимые метрики."""
    labels = [f"{r['title']}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_rmse = ax.bar(
        x - width / 2, df["rmse_improvement_pct"], width,
        label="улучшение качества, %", color="#4472C4",
    )
    speedup_display = df["speedup_pct"].copy()
    for idx, row in df.iterrows():
        if row["baseline_inference_ms"] < 10 or abs(row["speedup_pct"]) < 1.0:
            speedup_display.at[idx] = 0.0
    bars_lat = ax.bar(
        x + width / 2, speedup_display, width,
        label="ускорение инференса, % (*≈0, шум замеров)", color="#70AD47",
    )
    rmse_labels = []
    for v in df["rmse_improvement_pct"]:
        if v < 0:
            rmse_labels.append(f"−{abs(v):.1f}%")
        else:
            rmse_labels.append(f"{v:.1f}%")
    ax.bar_label(bars_rmse, labels=rmse_labels, padding=3, fontsize=9, fontweight="bold")
    lat_labels = []
    for _, row in df.iterrows():
        if row["baseline_inference_ms"] < 10 or abs(row["speedup_pct"]) < 1.0:
            lat_labels.append("≈0*")
        else:
            lat_labels.append(f"{row['speedup_pct']:.1f}%")
    ax.bar_label(bars_lat, labels=lat_labels, padding=3, fontsize=9, fontweight="bold")
    y_min = min(df["rmse_improvement_pct"].min(), 0) * 1.25
    y_max = max(df["rmse_improvement_pct"].max(), speedup_display.max(), 5) * 1.18
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Относительное изменение, %")
    ax.set_title("Эффект SLA-HWS: качество и latency (относительные метрики)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "multi_01_improvements.png", dpi=150)
    plt.close(fig)


def plot_ctr_reduction(df: pd.DataFrame) -> None:
    """Рисунок 2: сокращение CTR — отдельная шкала (другой порядок величины)."""
    labels = [f"{r['title']}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, df["ctr_reduction_pct"], color="#ED7D31", width=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Сокращение CTR-таблиц, %")
    ax.set_title("Эффект SLA-HWS: сокращение числа CTR-таблиц")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "multi_02_ctr_reduction.png", dpi=150)
    plt.close(fig)


def plot_relative_improvements(df: pd.DataFrame) -> None:
    """Рисунок 3: три метрики в % — отдельные subplot с единой шкалой 0–100."""
    metrics = [
        ("rmse_improvement_pct", "Улучшение RMSE, %"),
        ("speedup_pct", "Ускорение инференса, %"),
        ("ctr_reduction_pct", "Сокращение CTR, %"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(df))
    colors = ["#4472C4", "#70AD47", "#ED7D31"]

    for ax, (col, title), color in zip(axes, metrics, colors):
        bars = ax.bar(x, df[col], color=color, width=0.55)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["dataset"], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        ymax = max(df[col].max() * 1.15, 5)
        ax.set_ylim(0, ymax)

    fig.suptitle("Относительный эффект SLA-HWS по датасетам", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "multi_03_relative_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    if len(df) < 2:
        print("Comparison figures skipped: need >= 2 datasets")
        return
    plot_quality_and_latency(df)
    plot_ctr_reduction(df)
    plot_relative_improvements(df)
    print(f"Comparison figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
