#!/usr/bin/env python3
"""Графики процесса Optuna: история подбора, штрафы, важность параметров."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    optuna = None
    get_param_importances = None

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def load_payload() -> dict:
    with open(RESULTS / "experiment_results.json", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_result(payload: dict, name: str | None = None) -> tuple[str, dict]:
    if "datasets" in payload:
        primary = name or payload.get("primary_dataset", "amazon")
        return primary, payload["datasets"][primary]
    return "amazon", payload


def trials_to_df(trials: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(trials)
    if "params" in df.columns:
        params_df = pd.json_normalize(df["params"])
        df = pd.concat([df.drop(columns=["params"]), params_df], axis=1)
    return df


def _best_so_far(values: pd.Series) -> pd.Series:
    return values.cummin()


def _prefix(dataset_key: str, base: str) -> str:
    if dataset_key == "amazon":
        return base
    return f"optuna_{dataset_key}_{base.split('_', 1)[1]}"


def _focused_limits(
    values: list[float],
    *,
    pad_frac: float = 0.10,
    min_span_frac: float = 0.06,
) -> tuple[float, float]:
    """Оси по данным: ~80% площади под точками, без привязки к нулю."""
    arr = np.asarray(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi) * 0.05, 1e-6)
    min_span = max(span, abs(hi) * min_span_frac)
    if span < min_span:
        mid = (lo + hi) / 2
        lo, hi = mid - min_span / 2, mid + min_span / 2
    margin = (hi - lo) * pad_frac
    return lo - margin, hi + margin


def plot_optimization_history(
    df: pd.DataFrame,
    dataset_key: str,
    *,
    max_ms: float,
    max_rmse: float,
) -> None:
    """История по RMSE/Logloss (не штрафная objective) — видна динамика в SLA-зоне."""
    from matplotlib.lines import Line2D

    qual_name = "Logloss" if dataset_key == "amazon" else "RMSE"
    fig, ax = plt.subplots(figsize=(10, 5))
    x = df["number"]
    rmse = df["rmse"]
    ax.scatter(x, rmse, c="#A5A5A5", s=40, alpha=0.8, label=f"{qual_name} trial")

    ax.plot(
        x, _best_so_far(rmse), color="#C00000", linewidth=2.5,
        label=f"лучший {qual_name} на данный момент",
    )

    quality_ok = df["rmse"] <= max_rmse
    latency_ok = df["inference_ms"] <= max_ms
    full_sla = df[quality_ok & latency_ok]
    qual_only = df[quality_ok & ~latency_ok]

    sla_label = "trial в SLA-зоне"
    if not full_sla.empty:
        ax.scatter(
            full_sla["number"], full_sla["rmse"], c="#70AD47", s=55, alpha=0.9,
            label=sla_label, zorder=3,
        )
    elif not qual_only.empty:
        sla_label = "RMSE в SLA-зоне (latency — cold-замер)"
        ax.scatter(
            qual_only["number"], qual_only["rmse"], c="#70AD47", s=55, alpha=0.9,
            label=sla_label, zorder=3,
        )
    else:
        ax.scatter([], [], c="#70AD47", s=55, alpha=0.9, label=f"{sla_label} (нет trials)")

    handles, labels = ax.get_legend_handles_labels()
    if not any("SLA" in lb for lb in labels):
        handles.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#70AD47",
                markersize=8, label=sla_label,
            )
        )
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.set_xlabel("Номер trial (итерация Optuna)")
    ax.set_ylabel(qual_name)
    ax.set_title(f"История оптимизации: {qual_name} ({dataset_key})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_01_history.png"), dpi=150)
    plt.close(fig)


def plot_metrics_over_trials(df: pd.DataFrame, baseline: dict, dataset_key: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = df["number"]
    ax1.scatter(x, df["rmse"], c="#4472C4", s=35, alpha=0.7, label="RMSE trial")
    ax1.axhline(baseline["rmse"], color="#4472C4", linestyle="--", alpha=0.6, label="baseline RMSE")
    ax1.set_xlabel("Номер trial")
    ax1.set_ylabel("RMSE", color="#4472C4")
    ax1.tick_params(axis="y", labelcolor="#4472C4")

    ax2 = ax1.twinx()
    ax2.scatter(x, df["inference_ms"], c="#70AD47", s=35, alpha=0.7, marker="s", label="latency trial")
    ax2.axhline(baseline.get("constraints_max_ms", 16), color="red", linestyle="--", alpha=0.6, label="SLA")
    ax2.set_ylabel("Инференс, мс", color="#70AD47")
    ax2.tick_params(axis="y", labelcolor="#70AD47")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"Метрики качества и latency ({dataset_key})")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_02_metrics_over_trials.png"), dpi=150)
    plt.close(fig)


def plot_feasible_region(df: pd.DataFrame, res: dict, dataset_key: str) -> None:
    """Допустимая область: t_inf ≤ t_max и качество ≤ baseline·(1+Δ_max)."""
    from matplotlib.patches import Rectangle

    c = res["constraints"]
    baseline = res["baseline"]
    baseline_rmse = baseline["rmse"]
    baseline_ms = baseline["inference_ms"]
    max_rmse = baseline_rmse * (1 + c["max_rmse_degradation_pct"] / 100)
    max_ms = c["max_inference_ms"]
    from report_best import in_sla_box, pick_sla_best, sla_limits

    optimizer = res["optimizer_best"]
    max_ms, max_rmse = sla_limits(res)
    if not in_sla_box(optimizer["rmse"], optimizer["inference_ms"], max_ms, max_rmse):
        optimizer = pick_sla_best(res) or optimizer

    qual_name = (
        res.get("dataset", {}).get("metric_labels", {}).get("primary", "RMSE")
    )
    qdec = 4 if res.get("dataset", {}).get("task_type") == "classification" else 2

    fig, ax = plt.subplots(figsize=(9, 6))

    x_vals = list(df["inference_ms"]) + [baseline_ms, optimizer["inference_ms"]]
    y_vals = list(df["rmse"]) + [baseline_rmse, optimizer["rmse"]]
    x_lo, x_hi = _focused_limits(x_vals)
    y_lo, y_hi = _focused_limits(y_vals)

    # SLA-зона — только в видимой области (не от нуля по X)
    sla_x0 = max(x_lo, 0)
    sla_x1 = min(x_hi, max_ms)
    if sla_x1 > sla_x0:
        ax.add_patch(
            Rectangle(
                (sla_x0, y_lo), sla_x1 - sla_x0, max_rmse - y_lo,
                facecolor="#70AD47", alpha=0.15, edgecolor="#2E7D32",
                linewidth=1.2, linestyle="--", zorder=0,
            )
        )
    ax.text(
        sla_x0 + (sla_x1 - sla_x0) * 0.05 if sla_x1 > sla_x0 else x_lo,
        max_rmse - (max_rmse - y_lo) * 0.08,
        f"SLA: ≤{max_ms:.1f} мс\n{qual_name} ≤{max_rmse:.{qdec}f}",
        fontsize=9, color="#2E7D32", va="top",
    )

    quality_ok = df["rmse"] <= max_rmse
    latency_ok = df["inference_ms"] <= max_ms
    in_sla = df[quality_ok & latency_ok]
    out_sla = df[~(quality_ok & latency_ok)]

    ax.scatter(
        out_sla["inference_ms"], out_sla["rmse"],
        c="#ED7D31", s=55, alpha=0.75, label="trial вне SLA-области",
    )
    if not in_sla.empty:
        ax.scatter(
            in_sla["inference_ms"], in_sla["rmse"],
            c="#70AD47", s=75, alpha=0.9, label="trial в SLA-области",
        )

    ax.scatter(
        baseline_ms, baseline_rmse, s=110, c="#4472C4", marker="D",
        zorder=5, label="baseline",
    )
    ax.scatter(
        optimizer["inference_ms"], optimizer["rmse"], s=320, c="#C00000",
        marker="*", zorder=6, label="optimizer_best",
    )

    ax.axvline(max_ms, color="#C00000", linestyle="--", alpha=0.45, linewidth=1)
    ax.axhline(max_rmse, color="#C00000", linestyle="--", alpha=0.45, linewidth=1)
    ax.axhline(baseline_rmse, color="#4472C4", linestyle=":", alpha=0.4, linewidth=1)

    if optimizer["inference_ms"] > max_ms or optimizer["rmse"] > max_rmse:
        ax.annotate(
            "выбран по компромиссу\n(вне joint-SLA)",
            (optimizer["inference_ms"], optimizer["rmse"]),
            xytext=(12, 12), textcoords="offset points", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#C00000", lw=0.8),
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Время инференса, мс")
    ax.set_ylabel(qual_name)
    ax.set_title(f"Поиск в пространстве «latency × {qual_name}» ({dataset_key})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_03_feasible_region.png"), dpi=150)
    plt.close(fig)


def plot_penalty_mechanism(df: pd.DataFrame, dataset_key: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = df["number"]
    ax.bar(x, df["raw_rmse"], color="#4472C4", alpha=0.7, label="RMSE (база objective)")
    ax.bar(x, df["latency_penalty"], bottom=df["raw_rmse"], color="#ED7D31", alpha=0.8, label="штраф за latency")
    bottom = df["raw_rmse"] + df["latency_penalty"]
    ax.bar(x, df["quality_penalty"], bottom=bottom, color="#C00000", alpha=0.8, label="штраф за качество")
    ax.set_xlabel("Номер trial")
    ax.set_ylabel("Objective (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Разложение objective ({dataset_key})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_04_penalty_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_objective_vs_rmse(df: pd.DataFrame, dataset_key: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(df["feasible"], "#70AD47", "#ED7D31")
    ax.scatter(df["rmse"], df["value"], c=colors, s=60, alpha=0.8)
    ax.plot([df["rmse"].min(), df["rmse"].max()], [df["rmse"].min(), df["rmse"].max()],
            "k--", alpha=0.4, label="objective = RMSE")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.set_title(f"Objective vs RMSE ({dataset_key})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_05_objective_vs_rmse.png"), dpi=150)
    plt.close(fig)


def plot_param_effects(df: pd.DataFrame, dataset_key: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    params = [("iterations", "iterations"), ("depth", "depth"), ("max_ctr_complexity", "max_ctr_complexity")]
    for ax, (col, title) in zip(axes, params):
        sc = ax.scatter(df[col], df["rmse"], c=df["inference_ms"],
                        cmap="viridis_r", s=70, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(title)
        ax.set_ylabel("RMSE")
        ax.set_title(f"RMSE vs {title}")
        ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes, label="инференс, мс", shrink=0.8)
    fig.suptitle(f"Влияние параметров ({dataset_key})", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_06_param_effects.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ctr_preset_boxplot(df: pd.DataFrame, dataset_key: str) -> None:
    if "ctr_preset" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column="rmse", by="ctr_preset", ax=axes[0])
    axes[0].set_title("RMSE по пресетам CTR")
    axes[0].set_xlabel("")
    df.boxplot(column="inference_ms", by="ctr_preset", ax=axes[1])
    axes[1].set_title("Latency по пресетам CTR")
    axes[1].set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_07_ctr_preset_boxplot.png"), dpi=150)
    plt.close(fig)


def plot_param_importance(df: pd.DataFrame, dataset_key: str) -> None:
    if optuna is None or get_param_importances is None:
        return
    study = optuna.create_study(direction="minimize")
    param_cols = ["ctr_preset", "max_ctr_complexity", "iterations", "depth", "learning_rate"]
    depth_lo = int(df["depth"].min())
    depth_hi = int(df["depth"].max())
    for _, row in df.iterrows():
        params = {col: row[col] for col in param_cols if col in row}
        study.add_trial(
            optuna.create_trial(
                params=params,
                value=row["value"],
                distributions={
                    "ctr_preset": optuna.distributions.CategoricalDistribution(
                        sorted(df["ctr_preset"].unique())
                    ),
                    "max_ctr_complexity": optuna.distributions.IntDistribution(0, 1),
                    "iterations": optuna.distributions.IntDistribution(
                        int(df["iterations"].min()), int(df["iterations"].max()), step=50
                    ),
                    "depth": optuna.distributions.IntDistribution(depth_lo, depth_hi),
                    "learning_rate": optuna.distributions.FloatDistribution(0.03, 0.2, log=True),
                },
            )
        )
    importances = get_param_importances(study)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(importances.keys()), list(importances.values()), color="#4472C4")
    ax.set_xlabel("Относительная важность (fANOVA)")
    ax.set_title(f"Важность гиперпараметров ({dataset_key})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_08_param_importance.png"), dpi=150)
    plt.close(fig)


def plot_tpe_convergence(df: pd.DataFrame, dataset_key: str) -> None:
    window = max(3, len(df) // 10)
    df = df.sort_values("number").copy()
    df["feasible_rate"] = df["feasible"].astype(float).rolling(window, min_periods=1).mean()
    df["best_rmse"] = df["rmse"].cummin()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["number"], df["feasible_rate"], color="#70AD47", linewidth=2,
             label=f"доля допустимых (окно {window})")
    ax1.set_xlabel("Номер trial")
    ax1.set_ylabel("Доля feasible trials", color="#70AD47")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(df["number"], df["best_rmse"], color="#4472C4", linewidth=2, linestyle="--", label="лучший RMSE")
    ax2.set_ylabel("Лучший RMSE", color="#4472C4")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"Сходимость TPE ({dataset_key})")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / _prefix(dataset_key, "optuna_09_tpe_convergence.png"), dpi=150)
    plt.close(fig)


def generate_for_dataset(dataset_key: str, res: dict) -> None:
    df = trials_to_df(res["optuna_trials"])
    if df.empty:
        print(f"No trials for {dataset_key}, skipping figures")
        return
    baseline = dict(res["baseline"])
    baseline["constraints_max_ms"] = res["constraints"]["max_inference_ms"]
    max_rmse = baseline["rmse"] * (1 + res["constraints"]["max_rmse_degradation_pct"] / 100)

    plot_optimization_history(
        df, dataset_key,
        max_ms=baseline["constraints_max_ms"],
        max_rmse=max_rmse,
    )
    plot_metrics_over_trials(df, baseline, dataset_key)
    plot_feasible_region(df, res, dataset_key)
    plot_penalty_mechanism(df, dataset_key)
    plot_objective_vs_rmse(df, dataset_key)
    plot_param_effects(df, dataset_key)
    plot_ctr_preset_boxplot(df, dataset_key)
    plot_param_importance(df, dataset_key)
    plot_tpe_convergence(df, dataset_key)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    payload = load_payload()

    if "datasets" in payload:
        for key in payload["datasets"]:
            generate_for_dataset(key, payload["datasets"][key])
    else:
        generate_for_dataset("amazon", payload)

    print(f"Optuna figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
