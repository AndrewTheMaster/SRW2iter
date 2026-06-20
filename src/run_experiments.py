#!/usr/bin/env python3
"""Запуск экспериментов НИР2 на нескольких датасетах."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark import (
    AMAZON_FEATURE_LABELS,
    DEFAULT_INFERENCE_RUNS,
    FINAL_INFERENCE_RUNS,
    measure_inference_time,
    metric_labels,
    resolve_sla_ms,
)
from catboost_optimizer import CTR_PRESETS, CatBoostOptimizer, OptimizationConstraints, TrialResult
from datasets import DATASET_SPECS, list_datasets, load_dataset


def trial_to_dict(result: TrialResult) -> dict:
    data = asdict(result)
    data.pop("extra", None)
    return data


def _refresh_latency(result: TrialResult, optimizer: CatBoostOptimizer, n_runs: int) -> None:
    model = result.extra.get("model")
    if model is None:
        return
    inf = measure_inference_time(model, optimizer.X_test, n_runs=n_runs)
    result.inference_ms = inf["median"] * 1000
    result.inference_ci_lower = inf["ci_lower"] * 1000
    result.inference_ci_upper = inf["ci_upper"] * 1000


def paired_final_measurement(
    optimizer: CatBoostOptimizer,
    best: TrialResult,
    inference_runs: int = FINAL_INFERENCE_RUNS,
) -> tuple[TrialResult, TrialResult]:
    """Парный финальный замер baseline и best подряд (устраняет drift CPU/кэша)."""
    time.sleep(3)

    baseline = optimizer.train_and_evaluate(
        label="baseline_default",
        inference_runs=inference_runs,
    )

    preset = best.params.get("ctr_preset", "default")
    ctr = {k: v for k, v in best.params.items()
           if k not in {"iterations", "depth", "learning_rate", "ctr_preset", "optuna_trial"}}
    if preset in CTR_PRESETS:
        ctr = {**CTR_PRESETS[preset], **ctr}

    best_final = optimizer.train_and_evaluate(
        label="optimizer_best",
        iterations=int(best.params["iterations"]),
        depth=int(best.params["depth"]),
        learning_rate=float(best.params["learning_rate"]),
        ctr_params=ctr,
        inference_runs=inference_runs,
    )
    best_final.params = dict(best.params)

    optimizer.constraints.baseline_rmse = baseline.rmse

    # Повторный inference-only pass на тех же моделях (min медианы)
    for r in (baseline, best_final):
        first = r.inference_ms
        _refresh_latency(r, optimizer, inference_runs)
        second = r.inference_ms
        r.inference_ms = min(first, second)

    return baseline, best_final


def run_single_dataset(
    dataset_name: str,
    n_trials: int,
    max_rmse_degradation_pct: float,
    inference_runs: int,
    max_inference_ms: float | None,
    out: Path,
) -> dict:
    print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
    data = load_dataset(dataset_name)
    spec = DATASET_SPECS[dataset_name]

    tmp_dir = out / "tmp_models" / dataset_name
    constraints = OptimizationConstraints(
        max_inference_ms=max_inference_ms or 999.0,
        max_rmse_degradation_pct=max_rmse_degradation_pct,
    )

    optimizer = CatBoostOptimizer(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        cat_features=data["cat_features"],
        constraints=constraints,
        tmp_dir=tmp_dir,
        task_type=spec.task_type,
    )

    baseline = optimizer.fit_baseline(inference_runs=inference_runs)
    if max_inference_ms is None:
        constraints.max_inference_ms = resolve_sla_ms(dataset_name, baseline.inference_ms)
    print(
        f"Baseline (screening): RMSE={baseline.rmse:.2f}, inference={baseline.inference_ms:.2f} ms, "
        f"CTR={baseline.ctr_tables}, SLA={constraints.max_inference_ms:.2f} ms"
    )

    references = optimizer.run_reference_configs(inference_runs=inference_runs)
    best = optimizer.optimize(
        n_trials=n_trials,
        inference_runs=inference_runs,
        references=references,
    )

    params_path = tmp_dir / "best_params.json"
    params_path.write_text(
        json.dumps(best.params, ensure_ascii=False), encoding="utf-8"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "remeasure_final.py"),
            dataset_name,
            str(params_path),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        print(f"Warning: cold remeasure failed, using in-process paired:\n{proc.stderr}")
        baseline, best = paired_final_measurement(optimizer, best)
        baseline_d, best_d = trial_to_dict(baseline), trial_to_dict(best)
    else:
        rem = json.loads(proc.stdout)
        baseline_d, best_d = rem["baseline"], rem["optimizer_best"]

    constraints.baseline_rmse = baseline_d["rmse"]
    if max_inference_ms is None:
        constraints.max_inference_ms = resolve_sla_ms(dataset_name, baseline_d["inference_ms"])

    print(
        f"Final (cold): baseline inference={baseline_d['inference_ms']:.2f} ms, "
        f"best RMSE={best_d['rmse']:.2f}, inference={best_d['inference_ms']:.2f} ms, "
        f"CTR={best_d['ctr_tables']}"
    )

    rmse_gain = (baseline_d["rmse"] - best_d["rmse"]) / baseline_d["rmse"] * 100
    speed_gain = (
        (baseline_d["inference_ms"] - best_d["inference_ms"]) / baseline_d["inference_ms"] * 100
    )
    ctr_base = baseline_d["ctr_tables"] or 1
    ctr_best = best_d["ctr_tables"] or ctr_base
    ctr_reduction = (ctr_base - ctr_best) / ctr_base * 100
    max_rmse = baseline_d["rmse"] * (1 + constraints.max_rmse_degradation_pct / 100)
    sla_met = (
        best_d["inference_ms"] <= constraints.max_inference_ms
        and best_d["rmse"] <= max_rmse
    )

    labels = metric_labels(spec.task_type)
    dataset_meta = {
        "key": dataset_name,
        "title": spec.title_ru,
        "description": spec.description,
        "source": spec.source,
        "target_col": data["target_col"],
        "task_type": spec.task_type,
        "metric_labels": labels,
        "n_rows": data["n_rows"],
        "n_cols": data["n_cols"],
        "n_test": data["n_test"],
        "n_cat": len(data["cat_features"]),
        "n_num": len(data["num_features"]),
        "cat_features": data["cat_features"],
        "cardinality": data["cardinality"],
        "categorical_labels_ru": AMAZON_FEATURE_LABELS if dataset_name == "amazon" else {},
    }

    return {
        "constraints": asdict(constraints),
        "baseline": baseline_d,
        "optimizer_best": best_d,
        "references": [trial_to_dict(r) for r in references],
        "dataset": dataset_meta,
        "optuna_best_params": best.params,
        "optuna_n_trials": n_trials,
        "optuna_trials": optimizer.export_study_trials(),
        "algorithm": optimizer.export_algorithm_meta(mode="sla_hws"),
        "measurement": {
            "inference_runs": inference_runs,
            "final_inference_runs": FINAL_INFERENCE_RUNS,
            "warmup_runs": 10,
            "latency_stat": "median",
            "paired_final": True,
        },
        "summary": {
            "rmse_improvement_pct": round(rmse_gain, 2),
            "speedup_pct": round(speed_gain, 2),
            "ctr_reduction_pct": round(ctr_reduction, 1),
            "sla_met": sla_met,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost inference optimizer — multi-dataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_datasets(),
        choices=list_datasets(),
        help="Датасеты для прогона",
    )
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-inference-ms", type=float, default=None,
                        help="Фиксированный SLA (только если задан явно)")
    parser.add_argument("--max-rmse-degradation-pct", type=float, default=5.0)
    parser.add_argument("--inference-runs", type=int, default=DEFAULT_INFERENCE_RUNS)
    parser.add_argument("--with-ablation", action="store_true",
                        help="Ablation naive TPE vs SLA-HWS на primary-датасете (Amazon)")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Сохранить результаты других датасетов из experiment_results.json",
    )
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    results_path = out / "experiment_results.json"
    if args.merge and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            prev = json.load(f)
        all_results.update(prev.get("datasets", {}))
    for name in args.datasets:
        all_results[name] = run_single_dataset(
            dataset_name=name,
            n_trials=args.n_trials,
            max_rmse_degradation_pct=args.max_rmse_degradation_pct,
            inference_runs=args.inference_runs,
            max_inference_ms=args.max_inference_ms,
            out=out,
        )

    payload = {
        "n_trials": args.n_trials,
        "inference_runs": args.inference_runs,
        "datasets": all_results,
        "primary_dataset": "amazon",
    }

    from report_best import enrich_results

    payload = enrich_results(payload)

    with open(out / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.with_ablation:
        from run_ablation import main as run_ablation_main
        import sys as _sys
        _argv = _sys.argv
        _sys.argv = ["run_ablation.py", "--n-trials", str(args.n_trials),
                     "--inference-runs", str(args.inference_runs), "--output", str(out)]
        try:
            run_ablation_main()
        finally:
            _sys.argv = _argv

    # Сводная таблица
    rows = []
    for key, res in all_results.items():
        b, o = res["baseline"], res["optimizer_best"]
        s = res["summary"]
        rows.append({
            "dataset": key,
            "title": res["dataset"]["title"],
            "baseline_rmse": b["rmse"],
            "best_rmse": o["rmse"],
            "rmse_improvement_pct": s["rmse_improvement_pct"],
            "baseline_inference_ms": b["inference_ms"],
            "best_inference_ms": o["inference_ms"],
            "speedup_pct": s["speedup_pct"],
            "baseline_ctr": b["ctr_tables"],
            "best_ctr": o["ctr_tables"],
            "ctr_reduction_pct": s["ctr_reduction_pct"],
            "sla_ms": res["constraints"]["max_inference_ms"],
            "sla_met": s["sla_met"],
        })

    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "comparison_for_report.csv", index=False)

    try:
        from generate_optuna_figures import main as plot_optuna
        plot_optuna()
        print("Optuna figures generated.")
    except Exception as e:
        print(f"Warning: Optuna figures: {e}")

    try:
        from generate_comparison_figures import main as plot_compare
        plot_compare()
        print("Comparison figures generated.")
    except Exception as e:
        print(f"Warning: Comparison figures: {e}")

    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
