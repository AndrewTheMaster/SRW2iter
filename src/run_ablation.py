#!/usr/bin/env python3
"""Ablation: наивный TPE vs SLA-HWS (одинаковый бюджет trials)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark import DEFAULT_INFERENCE_RUNS, resolve_sla_ms
from catboost_optimizer import CatBoostOptimizer, OptimizationConstraints
from datasets import DATASET_SPECS, load_dataset
from run_experiments import paired_final_measurement, trial_to_dict

PRIMARY = "amazon"


def _sla_stats(
    trials: list[dict],
    max_ms: float,
    max_rmse: float,
    baseline_rmse: float,
) -> dict:
    in_zone = [
        t for t in trials
        if t.get("inference_ms", 1e9) <= max_ms and t.get("rmse", 1e9) <= max_rmse
    ]
    better_rmse = [t for t in trials if t.get("rmse", 1e9) < baseline_rmse]
    first_in_zone = None
    first_better = None
    for t in sorted(trials, key=lambda x: x["number"]):
        if first_in_zone is None and t.get("inference_ms", 1e9) <= max_ms and t.get("rmse", 1e9) <= max_rmse:
            first_in_zone = t["number"]
        if first_better is None and t.get("rmse", 1e9) < baseline_rmse:
            first_better = t["number"]
    return {
        "trials_total": len(trials),
        "trials_in_sla": len(in_zone),
        "trials_better_rmse": len(better_rmse),
        "first_in_sla_trial": first_in_zone,
        "first_better_rmse_trial": first_better,
        "best_in_sla_rmse": min((t["rmse"] for t in in_zone), default=None),
        "best_rmse": min((t["rmse"] for t in trials), default=None),
    }


def run_mode(
    mode: str,
    n_trials: int,
    inference_runs: int,
    out: Path,
) -> dict:
    data = load_dataset(PRIMARY)
    spec = DATASET_SPECS[PRIMARY]
    tmp_dir = out / "tmp_models" / f"ablation_{mode}"
    constraints = OptimizationConstraints()
    opt = CatBoostOptimizer(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"],
        data["cat_features"], constraints, tmp_dir=tmp_dir, task_type=spec.task_type,
    )
    baseline = opt.fit_baseline(inference_runs=inference_runs)
    constraints.max_inference_ms = resolve_sla_ms(PRIMARY, baseline.inference_ms)
    constraints.baseline_rmse = baseline.rmse

    references = opt.run_reference_configs(inference_runs=inference_runs) if mode == "sla_hws" else []
    best = opt.optimize(
        n_trials=n_trials,
        inference_runs=inference_runs,
        references=references,
        mode=mode,
        show_progress=True,
    )
    params_path = tmp_dir / "best_params.json"
    params_path.write_text(json.dumps(best.params, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "remeasure_final.py"), PRIMARY, str(params_path)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode == 0:
        rem = json.loads(proc.stdout)
        baseline_d, best_d = rem["baseline"], rem["optimizer_best"]
    else:
        baseline, best = paired_final_measurement(opt, best)
        baseline_d, best_d = trial_to_dict(baseline), trial_to_dict(best)

    constraints.baseline_rmse = baseline_d["rmse"]
    constraints.max_inference_ms = resolve_sla_ms(PRIMARY, baseline_d["inference_ms"])

    trials = opt.export_study_trials()
    max_rmse = baseline_d["rmse"] * (1 + constraints.max_rmse_degradation_pct / 100)
    stats = _sla_stats(trials, constraints.max_inference_ms, max_rmse, baseline_d["rmse"])

    return {
        "mode": mode,
        "baseline": baseline_d,
        "best": best_d,
        "constraints": asdict(constraints),
        "trials": trials,
        "stats": stats,
        "algorithm": opt.export_algorithm_meta(mode=mode),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--inference-runs", type=int, default=DEFAULT_INFERENCE_RUNS)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results")
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"Ablation on {PRIMARY}: naive_tpe vs sla_hws")
    results = {}
    for mode in ("naive_tpe", "sla_hws"):
        print(f"\n--- {mode} ---")
        results[mode] = run_mode(mode, args.n_trials, args.inference_runs, out)
        b = results[mode]["best"]
        s = results[mode]["stats"]
        print(
            f"Best quality={b['rmse']:.4f}, inference={b['inference_ms']:.2f} ms, "
            f"better-quality trials={s['trials_better_rmse']}/{s['trials_total']}, "
            f"first better at #{s['first_better_rmse_trial']}"
        )

    payload_path = out / "experiment_results.json"
    payload = {}
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))

    payload["ablation"] = {
        "n_trials": args.n_trials,
        "inference_runs": args.inference_runs,
        "dataset": PRIMARY,
        **results,
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAblation saved to {payload_path}")


if __name__ == "__main__":
    main()
