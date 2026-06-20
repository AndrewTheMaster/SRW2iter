#!/usr/bin/env python3
"""Холодный парный замер baseline/best в отдельном процессе (без нагрузки от Optuna)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from catboost_optimizer import CatBoostOptimizer, OptimizationConstraints, TrialResult
from datasets import DATASET_SPECS, load_dataset
from run_experiments import paired_final_measurement, trial_to_dict


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: remeasure_final.py <dataset> <best_params.json>", file=sys.stderr)
        sys.exit(1)

    dataset_name = sys.argv[1]
    params_path = Path(sys.argv[2])
    best_params = json.loads(params_path.read_text(encoding="utf-8"))

    data = load_dataset(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    tmp_dir = params_path.parent / "remeasure"
    opt = CatBoostOptimizer(
        data["X_train"], data["y_train"], data["X_test"], data["y_test"],
        data["cat_features"], OptimizationConstraints(), tmp_dir=tmp_dir,
        task_type=spec.task_type,
    )

    stub = TrialResult(
        label="optimizer_best",
        params=best_params,
        rmse=0, r2=0, inference_ms=0, inference_ci_lower=0, inference_ci_upper=0,
        ctr_tables=0, model_size_mb=0, train_time_sec=0,
    )
    baseline, best = paired_final_measurement(opt, stub)
    print(json.dumps({
        "baseline": trial_to_dict(baseline),
        "optimizer_best": trial_to_dict(best),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
