"""Прототип подбора конфигураций CatBoost: SLA-HWS (иерархический warm-start)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor

from benchmark import (
    DEFAULT_INFERENCE_RUNS,
    count_ctr_tables,
    evaluate_classification,
    evaluate_regression,
    measure_inference_time,
    model_size_mb,
)


CTR_PRESETS = {
    "default": {},
    "borders_only": {
        "simple_ctr": ["Borders"],
        "combinations_ctr": [],
        "max_ctr_complexity": 1,
    },
    "counter_only": {
        "simple_ctr": ["Counter"],
        "combinations_ctr": [],
        "max_ctr_complexity": 1,
    },
    "no_combinations": {"max_ctr_complexity": 0},
    "compact_model": {"model_size_reg": 2, "max_ctr_complexity": 1},
}

# Якорные точки для фазы скрининга пресетов (фиксированная структура ансамбля)
ANCHOR_HYPER = {"iterations": 200, "depth": 6, "learning_rate": 0.1}


@dataclass
class OptimizationConstraints:
    max_inference_ms: float = 16.0
    max_rmse_degradation_pct: float = 5.0
    baseline_rmse: float | None = None
    latency_penalty: float = 1e6
    quality_penalty: float = 1e4


@dataclass
class TrialResult:
    label: str
    params: dict[str, Any]
    rmse: float
    r2: float
    inference_ms: float
    inference_ci_lower: float
    inference_ci_upper: float
    ctr_tables: int | None
    model_size_mb: float
    train_time_sec: float
    objective_value: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CatBoostOptimizer:
    """SLA-HWS: скрининг CTR-пресетов → warm-start → сфокусированный TPE."""

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        cat_features: list[str],
        constraints: OptimizationConstraints | None = None,
        random_seed: int = 42,
        tmp_dir: Path | None = None,
        task_type: str = "regression",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cat_features = cat_features
        self.task_type = task_type
        self.constraints = constraints or OptimizationConstraints()
        self.random_seed = random_seed
        self.tmp_dir = tmp_dir or Path("results/tmp_models")
        self.study: optuna.Study | None = None
        self.best_result: TrialResult | None = None
        self.baseline_result: TrialResult | None = None
        self.screening_log: list[dict[str, Any]] = []
        self.allowed_presets: list[str] = list(CTR_PRESETS.keys())
        self.search_bounds: dict[str, tuple] = {}

    def _cat_indices(self) -> list[int]:
        return [self.X_train.columns.get_loc(col) for col in self.cat_features]

    def train_and_evaluate(
        self,
        label: str,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.1,
        ctr_params: dict[str, Any] | None = None,
        inference_runs: int = DEFAULT_INFERENCE_RUNS,
    ) -> TrialResult:
        ctr_params = ctr_params or {}
        common = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=self.random_seed,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            **ctr_params,
        )
        if self.task_type == "classification":
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                **common,
            )
        else:
            model = CatBoostRegressor(
                loss_function="RMSE",
                **common,
            )

        train_start = time.perf_counter()
        model.fit(
            self.X_train,
            self.y_train,
            cat_features=self._cat_indices(),
            verbose=False,
        )
        train_time = time.perf_counter() - train_start

        if self.task_type == "classification":
            y_pred = model.predict_proba(self.X_test)[:, 1]
            metrics = evaluate_classification(self.y_test, y_pred)
        else:
            y_pred = model.predict(self.X_test)
            metrics = evaluate_regression(self.y_test, y_pred)
        inf = measure_inference_time(model, self.X_test, n_runs=inference_runs)
        ctr = count_ctr_tables(model)
        size_mb = model_size_mb(model, self.tmp_dir)

        inference_ms = inf["median"] * 1000
        objective, _ = self._objective_value(metrics["rmse"], inference_ms)

        return TrialResult(
            label=label,
            params={
                "iterations": iterations,
                "depth": depth,
                "learning_rate": learning_rate,
                **ctr_params,
            },
            rmse=metrics["rmse"],
            r2=metrics["r2"],
            inference_ms=inference_ms,
            inference_ci_lower=inf["ci_lower"] * 1000,
            inference_ci_upper=inf["ci_upper"] * 1000,
            ctr_tables=ctr.get("total"),
            model_size_mb=size_mb,
            train_time_sec=train_time,
            objective_value=objective,
            extra={"model": model, "ctr_by_type": ctr.get("by_type", {})},
        )

    def _objective_value(self, rmse: float, inference_ms: float) -> tuple[float, dict[str, float]]:
        c = self.constraints
        latency_penalty = 0.0
        quality_penalty = 0.0
        degradation_pct = 0.0

        if inference_ms > c.max_inference_ms:
            latency_penalty = c.latency_penalty * (inference_ms - c.max_inference_ms)

        if c.baseline_rmse is not None and c.baseline_rmse > 0:
            degradation_pct = (rmse - c.baseline_rmse) / c.baseline_rmse * 100
            if degradation_pct > c.max_rmse_degradation_pct:
                quality_penalty = c.quality_penalty * (
                    degradation_pct - c.max_rmse_degradation_pct
                )

        total = rmse + latency_penalty + quality_penalty
        meta = {
            "raw_rmse": rmse,
            "latency_penalty": latency_penalty,
            "quality_penalty": quality_penalty,
            "degradation_pct": degradation_pct,
            "feasible": float(latency_penalty == 0.0 and quality_penalty == 0.0),
        }
        return total, meta

    def fit_baseline(self, inference_runs: int = DEFAULT_INFERENCE_RUNS) -> TrialResult:
        self.baseline_result = self.train_and_evaluate(
            label="baseline_default",
            ctr_params={},
            inference_runs=inference_runs,
        )
        self.constraints.baseline_rmse = self.baseline_result.rmse
        return self.baseline_result

    def _screen_ctr_presets(self, inference_runs: int, top_k: int = 3) -> list[str]:
        """Фаза 1: ранжирование CTR-пресетов на якорных гиперпараметрах по RMSE."""
        self.screening_log = []
        ranked: list[tuple[str, float, TrialResult]] = []

        for preset_name, preset_params in CTR_PRESETS.items():
            params = dict(preset_params)

            result = self.train_and_evaluate(
                label=f"screen_{preset_name}",
                ctr_params=params,
                inference_runs=min(inference_runs, 20),
                **ANCHOR_HYPER,
            )
            ranked.append((preset_name, result.rmse, result))
            self.screening_log.append({
                "preset": preset_name,
                "objective": result.rmse,
                "rmse": result.rmse,
                "inference_ms": result.inference_ms,
                "feasible": result.inference_ms <= self.constraints.max_inference_ms,
            })

        ranked.sort(key=lambda x: x[1])
        self.preset_rank = [name for name, _, _ in ranked]
        # Все пресеты остаются в TPE; скрининг задаёт только приоритет warm-start
        self.allowed_presets = list(CTR_PRESETS.keys())
        return self.preset_rank[:3]

    def _infer_search_bounds(self, references: list[TrialResult]) -> None:
        """Фаза 2: сужение диапазонов по эталонам и скринингу."""
        c = self.constraints
        max_rmse = (c.baseline_rmse or 1e9) * (1 + c.max_rmse_degradation_pct / 100)

        good = [
            r for r in references
            if r.inference_ms <= c.max_inference_ms and r.rmse <= max_rmse
        ]
        if not good:
            good = sorted(references, key=lambda r: r.objective_value or r.rmse)[:3]

        best = min(good, key=lambda r: r.rmse)
        it = int(best.params.get("iterations", 200))
        dp = int(best.params.get("depth", 6))

        self.search_bounds = {
            "iterations": (max(100, it - 50), min(500, it + 200)),
            "depth": (max(4, dp - 1), min(8, dp + 2)),
            "lr": (0.04, 0.18),
        }

    def _build_warm_start_trials(
        self,
        references: list[TrialResult],
    ) -> list[dict[str, Any]]:
        """Фаза 2: seed-trials из эталонов и доменных якорей НИР1."""
        seeds: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            preset: str,
            it: int,
            dp: int,
            lr: float,
            max_ctr: int = 1,
            *,
            force: bool = False,
        ) -> None:
            if not force and preset not in self.allowed_presets:
                return
            key = f"{preset}|{it}|{dp}|{lr:.4f}|{max_ctr}"
            if key in seen:
                return
            seen.add(key)
            seeds.append({
                "ctr_preset": preset,
                "iterations": it,
                "depth": dp,
                "learning_rate": lr,
                "max_ctr_complexity": max_ctr,
            })

        # Эталоны, попавшие в топ-пресеты
        for ref in references:
            preset = ref.params.get("ctr_preset")
            if not preset:
                for name, p in CTR_PRESETS.items():
                    if ref.label.replace("_default", "") in name or name in ref.label:
                        preset = name
                        break
                if ref.label == "max_ctr_complexity_1":
                    preset = "default"
                elif ref.label == "max_ctr_complexity_0":
                    preset = "no_combinations"
                elif ref.label in CTR_PRESETS:
                    preset = ref.label

            if preset and preset in self.allowed_presets:
                add(
                    preset,
                    int(ref.params.get("iterations", 200)),
                    int(ref.params.get("depth", 6)),
                    float(ref.params.get("learning_rate", 0.1)),
                    int(ref.params.get("max_ctr_complexity", 1)),
                )

        # Доменные якоря: для классификации — около baseline, для регрессии — якоря НИР1
        if self.task_type == "classification":
            add("default", 200, 6, 0.1, 1, force=True)
            add("default", 250, 6, 0.08, 1, force=True)
            add("default", 300, 6, 0.06, 1, force=True)
            add("no_combinations", 200, 6, 0.1, 0, force=True)
            add("compact_model", 250, 6, 0.08, 1, force=True)
        else:
            add("default", 350, 7, 0.164, 1, force=True)
            add("default", 300, 7, 0.12, 1, force=True)
        # Доп. seed-ы для top-пресетов по результатам скрининга
        for rank, preset in enumerate(getattr(self, "preset_rank", [])[:3]):
            reps = 3 - rank
            for _ in range(reps):
                add(preset, 250, 7, 0.14, 1)

        return seeds[:16]

    def _trial_params_to_ctr(self, trial_params: dict[str, Any]) -> dict[str, Any]:
        preset = CTR_PRESETS[trial_params["ctr_preset"]]
        merged = {**preset, "max_ctr_complexity": trial_params["max_ctr_complexity"]}
        return merged

    def _sla_constraint_values(
        self,
        inference_ms: float,
        degradation_pct: float,
    ) -> tuple[float, float]:
        """Нарушения SLA: >0 — ограничение не выполнено (формат Optuna constraints_func)."""
        c = self.constraints
        return (
            inference_ms - c.max_inference_ms,
            degradation_pct - c.max_rmse_degradation_pct,
        )

    def _constraints_func(self, trial: optuna.trial.FrozenTrial) -> list[float]:
        inference_ms = trial.user_attrs.get("inference_ms", float("inf"))
        degradation_pct = trial.user_attrs.get("degradation_pct", float("inf"))
        lat_v, qual_v = self._sla_constraint_values(inference_ms, degradation_pct)
        return [lat_v, qual_v]

    def _select_best_trial(
        self,
        complete: list[optuna.trial.FrozenTrial],
    ) -> optuna.trial.FrozenTrial:
        c = self.constraints
        max_rmse = (c.baseline_rmse or float("inf")) * (1 + c.max_rmse_degradation_pct / 100)

        def in_sla(t: optuna.trial.FrozenTrial) -> bool:
            ms = t.user_attrs.get("inference_ms", 1e9)
            rmse = t.user_attrs.get("rmse", 1e9)
            return ms <= c.max_inference_ms and rmse <= max_rmse

        sla_ok = [t for t in complete if in_sla(t)]
        quality_ok = [
            t for t in complete
            if t.user_attrs.get("degradation_pct", 100) <= c.max_rmse_degradation_pct
        ]

        if sla_ok:
            return min(sla_ok, key=lambda t: t.user_attrs.get("rmse", float("inf")))
        if quality_ok:
            return min(quality_ok, key=lambda t: t.user_attrs.get("rmse", float("inf")))
        # Не выбирать конфигурацию сильно хуже baseline по качеству
        baseline_rmse = c.baseline_rmse or float("inf")
        not_much_worse = [
            t for t in complete
            if t.user_attrs.get("rmse", float("inf")) <= baseline_rmse * 1.10
        ]
        if not_much_worse:
            return min(not_much_worse, key=lambda t: (
                t.user_attrs.get("rmse", float("inf")),
                t.user_attrs.get("inference_ms", float("inf")),
            ))
        return min(complete, key=lambda t: t.user_attrs.get("rmse", float("inf")))

    def _result_from_trial(
        self,
        trial: optuna.trial.FrozenTrial,
        inference_runs: int,
    ) -> TrialResult:
        merged_ctr = self._trial_params_to_ctr(trial.params)
        result = self.train_and_evaluate(
            label="optimizer_best",
            iterations=trial.params["iterations"],
            depth=trial.params["depth"],
            learning_rate=trial.params["learning_rate"],
            ctr_params=merged_ctr,
            inference_runs=inference_runs,
        )
        result.params = {
            "iterations": trial.params["iterations"],
            "depth": trial.params["depth"],
            "learning_rate": trial.params["learning_rate"],
            **merged_ctr,
            "ctr_preset": trial.params["ctr_preset"],
            "optuna_trial": trial.number,
        }
        return result

    def _run_tpe_study(
        self,
        n_trials: int,
        inference_runs: int,
        timeout_sec: int | None,
        show_progress: bool,
        warm_seeds: list[dict[str, Any]] | None,
        study_name: str,
    ) -> None:
        it_lo, it_hi = self.search_bounds["iterations"]
        dp_lo, dp_hi = self.search_bounds["depth"]
        lr_lo, lr_hi = self.search_bounds["lr"]

        def objective(trial: optuna.Trial) -> float:
            ctr_preset = trial.suggest_categorical("ctr_preset", self.allowed_presets)
            params = dict(CTR_PRESETS[ctr_preset])
            max_ctr = trial.suggest_int("max_ctr_complexity", 0, 1)
            if ctr_preset in {"borders_only", "counter_only", "compact_model"}:
                max_ctr = min(max_ctr, 1)
            params["max_ctr_complexity"] = max_ctr

            result = self.train_and_evaluate(
                label=f"trial_{trial.number}",
                iterations=trial.suggest_int("iterations", it_lo, it_hi, step=50),
                depth=trial.suggest_int("depth", dp_lo, dp_hi),
                learning_rate=trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                ctr_params=params,
                inference_runs=inference_runs,
            )

            objective_val, penalty_meta = self._objective_value(
                result.rmse, result.inference_ms
            )

            trial.set_user_attr("rmse", result.rmse)
            trial.set_user_attr("r2", result.r2)
            trial.set_user_attr("inference_ms", result.inference_ms)
            trial.set_user_attr("ctr_tables", result.ctr_tables)
            trial.set_user_attr("model_size_mb", result.model_size_mb)
            for key, value in penalty_meta.items():
                trial.set_user_attr(key, value)

            # Optuna минимизирует RMSE; SLA передаётся в TPE через constraints_func
            return result.rmse

        sampler_kwargs: dict[str, Any] = {
            "seed": self.random_seed,
            "multivariate": True,
            "warn_independent_sampling": False,
        }
        if study_name != "catboost_naive_tpe":
            sampler_kwargs["constraints_func"] = self._constraints_func

        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
        )

        if warm_seeds:
            for seed in warm_seeds:
                self.study.enqueue_trial(seed)

        if study_name == "catboost_sla_hws":
            if self.task_type == "classification":
                anchors = (
                    {"ctr_preset": "default", "iterations": 200, "depth": 6,
                     "learning_rate": 0.1, "max_ctr_complexity": 1},
                    {"ctr_preset": "no_combinations", "iterations": 250, "depth": 6,
                     "learning_rate": 0.08, "max_ctr_complexity": 0},
                )
            else:
                anchors = (
                    {"ctr_preset": "default", "iterations": 350, "depth": 7,
                     "learning_rate": 0.164, "max_ctr_complexity": 1},
                )
            for anchor in anchors:
                self.study.enqueue_trial(anchor)

        optuna.logging.set_verbosity(
            optuna.logging.WARNING if not show_progress else optuna.logging.INFO
        )

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_sec,
            show_progress_bar=show_progress,
        )

    def optimize(
        self,
        n_trials: int = 25,
        timeout_sec: int | None = None,
        inference_runs: int = DEFAULT_INFERENCE_RUNS,
        show_progress: bool = True,
        references: list[TrialResult] | None = None,
        mode: str = "sla_hws",
    ) -> TrialResult:
        if self.baseline_result is None:
            self.fit_baseline(inference_runs=inference_runs)

        refs = references or []

        if mode == "naive_tpe":
            self.allowed_presets = list(CTR_PRESETS.keys())
            if self.task_type == "classification":
                self.search_bounds = {
                    "iterations": (150, 400),
                    "depth": (4, 8),
                    "lr": (0.03, 0.2),
                }
            else:
                self.search_bounds = {
                    "iterations": (100, 400),
                    "depth": (4, 7),
                    "lr": (0.03, 0.2),
                }
            warm_seeds = None
            self.screening_log = []
            self._warm_start_count = 0
        else:
            self._screen_ctr_presets(inference_runs=inference_runs, top_k=3)
            self._infer_search_bounds(refs if refs else [self.baseline_result])
            warm_seeds = self._build_warm_start_trials(refs)
            self._warm_start_count = len(warm_seeds or [])

        self._run_tpe_study(
            n_trials=n_trials,
            inference_runs=inference_runs,
            timeout_sec=timeout_sec,
            show_progress=show_progress,
            warm_seeds=warm_seeds,
            study_name=f"catboost_{mode}",
        )

        complete = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete:
            raise RuntimeError("Optuna study produced no complete trials")

        best_trial = self._select_best_trial(complete)
        self.best_result = self._result_from_trial(best_trial, inference_runs)
        return self.best_result

    def optimize_naive(
        self,
        n_trials: int = 25,
        inference_runs: int = DEFAULT_INFERENCE_RUNS,
        show_progress: bool = True,
    ) -> TrialResult:
        """Наивный TPE: все пресеты, полное пространство, без скрининга и warm-start."""
        return self.optimize(
            n_trials=n_trials,
            inference_runs=inference_runs,
            show_progress=show_progress,
            mode="naive_tpe",
        )

    def export_study_trials(self) -> list[dict[str, Any]]:
        if self.study is None:
            return []
        rows = []
        for t in self.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            rows.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "rmse": t.user_attrs.get("rmse"),
                "r2": t.user_attrs.get("r2"),
                "inference_ms": t.user_attrs.get("inference_ms"),
                "ctr_tables": t.user_attrs.get("ctr_tables"),
                "raw_rmse": t.user_attrs.get("raw_rmse"),
                "latency_penalty": t.user_attrs.get("latency_penalty", 0),
                "quality_penalty": t.user_attrs.get("quality_penalty", 0),
                "degradation_pct": t.user_attrs.get("degradation_pct", 0),
                "feasible": bool(t.user_attrs.get("feasible", 0)),
            })
        return rows

    def export_algorithm_meta(self, mode: str = "sla_hws") -> dict[str, Any]:
        return {
            "name": mode,
            "phases": (
                [
                    "скрининг CTR-пресетов (ранжирование, без отсечения)",
                    "warm-start с приоритетом top-пресетов",
                    "TPE по полному набору пресетов в суженных bounds",
                ]
                if mode == "sla_hws"
                else ["TPE по полному пространству без warm-start и без constraints_func"]
            ),
            "constraints_func": mode == "sla_hws",
            "allowed_presets": self.allowed_presets,
            "preset_rank": getattr(self, "preset_rank", []),
            "search_bounds": self.search_bounds,
            "screening": self.screening_log,
            "warm_start_count": getattr(self, "_warm_start_count", 0),
            "inference_runs": DEFAULT_INFERENCE_RUNS,
        }

    def run_reference_configs(self, inference_runs: int = DEFAULT_INFERENCE_RUNS) -> list[TrialResult]:
        if self.baseline_result is None:
            self.fit_baseline(inference_runs=inference_runs)

        configs = [
            ("max_ctr_complexity_0", {"max_ctr_complexity": 0}),
            ("max_ctr_complexity_1", {"max_ctr_complexity": 1}),
            ("borders_only", {**CTR_PRESETS["borders_only"], "ctr_preset": "borders_only"}),
            ("counter_only", {**CTR_PRESETS["counter_only"], "ctr_preset": "counter_only"}),
            ("compact_model", {**CTR_PRESETS["compact_model"], "ctr_preset": "compact_model"}),
        ]
        results = [self.baseline_result]
        for label, params in configs:
            ctr = {k: v for k, v in params.items() if k != "ctr_preset"}
            r = self.train_and_evaluate(label=label, ctr_params=ctr, inference_runs=inference_runs)
            if "ctr_preset" in params:
                r.params["ctr_preset"] = params["ctr_preset"]
            results.append(r)
        return results
