"""Замеры инференса, подсчёт CTR-таблиц и загрузка датасетов."""

from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

from datasets import DATASET_SPECS, list_datasets, load_dataset  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent

AMAZON_FEATURE_LABELS = {
    "RESOURCE": "Ресурс / приложение",
    "MGR_ID": "ID менеджера",
    "ROLE_ROLLUP_1": "Уровень роли (rollup 1)",
    "ROLE_ROLLUP_2": "Уровень роли (rollup 2)",
    "ROLE_DEPTNAME": "Подразделение",
    "ROLE_TITLE": "Должность",
    "ROLE_FAMILY_DESC": "Семейство ролей (описание)",
    "ROLE_FAMILY": "Семейство ролей",
    "ROLE_CODE": "Код роли",
}

DEFAULT_SLA_MS: dict[str, float | None] = {
    "amazon": None,
    "students": None,
    "ames": None,
}

# Единый протокол замеров для всего pipeline (baseline, trials, финальная модель)
DEFAULT_INFERENCE_RUNS = 50
FINAL_INFERENCE_RUNS = 100
INFERENCE_WARMUP_RUNS = 10


def measure_inference_time(
    model,
    X,
    n_runs: int = DEFAULT_INFERENCE_RUNS,
    confidence_level: float = 0.95,
    warmup_runs: int = INFERENCE_WARMUP_RUNS,
) -> dict[str, float]:
    """Среднее время инференса по n_runs прогонам с доверительным интервалом."""
    for _ in range(warmup_runs):
        _ = model.predict(X)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        times.append(time.perf_counter() - start)

    times_arr = np.array(times)
    mean_time = float(np.mean(times_arr))
    median_time = float(np.median(times_arr))
    std_time = float(np.std(times_arr, ddof=1)) if n_runs > 1 else 0.0

    if n_runs > 1:
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_runs - 1)
        margin = t_critical * (std_time / np.sqrt(n_runs))
        ci_lower = mean_time - margin
        ci_upper = mean_time + margin
    else:
        ci_lower = ci_upper = mean_time

    return {
        "mean": mean_time,
        "median": median_time,
        "std": std_time,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_runs": n_runs,
    }


def count_ctr_tables(model) -> dict[str, Any]:
    """Возвращает количество CTR-таблиц и разбивку по типам."""
    summary: dict[str, Any] = {"total": 0, "by_type": {}}
    tmp_name = None
    try:
        with NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
            tmp_name = tmp.name
        model.save_model(tmp_name, format="json")
        with open(tmp_name, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        ctr_data = model_json.get("ctr_data")

        def bump(ctr_type: str) -> None:
            summary["by_type"][ctr_type] = summary["by_type"].get(ctr_type, 0) + 1

        if isinstance(ctr_data, dict):
            if "ctr_table_data" in ctr_data:
                tables = ctr_data.get("ctr_table_data") or []
                summary["total"] = len(tables)
                for table in tables:
                    bump(table.get("ctr_type") or table.get("type") or "unknown")
            else:
                summary["total"] = len(ctr_data)
                for key in ctr_data.keys():
                    ctr_type = "unknown"
                    if isinstance(key, str):
                        try:
                            ctr_type = json.loads(key).get("type", "unknown")
                        except json.JSONDecodeError:
                            pass
                    bump(ctr_type)
        elif isinstance(ctr_data, list):
            summary["total"] = len(ctr_data)
            for table in ctr_data:
                bump(table.get("ctr_type") or table.get("type") or "unknown")
    except Exception:
        summary["total"] = None
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)

    return summary


def model_size_mb(model, tmp_dir: Path) -> float:
    """Размер сохранённой модели в мегабайтах."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "_size_probe.cbm"
    model.save_model(path)
    size = path.stat().st_size / (1024 * 1024)
    path.unlink(missing_ok=True)
    return round(size, 3)


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def evaluate_classification(y_true, y_proba) -> dict[str, float]:
    """Logloss (→ rmse) и AUC (→ r2) для совместимости JSON-пipeline."""
    y = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
    ll = float(log_loss(y, proba, labels=[0, 1]))
    auc = float(roc_auc_score(y, proba))
    return {"rmse": ll, "r2": auc}


def metric_labels(task_type: str) -> dict[str, str]:
    if task_type == "classification":
        return {
            "primary": "Logloss",
            "secondary": "AUC",
            "primary_short": "Logloss",
            "improvement_label": "улучшение Logloss",
        }
    return {
        "primary": "RMSE",
        "secondary": "R²",
        "primary_short": "RMSE",
        "improvement_label": "улучшение RMSE",
    }


def resolve_sla_ms(dataset_name: str, baseline_inference_ms: float) -> float:
    """SLA по latency.

    Малые датасеты (baseline < 10 мс): не медленнее baseline.
    Остальные (Amazon и др.): улучшение latency минимум на 10 % от baseline.
    """
    if baseline_inference_ms < 10.0:
        return round(baseline_inference_ms, 3)
    return round(baseline_inference_ms * 0.90, 3)
