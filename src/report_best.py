"""Выбор лучшей конфигурации в SLA-зоне для отчёта и графиков."""

from __future__ import annotations

from typing import Any

from benchmark import resolve_sla_ms


def sla_limits(res: dict) -> tuple[float, float]:
    baseline = res["baseline"]
    c = res["constraints"]
    max_rmse = baseline["rmse"] * (1 + c["max_rmse_degradation_pct"] / 100)
    max_ms = c["max_inference_ms"]
    return max_ms, max_rmse


def in_sla_box(rmse: float, inference_ms: float, max_ms: float, max_rmse: float) -> bool:
    return inference_ms <= max_ms and rmse <= max_rmse


def _candidates(res: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for t in res.get("optuna_trials", []):
        row = {
            "rmse": t["rmse"],
            "inference_ms": t["inference_ms"],
            "r2": t.get("r2"),
            "ctr_tables": t.get("ctr_tables"),
            "params": t.get("params", {}),
            "number": t.get("number"),
        }
        out.append((f"trial_{t['number']}", row))
    for r in res.get("references", []):
        out.append((r["label"], r))
    out.append(("baseline_default", res["baseline"]))
    if res.get("optimizer_best"):
        out.append(("optimizer_best", res["optimizer_best"]))
    return out


def pick_sla_best(res: dict, dataset_key: str = "amazon") -> dict[str, Any] | None:
    """Лучший RMSE среди конфигураций внутри SLA-области (по метрикам trials)."""
    max_ms, max_rmse = sla_limits(res)
    in_zone = [
        (name, row)
        for name, row in _candidates(res)
        if in_sla_box(row["rmse"], row["inference_ms"], max_ms, max_rmse)
    ]
    if not in_zone:
        return None
    name, row = min(in_zone, key=lambda x: x[1]["rmse"])
    return {"label": name, **row}


def enrich_results(payload: dict) -> dict:
    """Добавляет report_best и синхронизирует summary с optimizer_best в SLA."""
    if "datasets" not in payload:
        return payload

    for key, res in payload["datasets"].items():
        baseline_ms = res["baseline"]["inference_ms"]
        resolved = resolve_sla_ms(key, baseline_ms)
        res["constraints"]["max_inference_ms"] = resolved
        rb = pick_sla_best(res, key)
        ob = res["optimizer_best"]
        max_ms, max_rmse = sla_limits(res)

        if rb and in_sla_box(rb["rmse"], rb["inference_ms"], max_ms, max_rmse):
            if rb["rmse"] < ob["rmse"] or not in_sla_box(
                ob["rmse"], ob["inference_ms"], max_ms, max_rmse
            ):
                res["report_best"] = rb
            else:
                res["report_best"] = {**ob, "label": "optimizer_best"}
        elif in_sla_box(ob["rmse"], ob["inference_ms"], max_ms, max_rmse):
            res["report_best"] = {**ob, "label": "optimizer_best"}
        elif rb:
            res["report_best"] = rb
        else:
            res["report_best"] = ob

        # summary — от report_best, если optimizer_best вне SLA или хуже baseline
        _update_summary(res, use_report_best=True)
    return payload


def _update_summary(res: dict, use_report_best: bool = True) -> None:
    b = res["baseline"]
    ob = res["optimizer_best"]
    rb = res.get("report_best") if use_report_best else None
    oq = rb or ob  # качество — report_best если лучше, иначе optimizer
    ctr_base = b.get("ctr_tables") or 1
    ctr_best = ob.get("ctr_tables") or ctr_base
    max_rmse = b["rmse"] * (1 + res["constraints"]["max_rmse_degradation_pct"] / 100)
    res["summary"] = {
        "rmse_improvement_pct": round((b["rmse"] - oq["rmse"]) / b["rmse"] * 100, 2),
        "speedup_pct": round((b["inference_ms"] - ob["inference_ms"]) / b["inference_ms"] * 100, 2),
        "ctr_reduction_pct": round((ctr_base - ctr_best) / ctr_base * 100, 1),
        "quality_degradation_pct": round(max(0, (ob["rmse"] - b["rmse"]) / b["rmse"] * 100), 2),
        "sla_met": (
            ob["inference_ms"] <= res["constraints"]["max_inference_ms"]
            and ob["rmse"] <= max_rmse
        ),
    }


def _merge_optimizer(report: dict, current: dict) -> dict:
    merged = dict(current)
    merged["label"] = "optimizer_best"
    merged["rmse"] = report["rmse"]
    merged["inference_ms"] = report["inference_ms"]
    if report.get("r2") is not None:
        merged["r2"] = report["r2"]
    if report.get("ctr_tables") is not None:
        merged["ctr_tables"] = report["ctr_tables"]
    if report.get("params"):
        merged["params"] = report["params"]
    return merged
