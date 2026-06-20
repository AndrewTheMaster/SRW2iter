# SRW2iter — SLA-HWS: оптимизация инференса CatBoost

Вторая итерация научно-исследовательской работы (ITMO, P4135).

**Тема:** оптимизация инференса CatBoost с категориальными признаками под production-SLA.

**Алгоритм SLA-HWS:** скрининг CTR-пресетов → warm-start → TPE с `constraints_func` (SLA-aware), objective = RMSE / Logloss без scalarization.

## Структура репозитория

```
├── src/                          # исходный код прототипа
│   ├── catboost_optimizer.py     # SLA-HWS + Optuna
│   ├── benchmark.py              # latency, CTR-таблицы, метрики
│   ├── datasets.py               # Amazon, Students, Ames
│   ├── prepare_datasets.py       # загрузка CSV
│   ├── run_experiments.py        # полный pipeline
│   ├── run_ablation.py           # naive TPE vs SLA-HWS
│   ├── remeasure_final.py        # cold-замер baseline/best
│   ├── report_best.py            # SLA-aware выбор конфигурации
│   ├── generate_comparison_figures.py
│   └── generate_optuna_figures.py
├── data/                         # CSV после prepare_datasets.py (не в git)
├── results/
│   ├── experiment_results.json   # результаты прогона
│   ├── comparison_for_report.csv
│   └── figures/                  # 8 рисунков из отчёта
└── requirements.txt
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# скачать датасеты (CSV не хранятся в репозитории)
cd src && python prepare_datasets.py

# полный прогон (Amazon 50 trials, Students/Ames 25)
python run_experiments.py --datasets amazon students ames --n-trials 50 --with-ablation --merge

# только ablation на Amazon
python run_ablation.py --n-trials 50

# пересобрать рисунки из JSON
python generate_comparison_figures.py
python generate_optuna_figures.py
```

Результаты: `results/experiment_results.json`, графики: `results/figures/`.

## Датасеты

CSV не включены в репозиторий — только ссылки и скрипт загрузки (`src/prepare_datasets.py`, см. также `data/README.md`).

| Ключ | Строк | Задача | Источник |
|------|-------|--------|----------|
| `amazon` | 32 769 | классификация (Logloss/AUC) | [OpenML Amazon_employee_access](https://www.openml.org/d/4135) |
| `students` | 1 044 | регрессия (RMSE) | [UCI Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |
| `ames` | 1 460 | регрессия (RMSE) | [OpenML house_prices](https://www.openml.org/d/42165) |

## Рисунки в отчёте

| Файл | Содержание |
|------|------------|
| `multi_01_improvements.png` | качество и latency, % |
| `multi_02_ctr_reduction.png` | сокращение CTR-таблиц |
| `optuna_03_feasible_region.png` | SLA-зона, Amazon |
| `optuna_students_03_feasible_region.png` | SLA-зона, Students |
| `optuna_ames_03_feasible_region.png` | SLA-зона, Ames |
| `optuna_01_history.png` | история Logloss, Amazon |
| `optuna_students_01_history.png` | история RMSE, Students |
| `optuna_ames_01_history.png` | история RMSE, Ames |

## Протокол

- `thread_count=2`, `random_seed=42`
- train/test 80/20, latency — медиана 50 прогонов + warmup
- SLA latency: не медленнее baseline (<10 ms) или −10% от baseline

## Автор

Баряев Андрей Алесеевич, P4135, ITMO.
