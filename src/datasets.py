"""Реестр датасетов для экспериментов: Amazon, Students, Ames Housing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TaskType = Literal["regression", "classification"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title_ru: str
    description: str
    target_col: str
    source: str
    task_type: TaskType = "regression"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "amazon": DatasetSpec(
        key="amazon",
        title_ru="Amazon Employee Access",
        description="Одобрение доступа сотрудника (ACTION), промышленный табличный датасет",
        target_col="ACTION",
        source="Kaggle / OpenML Amazon_employee_access",
        task_type="classification",
    ),
    "students": DatasetSpec(
        key="students",
        title_ru="Student Alcohol Consumption",
        description="Успеваемость школьников (G3), социально-демографические признаки",
        target_col="G3",
        source="UCI Student Performance",
        task_type="regression",
    ),
    "ames": DatasetSpec(
        key="ames",
        title_ru="Ames Housing",
        description="Цены на жильё в Эймсе, смешанные категориальные и числовые признаки",
        target_col="SalePrice",
        source="OpenML house_prices (De Cock, 2011)",
        task_type="regression",
    ),
}


def _prepare_splits(
    df: pd.DataFrame,
    target_col: str,
    task_type: TaskType = "regression",
    random_state: int = 42,
) -> dict[str, Any]:
    cat_features = [
        col for col in df.columns if df[col].dtype == "object" and col != target_col
    ]
    num_features = [
        col for col in df.columns if df[col].dtype != "object" and col != target_col
    ]

    for col in cat_features:
        df[col] = df[col].fillna("missing").astype(str)

    for col in num_features:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if task_type == "classification":
        y = df[target_col].astype(int)
    else:
        y = df[target_col].astype(float)
    X = df[cat_features + num_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    cardinality = {col: int(X[col].nunique()) for col in cat_features}

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_features": cat_features,
        "num_features": num_features,
        "cardinality": cardinality,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "target_col": target_col,
        "n_test": len(X_test),
        "task_type": task_type,
    }


def _load_amazon() -> dict[str, Any]:
    amazon_path = DATA_DIR / "amazon_employee_access.csv"
    if not amazon_path.exists():
        raise FileNotFoundError(
            f"Файл {amazon_path} не найден. Запустите: python prepare_datasets.py"
        )
    df = pd.read_csv(amazon_path)
    if "target" in df.columns and "ACTION" not in df.columns:
        df = df.rename(columns={"target": "ACTION"})
    spec = DATASET_SPECS["amazon"]
    for col in df.columns:
        if col != spec.target_col:
            df[col] = df[col].astype(str)
    data = _prepare_splits(df, spec.target_col, task_type=spec.task_type)
    data["name"] = "amazon"
    data["spec"] = spec
    return data


def _read_student_csv(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)


def _load_students() -> dict[str, Any]:
    mat_path = DATA_DIR / "student-mat.csv"
    por_path = DATA_DIR / "student-por.csv"
    if not mat_path.exists() or not por_path.exists():
        raise FileNotFoundError(
            f"Нужны {mat_path} и {por_path}. Запустите: python prepare_datasets.py"
        )
    mat = _read_student_csv(mat_path)
    por = _read_student_csv(por_path)
    mat["course"] = "mathematics"
    por["course"] = "portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    spec = DATASET_SPECS["students"]
    data = _prepare_splits(df, spec.target_col, task_type=spec.task_type)
    data["name"] = "students"
    data["spec"] = spec
    return data


def _load_ames() -> dict[str, Any]:
    ames_path = DATA_DIR / "ames_housing.csv"
    if not ames_path.exists():
        raise FileNotFoundError(
            f"Файл {ames_path} не найден. Запустите: python prepare_datasets.py"
        )
    df = pd.read_csv(ames_path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    spec = DATASET_SPECS["ames"]
    data = _prepare_splits(df, spec.target_col, task_type=spec.task_type)
    data["name"] = "ames"
    data["spec"] = spec
    return data


LOADERS = {
    "amazon": _load_amazon,
    "students": _load_students,
    "ames": _load_ames,
}


def list_datasets() -> list[str]:
    return list(LOADERS.keys())


def load_dataset(name: str = "amazon") -> dict[str, Any]:
    """Загружает датасет по ключу реестра."""
    if name not in LOADERS:
        raise ValueError(f"Неизвестный датасет: {name}. Доступны: {list_datasets()}")
    return LOADERS[name]()
