#!/usr/bin/env python3
"""Скачивает и подготавливает датасеты для экспериментов."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

STUDENT_URLS = {
    "student-mat.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv",
    "student-por.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-por.csv",
}


def download_students() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in STUDENT_URLS.items():
        out = DATA_DIR / name
        if out.exists():
            print(f"Already exists: {out}")
            continue
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, out)
        print(f"Saved -> {out}")


def download_ames() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "ames_housing.csv"
    if out.exists():
        print(f"Ames already exists: {out}")
        return

    from sklearn.datasets import fetch_openml

    print("Downloading Ames Housing from OpenML...")
    bunch = fetch_openml(name="house_prices", version=1, as_frame=True, parser="auto")
    bunch.frame.to_csv(out, index=False)
    print(f"Saved {len(bunch.frame)} rows -> {out}")


def download_amazon() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "amazon_employee_access.csv"
    if out.exists():
        print(f"Amazon already exists: {out}")
        return

    from sklearn.datasets import fetch_openml

    print("Downloading Amazon Employee Access from OpenML...")
    bunch = fetch_openml(name="Amazon_employee_access", version=1, as_frame=True, parser="auto")
    df = bunch.frame.rename(columns={"target": "ACTION"})
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} cols -> {out}")


def main() -> None:
    download_students()
    download_ames()
    download_amazon()
    print("Datasets ready in", DATA_DIR)


if __name__ == "__main__":
    main()
