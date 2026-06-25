"""Define the target and the FROZEN train/test split.

- Parse the price string -> float (EUR).
- Drop rows with no price; keep a sane price range.
- Target y = log1p(price).
- Split 80/20 GROUPED BY host_id (so one host's listings never straddle the
  split -> no host leakage). Deterministic via a fixed seed.

`load_split()` returns (train_df, test_df) and is the single source of truth the
feature/model code imports — every model reuses the exact same split.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "listings.csv.gz"

PRICE_MIN, PRICE_MAX = 10.0, 1000.0
TEST_SIZE = 0.20
SEED = 42


def parse_price(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"[$,]", "", regex=True)
        .replace("", pd.NA)
        .astype("float")
    )


def load_prepared() -> pd.DataFrame:
    df = pd.read_csv(RAW, compression="gzip", low_memory=False)
    n0 = len(df)
    df["price_eur"] = parse_price(df["price"])

    df = df[df["price_eur"].notna()]
    n_price = len(df)
    df = df[(df["price_eur"] >= PRICE_MIN) & (df["price_eur"] <= PRICE_MAX)]
    n_keep = len(df)

    df = df.copy()
    df["y_log"] = np.log1p(df["price_eur"])

    print(
        f"[prepare] {n0} rows -> {n_price} with price "
        f"({n0 - n_price} dropped null) -> {n_keep} in [{PRICE_MIN:.0f},{PRICE_MAX:.0f}] "
        f"({n_price - n_keep} trimmed as outliers)"
    )
    return df.reset_index(drop=True)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_prepared()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["host_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(
        f"[split] train {len(train_df)} | test {len(test_df)} "
        f"(grouped by host_id; "
        f"{len(set(train_df['host_id']) & set(test_df['host_id']))} hosts overlap -> must be 0)"
    )
    return train_df, test_df


if __name__ == "__main__":
    tr, te = load_split()
    print(tr[["price_eur", "y_log"]].describe().round(2).to_string())
