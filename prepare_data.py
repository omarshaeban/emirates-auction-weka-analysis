"""Prepare extracted vehicle listings for Python and WEKA model comparisons.

The original assignment did not define a prediction target. The optional
high_price target here is a new *demonstration choice*, not a recovered result.
"""
import argparse
import re
from pathlib import Path
import pandas as pd


def duration_minutes(value: str) -> float:
    if not isinstance(value, str) or not value.strip():
        return float("nan")
    t = value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    # Arabic / English units. Ambiguous countdown strings become missing.
    pieces = re.findall(r"(\d+)\s*(يوم|أيام|يومان|day(?:s)?|ساعة|ساعات|hour(?:s)?|دقيقة|دقائق|minute(?:s)?)", t, re.I)
    units = {"يوم": 1440, "أيام": 1440, "يومان": 1440, "day": 1440, "days": 1440,
             "ساعة": 60, "ساعات": 60, "hour": 60, "hours": 60,
             "دقيقة": 1, "دقائق": 1, "minute": 1, "minutes": 1}
    if pieces:
        return float(sum(int(n) * units[unit.lower()] for n,unit in pieces))
    # HH:MM:SS countdown, or H:MM
    match = re.fullmatch(r"\s*(\d{1,3}):(\d{2})(?::(\d{2}))?\s*", t)
    if match:
        a,b,c = match.groups()
        return float(int(a)*60 + int(b) + int(c or 0)/60) if c is not None else float(int(a)*60+int(b))
    return float("nan")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {"title", "sequence", "time_left", "bids", "price"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    data = df.copy().drop_duplicates(subset=["sequence", "title"])
    for col in ("bids", "price"):
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    data["remaining_minutes"] = data["time_left"].fillna("").map(duration_minutes)
    data["vehicle_brand"] = data["title"].fillna("").astype(str).str.strip().str.split().str[0].replace("", "Unknown")
    data = data.loc[data["price"].notna() & (data["price"] > 0)].copy()
    if not len(data):
        raise ValueError("No valid priced listings; cannot derive target.")
    threshold = data["price"].median()
    if data["price"].nunique() < 2:
        raise ValueError("All valid prices are identical; no distinct high_price classes.")
    data["high_price"] = data["price"].ge(threshold).map({True:"high",False:"low"})
    if data["high_price"].nunique() != 2:
        raise ValueError("Derived target contains only one class; more diverse listings are needed.")
    # Exclude price (label construction) and sequence (identifier): prevent target leakage.
    return data[["vehicle_brand", "bids", "remaining_minutes", "high_price"]].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Semicolon-separated original CSV")
    p.add_argument("--output", default="model_input.csv")
    args = p.parse_args()
    df = pd.read_csv(args.input, sep=";", encoding="utf-8-sig")
    cleaned = prepare(df)
    cleaned.to_csv(args.output,index=False,encoding="utf-8-sig")
    print(f"Prepared {len(cleaned)} records, target counts: {cleaned.high_price.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
