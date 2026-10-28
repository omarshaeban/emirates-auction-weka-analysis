"""Produce a WEKA-ready ARFF file (with a nominal class as the final column)."""
import argparse
from pathlib import Path
import pandas as pd

FIELDS = ("vehicle_brand", "bids", "remaining_minutes", "high_price")


def q(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


def to_arff(frame: pd.DataFrame) -> str:
    if not set(FIELDS).issubset(frame.columns):
        raise ValueError("Missing expected modeling columns")
    brands = sorted({str(x) for x in frame.vehicle_brand.dropna()})
    if not brands:
        raise ValueError("No vehicle brands")
    header = ["@relation auction_high_price", "", "@attribute vehicle_brand {" + ",".join(q(b) for b in brands) + "}",
              "@attribute bids numeric", "@attribute remaining_minutes numeric",
              "@attribute high_price {low,high}", "", "@data"]
    for row in frame.loc[:,FIELDS].itertuples(index=False, name=None):
        vals = ["?" if pd.isna(x) else (q(x) if i==0 else str(x)) for i,x in enumerate(row)]
        header.append(",".join(vals))
    return "\n".join(header) + "\n"


def main():
    a=argparse.ArgumentParser();a.add_argument("input");a.add_argument("--output",default="model_input.arff");args=a.parse_args()
    frame=pd.read_csv(args.input)
    Path(args.output).write_text(to_arff(frame),encoding="utf-8")
    print(f"Wrote {len(frame)} instances to {args.output}")


if __name__ == '__main__':main()
