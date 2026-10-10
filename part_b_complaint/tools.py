import sys
from pathlib import Path

import pandas as pd
import requests

from .app.schemas import CATEGORIES

DATA = Path(__file__).parent / "data"
COMPLAINTS, LABELED = DATA / "complaints.csv", DATA / "labeled_sample.csv"
N_COMPLAINTS, LABEL_SAMPLE, SEED = 300, 50, 42
DATE_MIN, DATE_MAX = "2024-01-01", "2024-03-31"


def fetch():
    """CFPB Consumer Complaint Database API (GET .../search/api/v1/), no key needed."""
    url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    # Oldest-first from DATE_MIN: even if the API ignores date_received_max, the first
    # 300 rows are still early 2024, so the slice stays inside the date window.
    params = {"has_narrative": "true", "date_received_min": DATE_MIN, "date_received_max": DATE_MAX,
              "size": N_COMPLAINTS, "sort": "created_date_asc", "no_aggs": "true", "field": "all"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    rows, skipped_blank, skipped_date = [], 0, 0
    for h in hits:
        src = h.get("_source", {})
        narrative = (src.get("complaint_what_happened") or "").strip()
        day = str(src.get("date_received", ""))[:10]
        if not narrative:
            skipped_blank += 1
        elif day and not (DATE_MIN <= day <= DATE_MAX):
            skipped_date += 1          # outside the slice
        else:
            rows.append({"id": src.get("complaint_id"), "date_received": day,
                         "product": src.get("product"), "narrative": narrative})
    print(f"API returned {len(hits)} hits | kept {len(rows)} | "
          f"blank narrative {skipped_blank} | outside date range {skipped_date}")
    if not rows:
        if hits and skipped_blank == len(hits) and "complaint_what_happened" not in hits[0].get("_source", {}):
            print("The API no longer returns narrative text (the CFPB has been removing it).")
            print("Use a copy of the dataset from before the removal:  python -m part_b_complaint.tools fromcsv PATH")
        if hits:
            print("first hit keys:", sorted(hits[0].get("_source", {}).keys()))
            print("first hit date_received:", hits[0].get("_source", {}).get("date_received"))
        sys.exit("No usable complaints. See the counts above; check params against the API docs.")
    df = pd.DataFrame(rows).drop_duplicates("id").sort_values("id")
    DATA.mkdir(exist_ok=True)
    df.to_csv(COMPLAINTS, index=False)
    print(f"saved {len(df)} complaints -> {COMPLAINTS}")


def fromcsv():
    """Build data/complaints.csv from a CFPB-format CSV (or .zip of one), read in chunks so
    multi-GB files do not run out of memory."""
    if len(sys.argv) < 3:
        sys.exit("usage: python -m part_b_complaint.tools fromcsv PATH_TO_CSV")
    path = sys.argv[2]
    head = pd.read_csv(path, dtype=str, nrows=5)
    cols = {c.strip().lower(): c for c in head.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        sys.exit(f"column not found, tried {names}; file has: {list(head.columns)}")

    idc, datec = col("complaint id", "complaint_id"), col("date received", "date_received")
    prodc = col("product")
    narc = col("consumer complaint narrative", "complaint_what_happened", "narrative",
               "consumer_complaint_narrative")
    lo, hi = pd.Timestamp(DATE_MIN), pd.Timestamp(DATE_MAX) + pd.Timedelta(days=1)
    kept, total = [], 0
    for chunk in pd.read_csv(path, dtype=str, usecols=[idc, datec, prodc, narc], chunksize=200_000):
        total += len(chunk)
        d = pd.to_datetime(chunk[datec], errors="coerce", utc=True).dt.tz_localize(None)
        n = chunk[narc].fillna("").str.strip()
        m = (n.str.len() > 0) & (d >= lo) & (d < hi)
        if m.any():
            kept.append(pd.DataFrame({"id": chunk.loc[m, idc], "date_received": d[m].dt.strftime("%Y-%m-%d"),
                                      "product": chunk.loc[m, prodc], "narrative": n[m]}))
    df = pd.concat(kept) if kept else pd.DataFrame()
    print(f"{total} rows read | {len(df)} have a narrative within {DATE_MIN}..{DATE_MAX}")
    if df.empty:
        sys.exit("Nothing in that slice. Check the date range and that the narrative column has text.")
    df = df.drop_duplicates("id")
    print("products in slice (before sampling):")
    print(df["product"].value_counts().head(10).to_string())
    if len(df) > N_COMPLAINTS:
        df = stratified_sample(df, N_COMPLAINTS)
    df = df.sort_values("id")
    DATA.mkdir(exist_ok=True)
    df.to_csv(COMPLAINTS, index=False)
    print("\nproducts in saved sample:")
    print(df["product"].value_counts().to_string())
    print(f"saved {len(df)} complaints -> {COMPLAINTS}")


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Balance across products so credit reporting (the bulk of CFPB data) does not swamp
    everything else. Equal share per product, small products give all they have and the
    remainder is redistributed. Seeded, so reproducible."""
    groups = {k: g.sample(frac=1, random_state=SEED) for k, g in df.groupby("product")}
    take = {k: 0 for k in groups}
    remaining = n
    while remaining > 0:
        open_ = [k for k in groups if take[k] < len(groups[k])]
        if not open_:
            break
        share = max(remaining // len(open_), 1)
        for k in open_:
            add = min(share, len(groups[k]) - take[k], remaining)
            take[k] += add
            remaining -= add
            if remaining == 0:
                break
    return pd.concat([groups[k].head(take[k]) for k in groups])


def labelsheet():
    """Fill category/urgency/requires_human BEFORE looking at any model output."""
    if LABELED.exists():
        sys.exit(f"{LABELED} exists; refusing to overwrite your hand labels.")
    df = pd.read_csv(COMPLAINTS).sample(LABEL_SAMPLE, random_state=SEED)
    sheet = df[["id", "narrative"]].copy()
    sheet["category"] = sheet["urgency"] = sheet["requires_human"] = ""
    sheet.to_csv(LABELED, index=False)
    print(f"wrote {LABELED}\ncategories: {CATEGORIES}\nurgency: low|medium|high  requires_human: True|False")


if __name__ == "__main__":
    {"fetch": fetch, "fromcsv": fromcsv, "labelsheet": labelsheet}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()