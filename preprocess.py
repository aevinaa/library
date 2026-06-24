import re
import pandas as pd

from config import READ_CHUNK_ROWS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens - shared by BM25 and fuzzy correction."""
    return _TOKEN_RE.findall((text or "").lower())


def clean_str(value) -> str:
    """Trimmed string; NaN / 'nan' / None all collapse to ''."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def extract_year(raw: str) -> int:
    """Pull a 4-digit year out of values like 'c2005.' or '[2010]'."""
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", raw or "")
    return int(m.group(1)) if m else 0


def build_corpus(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Stream the CSV and aggregate to one row per BibNum."""
    agg: dict[str, dict] = {}
    total = 0
    reader = pd.read_csv(
        csv_path, chunksize=READ_CHUNK_ROWS, dtype=str,
        keep_default_na=False, on_bad_lines="skip",
    )
    for chunk in reader:
        for _, row in chunk.iterrows():
            bib = clean_str(row.get("BibNum"))
            title = clean_str(row.get("Title"))
            if not bib or not title:
                continue
            try:
                copies = int(float(clean_str(row.get("ItemCount")) or 0))
            except ValueError:
                copies = 0
            loc = clean_str(row.get("ItemLocation"))

            if bib not in agg:
                if max_rows and total >= max_rows:
                    continue
                agg[bib] = {
                    "BibNum": bib,
                    "Title": title,
                    "Author": clean_str(row.get("Author")),
                    "Subjects": clean_str(row.get("Subjects")),
                    "Publisher": clean_str(row.get("Publisher")),
                    "PublicationYear": clean_str(row.get("PublicationYear")),
                    "Year": extract_year(clean_str(row.get("PublicationYear"))),
                    "ItemType": clean_str(row.get("ItemType")),
                    "Copies": 0,
                    "_locs": set(),
                }
                total += 1
            rec = agg.get(bib)
            if rec is not None:
                rec["Copies"] += copies
                if loc:
                    rec["_locs"].add(loc)

    rows = []
    for rec in agg.values():
        locs = sorted(rec.pop("_locs"))
        rec["Locations"] = "; ".join(locs)
        rec["NumLocations"] = len(locs)
        rec["text"] = " ".join(p for p in (
            rec["Title"], rec["Author"], rec["Subjects"], rec["ItemType"]
        ) if p)
        rows.append(rec)

    df = pd.DataFrame(rows).reset_index(drop=True)
    # compact dtypes - smaller Parquet, faster load
    df["Year"] = df["Year"].astype("int32")
    df["Copies"] = df["Copies"].astype("int32")
    df["NumLocations"] = df["NumLocations"].astype("int16")
    print(f"Corpus: {len(df):,} unique titles.")
    return df
