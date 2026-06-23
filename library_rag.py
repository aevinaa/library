"""
Scalable RAG for large library catalogues
==========================================

Built for the Seattle Public Library "Library Collection Inventory" dataset
(~2.5 million rows), but works on any CSV with similar columns.

Download the CSV from:
    https://data.seattle.gov/Community/Library-Collection-Inventory/6vkj-f5xf
    (or Kaggle: city-of-seattle/seattle-library-collection-inventory)

Why this differs from the toy version:
  * The file is far too big to load into a single DataFrame, so it is read
    in bounded chunks (streaming) and memory stays flat regardless of size.
  * Embedding millions of texts is the slow part, so encoding is batched and
    runs on GPU automatically if torch sees one.
  * The same title (BibNum) appears in many branches; we index ONE document
    per unique title instead of embedding the same book dozens of times.
  * Indexing is resumable: re-running skips items already in the store, so a
    crash at row 800k does not mean starting over.
  * A --max-rows cap lets you smoke-test on a few thousand rows first.

Install:
    pip install pandas chromadb sentence-transformers anthropic torch

Typical usage:
    # test on a slice first
    python library_rag_seattle.py --csv inventory.csv --max-rows 5000
    # full build (slow; leave it running)
    python library_rag_seattle.py --csv inventory.csv
    # one-shot question
    python library_rag_seattle.py --csv inventory.csv --query "books on machine learning"
"""

import os
import re
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
COLLECTION_NAME = "seattle_library"
PERSIST_DIR = "./chroma_seattle"
MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for catalogues

READ_CHUNK_ROWS = 50_000   # rows pulled into RAM per pass (bounds memory)
EMBED_BATCH = 1024          # texts per embedding forward-pass
ADD_BATCH = 5_000          # rows per Chroma write (stays under its hard cap)
TOP_K = 5

REQUIRED_COLUMNS = ["BibNum", "Title"]
META_FIELDS = ["Title", "Author", "ItemType", "ItemCollection",
               "ItemLocation", "Publisher", "ISBN", "Subjects"]


# --------------------------------------------------------------------------- #
# Cleaning helpers - real data is messy, so coerce everything defensively
# --------------------------------------------------------------------------- #
def clean_str(value) -> str:
    """Return a trimmed string, mapping NaN / 'nan' / None to ''."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def extract_year(raw: str):
    """Pull a 4-digit year out of messy values like 'c2005.' or '[2010]'."""
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", raw or "")
    return int(m.group(1)) if m else 0


def row_to_document(row) -> str:
    """The text that actually gets embedded. Subjects carry most signal."""
    title = clean_str(row.get("Title"))
    author = clean_str(row.get("Author"))
    subjects = clean_str(row.get("Subjects"))
    publisher = clean_str(row.get("Publisher"))
    year = clean_str(row.get("PublicationYear"))
    itemtype = clean_str(row.get("ItemType"))

    parts = [f"Title: {title}."]
    if author:
        parts.append(f"Author: {author}.")
    if subjects:
        parts.append(f"Subjects: {subjects}.")
    if itemtype:
        parts.append(f"Item type: {itemtype}.")
    year = year.rstrip(". ")
    pub = " ".join(p for p in (publisher, year) if p)
    if pub:
        parts.append(f"Published: {pub}.")
    return " ".join(parts)


def row_to_metadata(row) -> dict:
    """Structured fields kept beside each vector for filtering and display.

    Chroma metadata must be str/int/float/bool and never None, so we coerce.
    """
    meta = {f: clean_str(row.get(f))[:500] for f in META_FIELDS}
    meta["Year"] = extract_year(clean_str(row.get("PublicationYear")))
    try:
        meta["ItemCount"] = int(float(clean_str(row.get("ItemCount")) or 0))
    except ValueError:
        meta["ItemCount"] = 0
    return meta


# --------------------------------------------------------------------------- #
# Streaming reader: yields cleaned, de-duplicated batches
# --------------------------------------------------------------------------- #
def validate_columns(csv_path: str):
    header = pd.read_csv(csv_path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        sys.exit(f"CSV is missing required column(s): {missing}")
    expected = set(META_FIELDS + ["PublicationYear", "ItemCount"])
    absent = expected - set(header.columns)
    if absent:
        print(f"Note: optional columns absent, will be blank: {sorted(absent)}")


def stream_records(csv_path: str, max_rows=None):
    """Yield (ids, documents, metadatas) per chunk, one doc per unique title."""
    seen = set()      # BibNums already emitted this run (cross-chunk dedup)
    total = 0
    reader = pd.read_csv(
        csv_path,
        chunksize=READ_CHUNK_ROWS,
        dtype=str,                 # read raw, coerce ourselves
        keep_default_na=False,     # empty cells become '' not NaN
        on_bad_lines="skip",
    )
    for chunk in reader:
        chunk = chunk.drop_duplicates(subset=["BibNum"])   # within-chunk dedup
        ids, docs, metas = [], [], []
        for _, row in chunk.iterrows():
            bib = clean_str(row.get("BibNum"))
            title = clean_str(row.get("Title"))
            if not bib or not title or bib in seen:
                continue
            seen.add(bib)
            ids.append(bib)
            docs.append(row_to_document(row))
            metas.append(row_to_metadata(row))
            total += 1
            if max_rows and total >= max_rows:
                break
        if ids:
            yield ids, docs, metas
        if max_rows and total >= max_rows:
            break


# --------------------------------------------------------------------------- #
# Index construction
# --------------------------------------------------------------------------- #
def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_index(csv_path: str, max_rows=None, rebuild=False):
    import chromadb
    from sentence_transformers import SentenceTransformer

    validate_columns(csv_path)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    device = detect_device()
    print(f"Loading '{MODEL_NAME}' on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    indexed, skipped = 0, 0
    for ids, docs, metas in stream_records(csv_path, max_rows=max_rows):
        # Resume support: drop ids already committed in a previous run.
        existing = set()
        for s in range(0, len(ids), 500):
            existing.update(collection.get(ids=ids[s:s + 500])["ids"])
        if existing:
            keep = [i for i, _id in enumerate(ids) if _id not in existing]
            ids = [ids[i] for i in keep]
            docs = [docs[i] for i in keep]
            metas = [metas[i] for i in keep]
            skipped += len(existing)
        if not ids:
            continue

        embeddings = model.encode(
            docs,
            batch_size=EMBED_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,    # pairs with cosine space
            show_progress_bar=False,
        ).tolist()

        for s in range(0, len(ids), ADD_BATCH):
            e = s + ADD_BATCH
            collection.add(
                ids=ids[s:e],
                documents=docs[s:e],
                metadatas=metas[s:e],
                embeddings=embeddings[s:e],
            )
        indexed += len(ids)
        print(f"  indexed {indexed:,}  (skipped {skipped:,} existing)", end="\r")

    print(f"\nDone. Collection holds {collection.count():,} unique titles.")
    return collection, model


# --------------------------------------------------------------------------- #
# Retrieval + generation
# --------------------------------------------------------------------------- #
def retrieve(collection, model, query: str, k: int = TOP_K, where=None):
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k, where=where)
    return res["documents"][0], res["metadatas"][0]


def answer(query: str, docs, metas) -> str:
    lines = [
        f"- {d} (Location: {m.get('ItemLocation', '')}, "
        f"Copies: {m.get('ItemCount', '')})"
        for d, m in zip(docs, metas)
    ]
    context_block = "\n".join(lines)

    if not os.getenv("ANTHROPIC_API_KEY"):
        return "[No ANTHROPIC_API_KEY set - retrieved context only]\n" + context_block

    from anthropic import Anthropic
    client = Anthropic()
    prompt = (
        "You are a library assistant. Answer using ONLY the catalogue entries "
        "below. If the answer is not present, say you could not find it.\n\n"
        f"Catalogue entries:\n{context_block}\n\nQuestion: {query}"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Scalable library RAG")
    ap.add_argument("--csv", required=True, help="path to the inventory CSV")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows indexed (use a small value to test first)")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the existing index and rebuild from scratch")
    ap.add_argument("--query", default=None,
                    help="run one query and exit (otherwise interactive)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}")

    collection, model = build_index(args.csv, args.max_rows, args.rebuild)

    if args.query:
        docs, metas = retrieve(collection, model, args.query)
        print("\n" + answer(args.query, docs, metas))
        return

    print("\nReady. Ask a question (Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        docs, metas = retrieve(collection, model, q)
        print("\n" + answer(q, docs, metas))


if __name__ == "__main__":
    main()