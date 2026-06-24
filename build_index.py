import os
import pickle
import argparse

import numpy as np

import config
from preprocess import build_corpus, tokenize


def make_index(embeddings: np.ndarray):
    import faiss
    dim = embeddings.shape[1]
    itype = config.resolve_index_type(embeddings.shape[0])
    if itype == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = config.HNSW_EF_SEARCH
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, itype


def build_vocab(corpus):
    vocab = set()
    for col in ("Title", "Author"):
        for t in corpus[col].tolist():
            vocab.update(tokenize(t))
    with open(config.VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary: {len(vocab):,} tokens.")


def build(csv_path: str, max_rows: int | None = None):
    """Full build. Importable so the app can auto-build on first launch."""
    import faiss
    from rag import Embedder, SparseIndex
    os.makedirs(config.ARTEFACT_DIR, exist_ok=True)

    corpus = build_corpus(csv_path, max_rows=max_rows)
    corpus.to_parquet(config.CORPUS_PARQUET, index=False)

    print(f"Building sparse index ({config.SPARSE_BACKEND}) ...")
    SparseIndex.build(corpus["text"].tolist()).save()
    build_vocab(corpus)

    embedder = Embedder()
    embs = embedder.encode(corpus["text"].tolist()).astype("float32")
    np.save(config.EMB_CACHE, embs)
    index, itype = make_index(embs)
    faiss.write_index(index, config.FAISS_PATH)
    print(f"Dense index: {index.ntotal:,} vectors, dim {embs.shape[1]}, type '{itype}'.")
    print("Build complete ->", config.ARTEFACT_DIR)


def reindex():
    import faiss
    if not os.path.exists(config.EMB_CACHE):
        raise SystemExit("No cached embeddings; run a full build first.")
    embs = np.load(config.EMB_CACHE)
    index, itype = make_index(embs)
    faiss.write_index(index, config.FAISS_PATH)
    print(f"Re-indexed {index.ntotal:,} vectors as '{itype}'.")


def main():
    ap = argparse.ArgumentParser(description="Build the hybrid library index")
    ap.add_argument("--csv")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--reindex", action="store_true")
    args = ap.parse_args()
    if args.reindex:
        reindex()
    elif args.csv:
        build(args.csv, max_rows=args.max_rows)
    else:
        raise SystemExit("Pass --csv to build, or --reindex to reshape the index.")


if __name__ == "__main__":
    main()
