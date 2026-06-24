"""
config.py
=========
Central configuration. Sensible defaults so the system runs with zero tuning;
override any value with an environment variable where noted.
"""

import os

# ── Artefact paths ────────────────────────────────────────────────────────────
ARTEFACT_DIR = "./index"
CORPUS_PARQUET = os.path.join(ARTEFACT_DIR, "corpus.parquet")
FAISS_PATH = os.path.join(ARTEFACT_DIR, "dense.faiss")
EMB_CACHE = os.path.join(ARTEFACT_DIR, "embeddings.npy")
BM25_DIR = os.path.join(ARTEFACT_DIR, "bm25s")        # bm25s saves a directory
BM25_PATH = os.path.join(ARTEFACT_DIR, "bm25.pkl")    # rank_bm25 fallback pickle
VOCAB_PATH = os.path.join(ARTEFACT_DIR, "vocab.pkl")
HISTORY_CSV = os.path.join(ARTEFACT_DIR, "history.csv")

# ── Models ──────────────────────────────────────────────────────────────────--
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Local LLM: auto-downloaded from Hugging Face on first use - no manual step.
LLM_REPO = os.environ.get("LLM_REPO", "bartowski/Llama-3.2-3B-Instruct-GGUF")
LLM_FILE = os.environ.get("LLM_FILE", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
DEFAULT_LLM_PATH = os.environ.get("LLM_GGUF", "")     # explicit path overrides repo

# Optional: set LIBRARY_CSV so the app can auto-build the index on first launch.
LIBRARY_CSV = os.environ.get("LIBRARY_CSV", "")
BUILD_MAX_ROWS = int(os.environ.get("BUILD_MAX_ROWS", "0")) or None

# ── Ingestion / indexing ──────────────────────────────────────────────────────
READ_CHUNK_ROWS = 50_000
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "256"))   # raise to 512/1024 on GPU
SHOW_PROGRESS = True

# Sparse backend: "bm25s" (fast, memory-mapped) or "rank_bm25" (pure-python).
SPARSE_BACKEND = os.environ.get("SPARSE_BACKEND", "bm25s")

# Dense index: "auto" picks flat for small corpora, HNSW for large ones.
INDEX_TYPE = os.environ.get("INDEX_TYPE", "auto")     # "auto" | "flat" | "hnsw"
AUTO_HNSW_THRESHOLD = 200_000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# ── Retrieval / ranking ───────────────────────────────────────────────────────
RRF_K = 60
CANDIDATE_K = 50
FUZZY_MIN_SCORE = 82
ABSTAIN_BELOW = 0.30


def resolve_index_type(n_vectors: int) -> str:
    """Decide the concrete FAISS index type given the corpus size."""
    if INDEX_TYPE == "auto":
        return "hnsw" if n_vectors >= AUTO_HNSW_THRESHOLD else "flat"
    return INDEX_TYPE
