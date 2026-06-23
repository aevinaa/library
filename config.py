"""
config.py — Single source of truth for all project settings.
Both app.py and rag.py import from here. Never hardcode paths elsewhere.
"""

from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.resolve()

# ── Data Paths ────────────────────────────────────────────────────────────────

DATA_DIR            = ROOT_DIR / "data"
RAW_CSV_PATH        = DATA_DIR / "seattle_library.csv"   # Original Kaggle dataset
PROCESSED_CSV_PATH  = DATA_DIR / "books_clean.csv"       # Cleaned by preprocess.py

# ── Index Paths (built by build_index.py, gitignored) ─────────────────────────

FAISS_INDEX_PATH    = DATA_DIR / "faiss_index.index"
BM25_INDEX_PATH     = DATA_DIR / "bm25_corpus.pkl"

# ── Model Paths ───────────────────────────────────────────────────────────────

MODELS_DIR          = ROOT_DIR / "models"
LLM_MODEL_PATH      = MODELS_DIR / "phi-3-mini.gguf"    # Must match filename in /models

# ── LLM Settings ─────────────────────────────────────────────────────────────

LLM_CONTEXT_WINDOW  = 4096    # Max tokens in the model's context
LLM_MAX_TOKENS      = 512     # Max tokens to generate per response
LLM_TEMPERATURE     = 0.2     # Low = focused/factual; raise for creative answers
LLM_TOP_P           = 0.9
LLM_N_THREADS       = 8       # CPU threads for inference; match your core count
LLM_N_GPU_LAYERS    = 0       # Set > 0 to offload layers to GPU (requires CUDA build)

# ── Embedding Settings ────────────────────────────────────────────────────────

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"   # Fast, lightweight Sentence Transformer
EMBEDDING_DIM       = 384                   # Must match the model above
EMBEDDING_BATCH_SIZE = 64                   # Rows processed per batch during indexing

# ── Retrieval Parameters ──────────────────────────────────────────────────────

# How many candidates each retriever fetches before fusion
TOP_K_VECTOR        = 20
TOP_K_BM25          = 20

# Final number of results returned to the LLM after reranking
TOP_K_FINAL         = 5

# Reciprocal Rank Fusion constant (higher = less aggressive rank weighting)
RRF_K               = 60

# Fuzzy search: minimum similarity score to accept a typo-corrected match (0–100)
FUZZY_THRESHOLD     = 75

# Cross-encoder model for reranking
RERANKER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Multi-Query Expansion ─────────────────────────────────────────────────────

QUERY_EXPANSION_COUNT = 2    # Number of alternative queries the LLM generates

# ── Conversational Memory ─────────────────────────────────────────────────────

MEMORY_WINDOW       = 6      # Number of past (user, assistant) turns to keep

# ── Confidence Score Thresholds ───────────────────────────────────────────────

CONFIDENCE_HIGH     = 0.75   # >= this → "High confidence"
CONFIDENCE_MID      = 0.45   # >= this → "Moderate confidence"
                             #  < this  → "Low confidence"

# ── Analytics / Query Log ─────────────────────────────────────────────────────

QUERY_LOG_PATH      = DATA_DIR / "query_log.json"

# ── CSV Columns of Interest ───────────────────────────────────────────────────
# These are the column names used from the Seattle Library dataset.
# Update if Kaggle column names differ after preprocessing.

COL_TITLE           = "Title"
COL_AUTHOR          = "Author"
COL_SUBJECT         = "Subjects"
COL_ITEM_TYPE       = "ItemType"
COL_ITEM_LOCATION   = "ItemLocation"
COL_CALL_NUMBER     = "CallNumber"
COL_REPORT_DATE     = "ReportDate"
COL_AVAILABLE       = "ItemCount"

# Columns combined into a single text chunk for embedding
EMBED_COLUMNS       = [COL_TITLE, COL_AUTHOR, COL_SUBJECT, COL_ITEM_TYPE]