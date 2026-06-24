# Hybrid Library Search

A local, hybrid search system over a large library catalogue (built and tested
against the Seattle Public Library "Library Collection Inventory", ~2.5 million
rows). It combines semantic and keyword search, repairs typos, reranks results
with a cross-encoder, answers questions in natural language with a **local**
LLM, and ships with a Streamlit UI that includes filters, availability lookup,
source citations, confidence scores, and search analytics.

Everything runs on your own machine. No data leaves it, and no paid API is
required - the optional language model is a quantised GGUF that downloads and
runs locally.

---

## Table of contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Project structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Getting the dataset](#getting-the-dataset)
7. [Building the index](#building-the-index)
8. [Running the app](#running-the-app)
9. [One-command launch (auto-build)](#one-command-launch-auto-build)
10. [Integrating a local LLM](#integrating-a-local-llm)
11. [Configuration reference](#configuration-reference)
12. [Efficiency and scaling](#efficiency-and-scaling)
13. [Troubleshooting](#troubleshooting)
14. [Extending the project](#extending-the-project)

---

## What it does

| Feature | How it is implemented |
|---|---|
| Semantic search | sentence-transformers embeddings + FAISS vector index |
| Keyword search | BM25 via `bm25s` (fast, memory-mapped) |
| Hybrid search | Reciprocal Rank Fusion (RRF) of the dense and sparse rankings |
| Fuzzy / typo correction | `rapidfuzz` repairs query tokens against the corpus vocabulary |
| Cross-encoder reranking | a cross-encoder re-scores the fused candidates for precision |
| Multi-query expansion | the local LLM paraphrases your query; all variants are searched and fused |
| Metadata filtering | author, subject, item type, and year-range filters on the catalogue |
| Availability & location | copies and branches are aggregated per title during ingest |
| Source citations + confidence | every result carries a `[BibNum ...]` citation and a 0-1 confidence score |
| Conversational memory | prior turns are threaded into the answer prompt |
| Search history & analytics | every query is logged and summarised in an analytics tab |

Search, filtering, reranking, availability, citations, and analytics all work
**with no language model at all**. The LLM only adds conversational phrasing of
the answers and multi-query expansion, and it is opt-in.

---

## How it works

The system is split into an **offline indexing** stage and an **online query**
stage.

**Indexing (`build_index.py`), run once:**

1. `preprocess.build_corpus` streams the CSV in chunks so memory stays flat
   regardless of file size, cleans the messy fields, and collapses the many
   per-branch rows into **one aggregated record per title** (BibNum), summing
   total copies and collecting the set of branch locations.
2. A sparse **BM25** index is built over the title/author/subject text.
3. A **vocabulary** of tokens is precomputed for fuzzy correction (so the app
   never has to scan millions of titles at startup).
4. Each record is embedded into a vector; the vectors are cached to disk and
   loaded into a **FAISS** index (exact `flat` for small corpora, approximate
   `HNSW` for large ones - chosen automatically).

**Querying (`rag.py`), at runtime:**

1. The query is run through fuzzy **typo correction**.
2. It is searched against both the **dense** (FAISS) and **sparse** (BM25)
   indices; the two rankings are merged with **RRF**.
3. Optional **metadata filters** are applied.
4. The top candidates are **reranked** by the cross-encoder.
5. A **confidence** score is derived from the top result; below a threshold the
   answer is flagged as uncertain.
6. The results, with **citations**, are passed to the local LLM for a grounded
   answer (or returned as a clean list if no LLM is enabled).

---

## Project structure

```
RagAgent/
├── config.py          # all settings and tunables (override via env vars)
├── preprocess.py      # data layer: clean + aggregate the CSV
├── build_index.py     # offline indexer (dense + sparse + vocab)
├── rag.py             # runtime engine: hybrid search, rerank, fuzzy, answers
├── app.py             # Streamlit UI
├── requirements.txt   # dependencies
├── inventory.csv      # the catalogue (you download this)
└── index/             # generated artefacts (created by build_index.py)
    ├── corpus.parquet
    ├── bm25s/
    ├── vocab.pkl
    ├── embeddings.npy
    ├── dense.faiss
    └── history.csv
```

> **Note on `preprocess.py`:** if your Python environment already has an
> unrelated package called `preprocess`, it can shadow this file and cause an
> `ImportError`. If that happens, rename `preprocess.py` to `data_prep.py` and
> change the import lines in `build_index.py` and `rag.py` accordingly. See
> [Troubleshooting](#troubleshooting).

---

## Prerequisites

- **Python 3.10 or 3.11** (3.11 recommended).
- About **3 GB of disk** for a 50k-row slice, more for the full dataset, plus
  ~2 GB if you enable the local LLM.
- **Optional, for speed:** an NVIDIA GPU with a recent driver. The embedding
  step and the LLM both benefit substantially; FAISS does not need it.

---

## Installation

From the project folder:

```powershell
python -m pip install -r requirements.txt
```

Use `python -m pip` (not bare `pip`) so packages land in the exact interpreter
you will run the scripts with - this is the usual cause of "I installed it but
it says not found" on Windows.

> **If `llama-cpp-python` fails or hangs during install:** it compiles from
> source and can be fussy on Windows. Comment that line out in
> `requirements.txt` for now (put a `#` in front of it), finish the install, and
> add it back later when you want AI answers. Everything else works without it.

---

## Getting the dataset

Download the Seattle **"Library Collection Inventory"** CSV from
`data.seattle.gov` (search for that dataset; export as CSV), or use the Kaggle
mirror `city-of-seattle/seattle-library-collection-inventory`. Save it into the
project folder as `inventory.csv`.

The expected columns are:
`BibNum, Title, Author, ISBN, PublicationYear, Publisher, Subjects, ItemType,
ItemCollection, FloatingItem, ItemLocation, ReportDate, ItemCount`.

Any CSV with at least `BibNum` and `Title` will build; the other columns enrich
filtering and display.

---

## Building the index

Build a **small slice first** to confirm the whole pipeline runs end to end
before committing to the full multi-million-row build:

```powershell
python build_index.py --csv inventory.csv --max-rows 50000
```

You will see, in order: the corpus size, the vocabulary size, the embedding
progress bar, and finally a "Build complete" line. **The embedding model
(~90 MB) and cross-encoder download from Hugging Face on the first run**, so the
first build pauses silently for a minute or two before the progress bar appears
- that is normal, not a hang.

When it finishes, an `index/` folder will contain all the artefacts.

Once the slice works, build the **full dataset** by dropping the cap:

```powershell
python build_index.py --csv inventory.csv
```

On the full set the dense index switches to HNSW automatically, so queries stay
fast.

---

## Running the app

```powershell
streamlit run app.py
```

Use `streamlit run app.py`, **not** `python app.py` - Streamlit apps must be
launched through the `streamlit` command. It opens a browser tab at
`http://localhost:8501`. The first load spins while it loads the indices and the
embedding/rerank models (the cross-encoder downloads once here too), then the
search box appears.

The UI has three tabs:

- **Search** - a chat-style box with conversation memory, result cards showing
  score, citation, copies, and branches, plus a confidence/latency line.
- **Availability** - look up a specific title (typos tolerated) to see total
  copies and which branches hold it.
- **Analytics** - totals, average latency and confidence, top queries, a
  searches-over-time chart, and a recent-searches table.

Stop the app with **Ctrl-C** in the terminal.

> If you rebuild the index while the app is running, use the browser's "Rerun"
> and clear the cache from the top-right menu, since the engine is cached on
> load.

---

## One-command launch (auto-build)

If you would rather not run the build step yourself, point the app at the CSV
and it builds the index automatically on first launch:

```powershell
# PowerShell
$env:LIBRARY_CSV = "inventory.csv"
$env:BUILD_MAX_ROWS = "50000"      # optional: index a slice first
streamlit run app.py
```

The app detects the missing index, builds it with a progress spinner, then runs
normally. On later launches it simply loads the existing index.

---

## Integrating a local LLM

The conversational answers and multi-query expansion are powered by a **local**
language model run in-process through `llama-cpp-python` (which is `llama.cpp`
with Python bindings). There is no external API and no key. There are two ways
to provide the model.

### Option A - automatic download (default, easiest)

This is the path we use by default. When you enable AI answers, the app calls
`llama-cpp-python`'s `from_pretrained`, which downloads a quantised GGUF file
from Hugging Face and caches it locally. No manual download, no file paths.

1. Make sure `llama-cpp-python` installed successfully (see Installation).
2. Run the app and flip the **"AI answers"** toggle in the sidebar.
3. The first time, the model downloads (~2 GB) and is cached; subsequent runs
   load it instantly.

The default model is set in `config.py`:

```python
LLM_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
LLM_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

To use a **different** model from Hugging Face, override those two values with
environment variables before launching - point them at any GGUF repo and file:

```powershell
$env:LLM_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
$env:LLM_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
streamlit run app.py
```

The repo id is the part after `huggingface.co/`, and the filename is the exact
`.gguf` file from that repo's "Files" tab.

### Option B - use a model file you already have

If you have downloaded a GGUF yourself (for example via
`hf download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" --local-dir C:\models`),
point the app straight at the file. This overrides the auto-download:

```powershell
$env:LLM_GGUF = "C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
streamlit run app.py
```

### Choosing a model and quantisation

- **Size:** a 3B model at `Q4_K_M` (~2 GB) is a comfortable default for a
  laptop. Step up to a 7-8B model if you have the RAM/VRAM and want better
  answers; drop to the 1B variant if 3B feels heavy.
- **Quantisation:** `Q4_K_M` is the best size/quality balance and the default.
  `Q5_K_M` or `Q6_K` give slightly better quality for more memory.
- **Where to find them:** the `bartowski/*-GGUF` repos on Hugging Face are a
  reliable source of community GGUF builds across many base models.

### Running the model on the GPU

The default `pip install llama-cpp-python` is a **CPU-only** build. For GPU
offload (much faster generation), reinstall it with the CUDA backend compiled
in:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

This needs the Visual Studio C++ build tools and the CUDA toolkit present. Once
installed, the model offloads to the GPU automatically - the wrapper passes
`n_gpu_layers=-1`, which loads all layers onto the GPU when a CUDA build is in
use and is simply ignored on a CPU build. If the compile is troublesome, run the
CPU build first to confirm the app works, then optimise.

### How it is wired internally

In `rag.py`, the `LocalLLM` class wraps `llama_cpp.Llama` and exposes:

- `LocalLLM.from_repo(repo, filename)` - the auto-download path (Option A).
- `LocalLLM.from_path(path)` - load a local file (Option B).
- `chat(system, user)` - a single grounded completion.
- `expand_queries(query)` - generates paraphrases for multi-query search.

The `build_llm()` helper picks Option B if `LLM_GGUF` is set and the file
exists, otherwise falls back to Option A. The Streamlit app loads the model
lazily (only when the toggle is on) and caches it, so it is fetched at most once
per session.

---

## Configuration reference

Everything lives in `config.py` and most values can be overridden with
environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `LIBRARY_CSV` | (unset) | CSV path for the app to auto-build from on first launch |
| `BUILD_MAX_ROWS` | (unset) | cap rows indexed during auto-build |
| `LLM_REPO` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Hugging Face repo for the auto-download model |
| `LLM_FILE` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | GGUF filename within that repo |
| `LLM_GGUF` | (unset) | explicit local GGUF path; overrides the repo download |
| `EMBED_BATCH` | `256` | embedding batch size; raise to 512-1024 on a GPU |
| `SPARSE_BACKEND` | `bm25s` | `bm25s` (fast) or `rank_bm25` (pure-python fallback) |
| `INDEX_TYPE` | `auto` | `auto` / `flat` (exact) / `hnsw` (approximate, fast at scale) |

Other tunables in `config.py` (not env-driven by default): `RRF_K` (fusion
constant), `CANDIDATE_K` (candidates pulled before reranking), `FUZZY_MIN_SCORE`
(typo-correction threshold), `ABSTAIN_BELOW` (confidence floor), and the HNSW
graph parameters.

---

## Efficiency and scaling

- **Sparse search** uses `bm25s`, which is far faster than `rank-bm25` and loads
  its index memory-mapped, so it barely touches your heap. To fall back, set
  `SPARSE_BACKEND=rank_bm25` (and uncomment `rank-bm25` in `requirements.txt`).
- **Dense index type** is chosen automatically: exact `flat` for small corpora,
  approximate `HNSW` once you pass ~200k titles, which keeps queries fast into
  the millions while staying on CPU.
- **Embeddings are cached** to `index/embeddings.npy`. You can rebuild the FAISS
  structure (for example to force a different index type) **without
  re-embedding**:

  ```powershell
  $env:INDEX_TYPE = "hnsw"
  python build_index.py --reindex
  ```

- **The GPU helps the embedding step most**, not FAISS. Confirm
  `torch.cuda.is_available()` is `True` (install the CUDA build of PyTorch) and
  raise `EMBED_BATCH` to 512-1024. Installing `faiss-gpu` is rarely worth the
  trouble for a single-user app.

---

## Troubleshooting

**`error: unrecognized arguments: [--max-rows 50000]`**
Square brackets in instructions mean "optional" - do not type them. Run
`python build_index.py --csv inventory.csv --max-rows 50000`.

**`ModuleNotFoundError: No module named 'faiss'` (or `bm25s`, `rapidfuzz`, ...)**
A dependency is missing from the interpreter you are using. Install with
`python -m pip install -r requirements.txt` (note `python -m pip`).

**`ImportError: cannot import name 'build_corpus' from 'preprocess'`**
The import is resolving to a different `preprocess` package installed elsewhere
(check the path in the error). Fix by renaming your file to avoid the clash:
`ren preprocess.py data_prep.py`, then change `from preprocess import ...` to
`from data_prep import ...` in both `build_index.py` and `rag.py`.

**The build "just sits there" with no output**
It is almost always the first-run model download followed by the silent
embedding step. Give it a few minutes. To confirm progress, watch the artefacts
appear with `dir index` in a second terminal, and/or note that the embedding
progress bar prints once the model has loaded. Test with `--max-rows 2000` for a
near-instant run.

**It prints `on cpu` and is slow**
Your PyTorch is the CPU-only build. Install the CUDA build from
`pytorch.org/get-started/locally` into the same interpreter, then verify with
`python -c "import torch; print(torch.cuda.is_available())"`.

**`streamlit` is not recognised**
Install it (`python -m pip install streamlit`) or launch via
`python -m streamlit run app.py`.

**`llama-cpp-python` will not install / compile**
Comment it out in `requirements.txt` and use the app without AI answers, or
install a prebuilt wheel. The CUDA build additionally needs Visual Studio C++
build tools and the CUDA toolkit.

**`huggingface-cli` says it is deprecated**
Use `hf` instead - same arguments, e.g.
`hf download <repo> --include "<file>.gguf" --local-dir C:\models`.

---

## Extending the project

- **Evaluation harness** (`eval.py`): measure recall@k and MRR on a set of
  known query→title pairs to prove retrieval quality, not just speed.
- **Query router**: send exact/aggregate questions ("how many copies of X",
  "everything at branch Y") to a direct pandas/SQL lookup, and only the
  open-ended discovery questions through the hybrid retriever - this fixes the
  one class of question semantic search answers badly.
- **Incremental re-indexing**: use the dataset's `ReportDate` to ingest updates
  without a full rebuild.
- **Faster reranking**: lower `CANDIDATE_K` or use an ONNX-quantised
  cross-encoder if query latency matters.
