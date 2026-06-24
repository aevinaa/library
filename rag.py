"""
rag.py
======
Runtime engine: hybrid retrieval (dense FAISS + sparse BM25 via a swappable
backend, fused with RRF), fuzzy correction, cross-encoder reranking, optional
LLM multi-query expansion and grounded answers, metadata filtering,
availability lookup, confidence/citations, and history logging.

Also holds the lazy model wrappers (Embedder, Reranker, LocalLLM, SparseIndex)
so build_index.py can reuse them.
"""

from __future__ import annotations

import os
import csv
import pickle
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from preprocess import tokenize


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _import_bm25s():
    try:
        return __import__("bm25s")
    except ImportError:
        return None


# ── sparse backend (bm25s preferred, rank_bm25 fallback) ──────────────────────
class SparseIndex:
    def __init__(self, backend: str, obj, n_docs: int):
        self.backend = backend
        self.obj = obj
        self.n_docs = n_docs

    @classmethod
    def build(cls, texts: list[str]) -> "SparseIndex":
        n = len(texts)
        if config.SPARSE_BACKEND == "bm25s":
            bm25s = _import_bm25s()
            if bm25s is None:
                from rank_bm25 import BM25Okapi
                return cls("rank_bm25", BM25Okapi([tokenize(t) for t in texts]), n)
            r = bm25s.BM25()
            r.index(bm25s.tokenize(texts, stopwords="en", show_progress=False),
                    show_progress=False)
            return cls("bm25s", r, n)
        from rank_bm25 import BM25Okapi
        return cls("rank_bm25", BM25Okapi([tokenize(t) for t in texts]), n)

    def save(self):
        if self.backend == "bm25s":
            self.obj.save(config.BM25_DIR)
        else:
            with open(config.BM25_PATH, "wb") as f:
                pickle.dump(self.obj, f)

    @classmethod
    def load(cls, n_docs: int) -> "SparseIndex":
        if config.SPARSE_BACKEND == "bm25s":
            bm25s = _import_bm25s()
            if bm25s is None:
                with open(config.BM25_PATH, "rb") as f:
                    return cls("rank_bm25", pickle.load(f), n_docs)
            return cls("bm25s", bm25s.BM25.load(config.BM25_DIR, mmap=True), n_docs)
        with open(config.BM25_PATH, "rb") as f:
            return cls("rank_bm25", pickle.load(f), n_docs)

    def query(self, query: str, k: int) -> list[int]:
        k = max(1, min(k, self.n_docs))
        if self.backend == "bm25s":
            bm25s = _import_bm25s()
            if bm25s is None:
                return []
            try:
                res, _ = self.obj.retrieve(
                    bm25s.tokenize(query, stopwords="en", show_progress=False),
                    k=k, show_progress=False)
                return [int(i) for i in res[0]]
            except Exception:
                return []
        scores = self.obj.get_scores(tokenize(query))
        if not len(scores):
            return []
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [int(i) for i in top if scores[i] > 0]


# ── model wrappers ────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        device = device or get_device()
        print(f"Embedder '{model_name}' on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = config.EMBED_BATCH):
        return self.model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=config.SHOW_PROGRESS,
        ).astype("float32")


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device or get_device())

    def scores(self, query: str, docs: list[str]):
        return np.asarray(self.model.predict([(query, d) for d in docs]))


class LocalLLM:
    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def from_path(cls, path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        return cls(Llama(model_path=path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, verbose=False))

    @classmethod
    def from_repo(cls, repo: str = config.LLM_REPO, filename: str = config.LLM_FILE,
                  n_ctx: int = 4096, n_gpu_layers: int = -1):
        """Auto-download the GGUF from Hugging Face and load it - no manual step."""
        from llama_cpp import Llama
        return cls(Llama.from_pretrained(repo_id=repo, filename=filename,
                                         n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                                         verbose=False))

    def chat(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2):
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()

    def expand_queries(self, query: str, n: int = 3) -> list[str]:
        sys = ("You rewrite a library search query into alternative phrasings "
               "to improve retrieval. Output one rewrite per line, no numbering.")
        try:
            text = self.chat(sys, f"Query: {query}\nGive {n} rewrites.",
                             max_tokens=128, temperature=0.4)
            return [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()][:n]
        except Exception:
            return []


@dataclass
class SearchResult:
    bibnum: str
    title: str
    author: str
    subjects: str
    item_type: str
    year: int
    copies: int
    locations: str
    score: float
    citation: str = field(default="")


class LibrarySearchEngine:
    def __init__(self, corpus, faiss_index, sparse: SparseIndex, vocab=None,
                 embedder=None, reranker=None, llm=None):
        self.corpus = corpus
        self.index = faiss_index
        self.sparse = sparse
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        if vocab is None:
            vocab = set()
            for col in ("Title", "Author"):
                for t in corpus[col].tolist():
                    vocab.update(tokenize(t))
        self.vocab = vocab

    @classmethod
    def load(cls, with_rerank: bool = True, enable_llm: bool = False):
        import faiss
        corpus = pd.read_parquet(config.CORPUS_PARQUET)
        index = faiss.read_index(config.FAISS_PATH)
        sparse = SparseIndex.load(len(corpus))
        vocab = None
        if os.path.exists(config.VOCAB_PATH):
            with open(config.VOCAB_PATH, "rb") as f:
                vocab = pickle.load(f)
        embedder = Embedder()
        reranker = Reranker() if with_rerank else None
        llm = build_llm() if enable_llm else None
        return cls(corpus, index, sparse, vocab=vocab,
                   embedder=embedder, reranker=reranker, llm=llm)

    # ---- retrieval --------------------------------------------------------
    def _dense(self, query: str, k: int) -> list[int]:
        emb = self.embedder.encode([query])
        _, idx = self.index.search(emb, k)
        return [int(i) for i in idx[0] if i >= 0]

    @staticmethod
    def _rrf(ranked_lists):
        fused: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, pos in enumerate(lst):
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def correct_query(self, query: str):
        from rapidfuzz import process, fuzz
        out, changed = [], False
        for tok in tokenize(query):
            if tok in self.vocab or len(tok) <= 3:
                out.append(tok); continue
            match = process.extractOne(tok, self.vocab, scorer=fuzz.ratio)
            if match and match[1] >= config.FUZZY_MIN_SCORE:
                out.append(match[0]); changed = True
            else:
                out.append(tok)
        return " ".join(out), changed

    def _apply_filters(self, idxs, filters):
        if not filters:
            return idxs
        sub = self.corpus.iloc[idxs]
        mask = pd.Series(True, index=sub.index)
        if filters.get("author"):
            mask &= sub["Author"].str.contains(filters["author"], case=False, na=False)
        if filters.get("subjects"):
            mask &= sub["Subjects"].str.contains(filters["subjects"], case=False, na=False)
        if filters.get("item_type"):
            mask &= sub["ItemType"].isin(filters["item_type"])
        if filters.get("year_min"):
            mask &= sub["Year"] >= int(filters["year_min"])
        if filters.get("year_max"):
            mask &= sub["Year"] <= int(filters["year_max"])
        return [idxs[i] for i, keep in enumerate(mask.tolist()) if keep]

    def search(self, query: str, k: int = 10, filters: dict | None = None,
               use_rerank: bool = True, use_multiquery: bool = False,
               candidate_k: int = config.CANDIDATE_K, log: bool = True):
        t0 = datetime.now()
        corrected, changed = self.correct_query(query)
        queries = [corrected]
        if use_multiquery and self.llm is not None:
            queries += self.llm.expand_queries(corrected)

        ranked_lists = []
        for q in queries:
            ranked_lists.append(self._dense(q, candidate_k))
            ranked_lists.append(self.sparse.query(q, candidate_k))
        fused = self._rrf(ranked_lists)
        idxs = [pos for pos, _ in fused]
        fused_score = {pos: s for pos, s in fused}
        idxs = self._apply_filters(idxs, filters)[: max(candidate_k, k)]

        if use_rerank and self.reranker is not None and idxs:
            rr = self.reranker.scores(corrected, self.corpus.iloc[idxs]["text"].tolist())
            conf = sigmoid(rr)
            order = np.argsort(-rr)
            idxs = [idxs[i] for i in order]
            scores = [float(conf[i]) for i in order]
        elif idxs:
            raw = np.array([fused_score.get(p, 0.0) for p in idxs])
            lo, hi = raw.min(), raw.max()
            scores = list((raw - lo) / (hi - lo + 1e-9))
        else:
            scores = []

        results = []
        for pos, sc in list(zip(idxs, scores))[:k]:
            r = self.corpus.iloc[pos]
            results.append(SearchResult(
                bibnum=r["BibNum"], title=r["Title"], author=r["Author"],
                subjects=r["Subjects"], item_type=r["ItemType"],
                year=int(r["Year"]), copies=int(r["Copies"]),
                locations=r["Locations"], score=round(float(sc), 3),
                citation=f"[BibNum {r['BibNum']}]",
            ))
        meta = {"original": query, "corrected": corrected, "typo_fixed": changed,
                "confidence": results[0].score if results else 0.0,
                "latency_ms": int((datetime.now() - t0).total_seconds() * 1000),
                "n_results": len(results)}
        if log:
            self._log(meta)
        return results, meta

    def lookup_availability(self, title_query: str):
        from rapidfuzz import process, fuzz
        match = process.extractOne(title_query, self.corpus["Title"].tolist(),
                                   scorer=fuzz.WRatio)
        if not match or match[1] < 60:
            return None
        row = self.corpus.iloc[match[2]]
        return {"title": row["Title"], "author": row["Author"],
                "copies": int(row["Copies"]),
                "locations": row["Locations"].split("; ") if row["Locations"] else [],
                "match_score": round(match[1] / 100, 2)}

    def answer(self, query, results, meta, history=None):
        if not results:
            return "I could not find anything matching that in the catalogue."
        note = ("" if meta["confidence"] >= config.ABSTAIN_BELOW else
                "I am not very confident - the closest matches are weak, so "
                "treat these as guesses:\n")
        context = "\n".join(
            f"{r.citation} '{r.title}' by {r.author or 'unknown'} "
            f"({r.item_type}, {r.year}). Subjects: {r.subjects or 'n/a'}. "
            f"Copies: {r.copies}, branches: {r.locations or 'n/a'}." for r in results)
        if self.llm is None:
            return note + context
        sys = ("You are a library assistant. Answer using ONLY the catalogue "
               "entries provided. Cite the BibNum in square brackets after any "
               "book you mention. If the entries do not answer the question, "
               "say so plainly.")
        convo = ""
        if history:
            convo = "Earlier turns:\n" + "\n".join(
                f"{h['role']}: {h['content']}" for h in history[-4:]) + "\n\n"
        return note + self.llm.chat(sys, f"{convo}Catalogue entries:\n{context}\n\nQuestion: {query}")

    def _log(self, meta):
        os.makedirs(config.ARTEFACT_DIR, exist_ok=True)
        new = not os.path.exists(config.HISTORY_CSV)
        with open(config.HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "query", "corrected", "typo_fixed",
                            "confidence", "latency_ms", "n_results"])
            w.writerow([datetime.now(timezone.utc).isoformat(), meta["original"],
                        meta["corrected"], meta["typo_fixed"], meta["confidence"],
                        meta["latency_ms"], meta["n_results"]])


def build_llm():
    """Resolve a local LLM: explicit path if given, else auto-download by repo."""
    if config.DEFAULT_LLM_PATH and os.path.exists(config.DEFAULT_LLM_PATH):
        return LocalLLM.from_path(config.DEFAULT_LLM_PATH)
    return LocalLLM.from_repo()
