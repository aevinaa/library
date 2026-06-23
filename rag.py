import os
import re
import csv
import pickle
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ARTEFACT_DIR = "./index"
CORPUS_PARQUET = os.path.join(ARTEFACT_DIR, "corpus.parquet")
FAISS_PATH = os.path.join(ARTEFACT_DIR, "dense.faiss")
BM25_PATH = os.path.join(ARTEFACT_DIR, "bm25.pkl")
HISTORY_CSV = os.path.join(ARTEFACT_DIR, "history.csv")

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM_PATH = os.environ.get("LLM_GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf") 

READ_CHUNK_ROWS = 50_000
EMBED_BATCH = 256
RRF_K = 60                 
CANDIDATE_K = 50           
FUZZY_MIN_SCORE = 82       
ABSTAIN_BELOW = 0.30       

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def clean_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def extract_year(raw: str) -> int:
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", raw or "")
    return int(m.group(1)) if m else 0

def build_corpus(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Stream the CSV, collapse to one row per BibNum, aggregating copies and
    the set of branch locations so availability lookups actually work."""
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
    print(f"Corpus: {len(df):,} unique titles from the catalogue.")
    return df


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if _cuda() else "cpu"
        print(f"Embedder '{model_name}' on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=EMBED_BATCH):
        return self.model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None):
        from sentence_transformers import CrossEncoder
        if device is None:
            device = "cuda" if _cuda() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def scores(self, query: str, docs: list[str]):
        return np.asarray(self.model.predict([(query, d) for d in docs]))


class LocalLLM:
    """Thin wrapper over llama-cpp-python (in-process llama.cpp)."""
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, verbose=False)

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


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    def __init__(self, corpus: pd.DataFrame, faiss_index, bm25,
                 embedder: Embedder | None = None,
                 reranker: Reranker | None = None,
                 llm: LocalLLM | None = None):
        self.corpus = corpus
        self.index = faiss_index
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm

        self.vocab = set()
        for col in ("Title", "Author"):
            for txt in corpus[col].tolist():
                self.vocab.update(tokenize(txt))

 
    @classmethod
    def build(cls, csv_path: str, max_rows: int | None = None,
              with_models: bool = True):
        import faiss
        from rank_bm25 import BM25Okapi
        os.makedirs(ARTEFACT_DIR, exist_ok=True)

        corpus = build_corpus(csv_path, max_rows=max_rows)
        corpus.to_parquet(CORPUS_PARQUET, index=False)


        print("Building BM25 (sparse) index ...")
        tokenised = [tokenize(t) for t in corpus["text"].tolist()]
        bm25 = BM25Okapi(tokenised)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

        embedder = Embedder() if with_models else None
        if embedder is not None:
            print("Embedding documents for the dense index ...")
            embs = embedder.encode(corpus["text"].tolist())
            index = faiss.IndexFlatIP(embs.shape[1])   
            index.add(embs)
            faiss.write_index(index, FAISS_PATH)
            print(f"Dense index: {index.ntotal:,} vectors, dim {embs.shape[1]}.")
        else:
            index = None
        return cls(corpus, index, bm25, embedder=embedder)

    @classmethod
    def load(cls, llm_path: str | None = None, with_rerank: bool = True):
        import faiss
        corpus = pd.read_parquet(CORPUS_PARQUET)
        index = faiss.read_index(FAISS_PATH)
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        embedder = Embedder()
        reranker = Reranker() if with_rerank else None
        llm = None
        path = llm_path or DEFAULT_LLM_PATH
        if path and os.path.exists(path):
            llm = LocalLLM(path)
        return cls(corpus, index, bm25, embedder=embedder,
                   reranker=reranker, llm=llm)


    def _dense(self, query: str, k: int) -> list[int]:
        emb = self.embedder.encode([query])
        _, idx = self.index.search(emb, k)
        return [int(i) for i in idx[0] if i >= 0]

    def _sparse(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [int(i) for i in top if scores[i] > 0]

    @staticmethod
    def _rrf(ranked_lists: list[list[int]]) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion across any number of ranked position lists."""
        fused: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, pos in enumerate(lst):
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])


    def correct_query(self, query: str) -> tuple[str, bool]:
        from rapidfuzz import process, fuzz
        out, changed = [], False
        for tok in tokenize(query):
            if tok in self.vocab or len(tok) <= 3:
                out.append(tok)
                continue
            match = process.extractOne(tok, self.vocab, scorer=fuzz.ratio)
            if match and match[1] >= FUZZY_MIN_SCORE:
                out.append(match[0]); changed = True
            else:
                out.append(tok)
        return " ".join(out), changed

    def _apply_filters(self, idxs: list[int], filters: dict) -> list[int]:
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
               candidate_k: int = CANDIDATE_K, log: bool = True):
        t0 = datetime.now()
        corrected, changed = self.correct_query(query)

        queries = [corrected]
        if use_multiquery and self.llm is not None:
            queries += self.llm.expand_queries(corrected)

        ranked_lists = []
        for q in queries:
            ranked_lists.append(self._dense(q, candidate_k))
            ranked_lists.append(self._sparse(q, candidate_k))
        fused = self._rrf(ranked_lists)
        idxs = [pos for pos, _ in fused]
        fused_score = {pos: s for pos, s in fused}

        idxs = self._apply_filters(idxs, filters)[: max(candidate_k, k)]

        if use_rerank and self.reranker is not None and idxs:
            docs = self.corpus.iloc[idxs]["text"].tolist()
            rr = self.reranker.scores(corrected, docs)
            conf = sigmoid(rr)
            order = np.argsort(-rr)
            idxs = [idxs[i] for i in order]
            scores = [float(conf[i]) for i in order]
        else:

            if idxs:
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

        meta = {
            "original": query,
            "corrected": corrected,
            "typo_fixed": changed,
            "confidence": results[0].score if results else 0.0,
            "latency_ms": int((datetime.now() - t0).total_seconds() * 1000),
            "n_results": len(results),
        }
        if log:
            self._log(meta)
        return results, meta

    def lookup_availability(self, title_query: str):
        from rapidfuzz import process, fuzz
        titles = self.corpus["Title"].tolist()
        match = process.extractOne(title_query, titles, scorer=fuzz.WRatio)
        if not match or match[1] < 60:
            return None
        row = self.corpus.iloc[match[2]]
        return {
            "title": row["Title"], "author": row["Author"],
            "copies": int(row["Copies"]),
            "locations": row["Locations"].split("; ") if row["Locations"] else [],
            "match_score": round(match[1] / 100, 2),
        }

    def answer(self, query: str, results: list[SearchResult],
               meta: dict, history: list | None = None) -> str:
        if not results:
            return "I could not find anything matching that in the catalogue."
        if meta["confidence"] < ABSTAIN_BELOW:
            note = ("I am not very confident about this - the closest matches "
                    "are weak, so treat these as guesses:\n")
        else:
            note = ""

        context = "\n".join(
            f"{r.citation} '{r.title}' by {r.author or 'unknown'} "
            f"({r.item_type}, {r.year}). Subjects: {r.subjects or 'n/a'}. "
            f"Copies: {r.copies}, branches: {r.locations or 'n/a'}."
            for r in results
        )
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
        user = f"{convo}Catalogue entries:\n{context}\n\nQuestion: {query}"
        return note + self.llm.chat(sys, user)

    def _log(self, meta: dict):
        os.makedirs(ARTEFACT_DIR, exist_ok=True)
        new = not os.path.exists(HISTORY_CSV)
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "query", "corrected", "typo_fixed",
                            "confidence", "latency_ms", "n_results"])
            w.writerow([datetime.now(timezone.utc).isoformat(), meta["original"],
                        meta["corrected"], meta["typo_fixed"], meta["confidence"],
                        meta["latency_ms"], meta["n_results"]])


def main():
    ap = argparse.ArgumentParser(description="Build the hybrid library index")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    if args.build:
        LibrarySearchEngine.build(args.csv, max_rows=args.max_rows)
        print("Index built in", ARTEFACT_DIR)
    else:
        print("Nothing to do. Pass --build to index.")


if __name__ == "__main__":
    main()
