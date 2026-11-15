"""
bm25_index.py — BM25 over the frozen corpus.

Imported by the support check and by pooling. Builds in ~90 seconds
over 190k documents; no index is cached to disk because rank_bm25's
in-memory structures pickle larger and load slower than a rebuild.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CORPUS = Path("data/corpus/corpus.jsonl")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "to", "we", "with", "which", "these", "their",
    "can", "such", "our", "also", "but", "not", "they", "than",
}

# Keeps compounds intact: "over-smoothing", "low-rank", "state-of-the-art".
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text):
    """Lowercase, split, and emit hyphenated compounds twice.

    A query for "over smoothing" and a query for "over-smoothing"
    should both match a document containing either form, so the
    compound is indexed alongside its parts.
    """
    tokens = []
    for term in TOKEN_RE.findall(text.lower()):
        if term not in STOPWORDS and len(term) > 1:
            tokens.append(term)
        if "-" in term:
            tokens.extend(
                part for part in term.split("-")
                if part not in STOPWORDS and len(part) > 1
            )
    return tokens


def load_corpus(path=CORPUS):
    """Return the corpus as a list of dicts, in file order."""
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class Index:
    """BM25 over title + abstract, with doc_id lookup."""

    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi(
            [tokenize(f"{d['title']} {d['abstract']}") for d in docs]
        )

    @classmethod
    def build(cls, path=CORPUS, verbose=True):
        docs = load_corpus(path)
        if verbose:
            print(f"Indexing {len(docs):,} documents...")
        index = cls(docs)
        if verbose:
            print("Ready.")
        return index

    def search(self, query, k=20):
        """Top-k documents for a query, as (doc, score) pairs."""
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], float(scores[i])) for i in top]