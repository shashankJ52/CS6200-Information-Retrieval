"""
pool.py — build the judgment pool for each query.

Runs two retrievers over the frozen corpus and unions their top-k. Pooling
from more than one system is what keeps the collection from being biased
toward whichever retriever built it: a document only one system finds is
still judged, so a future system that finds it is not penalised for
retrieving something unlabelled.

    python src/pool.py --depth 25 --dry-run    # report sizes, write nothing
    python src/pool.py --depth 25
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from bm25_index import Index

QUERIES = Path("data/queries/queries.json")
EMB_DIR = Path("data/embeddings")
OUT_DIR = Path("data/pools")

MODEL_NAME = "allenai/specter2_base"
SHUFFLE_SEED = 20261119


def embed_queries(texts, device):
    """Encode query strings with the same model used for the corpus.

    SPECTER2 was trained on 'title [SEP] abstract', so a short query is
    out of distribution. Repeating the query on both sides of the
    separator matches the training format's shape, which is the usual
    workaround. Good enough for pooling, where the job is to surface
    plausible candidates rather than to rank them correctly.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    formatted = [f"{t}{tokenizer.sep_token}{t}" for t in texts]
    enc = tokenizer(formatted, padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)

    with torch.inference_mode():
        cls = model(**enc).last_hidden_state[:, 0, :]
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=25,
                    help="Top-k taken from each retriever before the union.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report pool sizes and overlap, write nothing.")
    args = ap.parse_args()

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))

    # Dense side: corpus vectors are already L2-normalised, so cosine
    # similarity is a dot product.
    emb = np.load(EMB_DIR / "embeddings.npy").astype(np.float32)
    emb_ids = json.loads((EMB_DIR / "ids.json").read_text(encoding="utf-8"))

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    q_vecs = embed_queries([q["query"] for q in queries], device)

    index = Index.build()
    by_id = {d["doc_id"]: d for d in index.docs}

    rows, sizes = [], []
    for q, q_vec in zip(queries, q_vecs):
        qid = q["query_id"]

        bm25 = [d["doc_id"] for d, _ in index.search(q["query"], k=args.depth)]

        sims = emb @ q_vec
        dense = [emb_ids[i] for i in np.argsort(-sims)[:args.depth]]

        # Record which retriever found each document and at what rank.
        # Needed later to report pool composition; never shown during
        # annotation, where it could bias the judgment.
        found = {}
        for rank, doc_id in enumerate(bm25, 1):
            found.setdefault(doc_id, {})["bm25_rank"] = rank
        for rank, doc_id in enumerate(dense, 1):
            found.setdefault(doc_id, {})["dense_rank"] = rank

        # Shuffle so the annotator does not see BM25 order and read it as
        # a relevance signal. Seeded per query, so the order is reproducible.
        pool = list(found)
        random.Random(f"{SHUFFLE_SEED}-{qid}").shuffle(pool)

        overlap = len(set(bm25) & set(dense))
        sizes.append(len(pool))
        rows.append((qid, len(bm25), len(dense), overlap, len(pool)))

        if not args.dry_run:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{qid}.json").write_text(json.dumps({
                "query_id": qid,
                "query": q["query"],
                "depth": args.depth,
                "n_candidates": len(pool),
                "n_overlap": overlap,
                "candidates": [{
                    "doc_id": doc_id,
                    "title": by_id[doc_id]["title"],
                    "abstract": by_id[doc_id]["abstract"],
                    "retrieved_by": found[doc_id],
                } for doc_id in pool],
            }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'query':6} {'bm25':>5} {'dense':>6} {'both':>5} {'pool':>5}")
    print("-" * 32)
    for qid, nb, nd, ov, np_ in rows:
        print(f"{qid:6} {nb:>5} {nd:>6} {ov:>5} {np_:>5}")
    print("-" * 32)
    print(f"total {sum(sizes):>26}")
    print(f"mean pool {sum(sizes)/len(sizes):>22.1f}")
    print(f"est. hours at 2 min each {sum(sizes)*2/60:>8.1f}")

    if args.dry_run:
        print("\nDry run — nothing written.")
    else:
        print(f"\nWrote {len(rows)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()