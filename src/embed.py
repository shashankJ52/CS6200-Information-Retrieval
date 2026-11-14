"""
embed.py — one-time SPECTER2 embedding pass over the frozen corpus.

Produces a dense vector per abstract, used in three places downstream:
pooling (as a second retriever beside BM25), MMR (cosine penalty), and
the DPP kernel (volume of the embedded set).

Output row i corresponds to ids[i]. The order is NOT corpus order —
documents are sorted by length for batching efficiency, and the id list
records the order actually used. Always look up by doc_id.

    python src/embed.py --limit 500      # smoke test, ~1 minute
    python src/embed.py                  # full run, ~1-2 hours
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CORPUS = Path("data/corpus/corpus.jsonl")
OUT_DIR = Path("data/embeddings")
PARTS_DIR = OUT_DIR / "parts"

MODEL_NAME = "allenai/specter2_base"
MAX_TOKENS = 512
BATCH_SIZE = 64
CHUNK_SIZE = 20_000


def load_docs(path):
    """Read the corpus and order it deterministically by length.

    Sorting groups similar-length abstracts into the same batch, so
    padding wastes far less compute. The tiebreak on doc_id makes the
    order reproducible, which is what lets a crashed run resume.
    """
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append((rec["doc_id"], rec["title"], rec["abstract"]))
    docs.sort(key=lambda d: (len(d[2]), d[0]))
    return docs


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def embed_batch(batch, tokenizer, model, device):
    """Encode one batch to L2-normalised CLS vectors.

    SPECTER's document representation is the [CLS] token, not a mean
    over tokens, and title and abstract are joined by the separator
    the model was trained with.
    """
    texts = [f"{title}{tokenizer.sep_token}{abstract}"
             for _, title, abstract in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out = model(**enc)

    cls = out.last_hidden_state[:, 0, :]
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Embed only the first N documents (smoke test).")
    args = parser.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"No such file: {args.corpus}")

    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs(args.corpus)
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs):,} documents")

    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    n_parts = (len(docs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    started = time.time()

    for part in range(n_parts):
        part_path = PARTS_DIR / f"part_{part:04d}.npy"
        if part_path.exists():
            print(f"part {part + 1}/{n_parts}: already done, skipping")
            continue

        chunk = docs[part * CHUNK_SIZE:(part + 1) * CHUNK_SIZE]
        vectors = []
        for start in tqdm(range(0, len(chunk), BATCH_SIZE),
                          desc=f"part {part + 1}/{n_parts}"):
            vectors.append(
                embed_batch(chunk[start:start + BATCH_SIZE],
                            tokenizer, model, device)
            )

        # Written only once the whole part succeeded, so a half-finished
        # part is never mistaken for a complete one on resume.
        np.save(part_path, np.vstack(vectors))

    parts = [np.load(PARTS_DIR / f"part_{p:04d}.npy") for p in range(n_parts)]
    embeddings = np.vstack(parts)

    ids = [doc_id for doc_id, _, _ in docs]
    if embeddings.shape[0] != len(ids):
        raise SystemExit(
            f"Row count {embeddings.shape[0]} does not match {len(ids)} ids"
        )
    if not np.isfinite(embeddings).all():
        raise SystemExit("Embeddings contain NaN or inf")

    norms = np.linalg.norm(embeddings[:1000].astype(np.float32), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise SystemExit(f"Vectors are not unit length (min {norms.min():.4f})")

    np.save(OUT_DIR / "embeddings.npy", embeddings)
    (OUT_DIR / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (OUT_DIR / "embed_meta.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "pooling": "cls",
        "normalized": True,
        "dtype": "float16",
        "max_tokens": MAX_TOKENS,
        "n_docs": len(ids),
        "dim": int(embeddings.shape[1]),
        "device": str(device),
        "minutes": round((time.time() - started) / 60, 1),
    }, indent=2), encoding="utf-8")

    print(f"\n{embeddings.shape[0]:,} x {embeddings.shape[1]} vectors "
          f"in {(time.time() - started) / 60:.1f} min")
    print(f"Wrote {OUT_DIR / 'embeddings.npy'} and {OUT_DIR / 'ids.json'}")


if __name__ == "__main__":
    main()