"""
build_corpus.py — one-time corpus construction.

Reads the arXiv metadata dump, applies three filters (category, date
window, survey exclusion), samples to a target size, and writes a
frozen corpus plus an ID manifest.

Runs ONCE for real. After that, data/corpus/ is immutable — every
qrel and run file downstream references these document IDs.

    python src/build_corpus.py --raw data/raw/sample.json --dry-run
    python src/build_corpus.py
"""

import argparse
import json
import random
import re
from pathlib import Path

OUT_DIR = Path("data/corpus")

CATEGORIES = {"cs.CL", "cs.IR", "cs.LG"}
DATE_FROM = "2024-01-01"
DATE_TO = "2026-11-13"
TARGET_SIZE = 200_000
SEED = 20261114

SURVEY_PATTERN = re.compile(
    r"\b(survey|a review of|systematic review|overview of|"
    r"taxonomy of|tutorial)\b",
    re.IGNORECASE,
)

MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def first_version_date(record):
    """Original submission date, not the latest revision.

    A 2022 paper revised last month must not count as 2026 — that
    would smuggle older literature past the date filter.
    """
    versions = record.get("versions") or []
    if not versions:
        return None
    return versions[0].get("created")  # "Mon, 2 Apr 2007 19:18:42 GMT"


def in_date_window(created_str):
    if not created_str:
        return False
    try:
        _, day, mon, year = created_str.split()[:4]
        iso = f"{year}-{MONTHS[mon]:02d}-{int(day):02d}"
    except (ValueError, KeyError, IndexError):
        return False
    return DATE_FROM <= iso <= DATE_TO


def is_survey(title, abstract):
    """Exclude survey-type papers.

    Scoping decision, not cleanup: a survey covers every facet by
    construction, so it would inflate coverage metrics without any
    system having diversified. The task is about assembling coverage
    from primary literature.

    Checks the title and the opening of the abstract — many surveys
    announce themselves in the first sentence without saying so in
    the title. Scanning the whole abstract would over-trigger on
    papers that merely cite surveys.
    """
    return bool(SURVEY_PATTERN.search(title)
                or SURVEY_PATTERN.search(abstract[:300]))


def iter_candidates(path):
    """Stream the dump, yielding records that pass every filter.

    Generator rather than a full load: the dump is ~4GB and we keep
    a small fraction of it.
    """
    stats = {"seen": 0, "category": 0, "date": 0, "survey": 0, "kept": 0}

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stats["seen"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = set((rec.get("categories") or "").split())
            if not cats & CATEGORIES:
                continue
            stats["category"] += 1

            if not in_date_window(first_version_date(rec)):
                continue
            stats["date"] += 1

            # Collapse embedded newlines and runs of whitespace;
            # BM25 tokenization and SPECTER2 both suffer without this.
            title = " ".join((rec.get("title") or "").split())
            abstract = " ".join((rec.get("abstract") or "").split())
            if not title or not abstract:
                continue

            if is_survey(title, abstract):
                stats["survey"] += 1
                continue

            stats["kept"] += 1
            yield {
                "doc_id": rec["id"],
                "title": title,
                "abstract": abstract,
                "categories": sorted(cats & CATEGORIES),
                "created": first_version_date(rec),
            }

    iter_candidates.stats = stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/arxiv-metadata-oai-snapshot.json"),
        help="Path to the arXiv metadata dump (JSON Lines).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report stats, but write nothing.",
    )
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"No such file: {args.raw}")

    print(f"Streaming {args.raw} ...")
    pool = list(iter_candidates(args.raw))
    stats = iter_candidates.stats

    print(f"  records seen:        {stats['seen']:>8,}")
    print(f"  in target categories:{stats['category']:>8,}")
    print(f"  in date window:      {stats['date']:>8,}")
    print(f"  surveys excluded:    {stats['survey']:>8,}")
    print(f"  eligible:            {stats['kept']:>8,}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(pool) > TARGET_SIZE:
        random.Random(SEED).shuffle(pool)
        pool = pool[:TARGET_SIZE]
        print(f"  sampled down to:     {len(pool):>8,}  (seed {SEED})")
    else:
        print(f"  keeping all          {len(pool):>8,}  (below target)")

    corpus_path = OUT_DIR / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc in pool:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    manifest_path = OUT_DIR / "manifest.txt"
    manifest_path.write_text(
        "\n".join(sorted(d["doc_id"] for d in pool)) + "\n",
        encoding="utf-8",
    )

    (OUT_DIR / "build_stats.json").write_text(
        json.dumps({**stats, "final_size": len(pool), "seed": SEED,
                    "categories": sorted(CATEGORIES),
                    "date_from": DATE_FROM, "date_to": DATE_TO},
                   indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {corpus_path} and {manifest_path}")
    print("Corpus is now FROZEN. Commit it before going further.")


if __name__ == "__main__":
    main()