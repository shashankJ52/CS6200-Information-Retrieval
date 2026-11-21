"""
build_qrels.py — turn exported judgments into the frozen answer key.

Reads one or more exports from the annotation page and writes the qrels
in two forms: a JSON file carrying the full two-part label, and a
TREC-format file carrying the relevance grade alone, which is what ranx
reads for NDCG.

Alpha-NDCG needs the per-facet vectors, so the JSON is the primary
artifact and the TREC file is a convenience.

    python src/build_qrels.py annotation/ir-self-2026-11-25.json
    python src/build_qrels.py --check annotation/ir-self-*.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

QUERIES = Path("data/queries/queries.json")
OUT_DIR = Path("data/qrels")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exports", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="Report coverage and facet statistics, write nothing.")
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in
               json.loads(QUERIES.read_text(encoding="utf-8"))}

    judgments = {}
    for path in args.exports:
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data["judgments"]:
            judgments[(r["query_id"], r["doc_id"])] = r

    # Every facet vector must match the arity its query declares; a
    # mismatch means the export predates a change to the facets.
    problems = []
    for (qid, doc_id), r in judgments.items():
        n = len(queries[qid]["facets"])
        if r["grade"] != 0 and len(r.get("facets", [])) != n:
            problems.append(f"{qid}/{doc_id}: {len(r.get('facets', []))} facets, expected {n}")
        if r["grade"] == 0 and any(r.get("facets", [])):
            problems.append(f"{qid}/{doc_id}: grade 0 with facets set")
    if problems:
        print("PROBLEMS")
        for p in problems[:20]:
            print(" ", p)
        if not args.check:
            raise SystemExit("Fix these before freezing.")

    print(f"\n{'query':6} {'judged':>7} {'rel':>5} {'grade 2':>8}  facet positives")
    print("-" * 64)
    for qid, q in queries.items():
        rows = [r for (k, _), r in
                (((k, d), r) for (k, d), r in judgments.items()) if k == qid]
        rel = [r for r in rows if r["grade"] > 0]
        counts = Counter()
        for r in rel:
            for i, v in enumerate(r.get("facets", [])):
                if v and i < len(q["facets"]):
                    counts[q["facets"][i]["id"]] += 1
        spread = " ".join(f"{f['id']}:{counts.get(f['id'], 0)}" for f in q["facets"])
        print(f"{qid:6} {len(rows):>7} {len(rel):>5} "
              f"{sum(1 for r in rel if r['grade'] == 2):>8}  {spread}")

    flagged = [k for k, r in judgments.items() if r.get("flag")]
    print(f"\n{len(judgments)} judgments, {len(flagged)} flagged for review")

    if args.check:
        print("\nCheck only — nothing written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = sorted(judgments.values(), key=lambda r: (r["query_id"], r["doc_id"]))
    (OUT_DIR / "qrels.json").write_text(json.dumps({
        "n_queries": len(queries),
        "n_judgments": len(rows),
        "judgments": [{k: r[k] for k in ("query_id", "doc_id", "grade", "facets")
                       if k in r} for r in rows],
    }, indent=2), encoding="utf-8")

    with (OUT_DIR / "qrels.trec").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"{r['query_id']} 0 {r['doc_id']} {r['grade']}\n")

    print(f"\nWrote {OUT_DIR}/qrels.json and {OUT_DIR}/qrels.trec")
    print("Commit and tag this before running any system.")


if __name__ == "__main__":
    main()