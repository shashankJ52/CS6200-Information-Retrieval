"""
support_check.py — does a facet have enough documents to measure?

Milestone B3. A facet with fewer than ~10 relevant documents in the
corpus makes alpha-NDCG@10 meaningless for its query, so every facet
is checked before the collection is frozen.

Judgement is by eye: BM25 scores are not comparable across queries,
so a score threshold would be arbitrary. The tool prints titles and
you count how many are plausibly about the facet.

    python src/support_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bm25_index import Index


def main():
    index = Index.build()

    print("\nType facet terms, or 'quit'.\n")
    while True:
        try:
            query = input("facet> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query == "quit":
            break

        for rank, (doc, score) in enumerate(index.search(query, k=20), 1):
            print(f"{rank:>3}. {score:6.2f}  {doc['title'][:95]}")
        print()


if __name__ == "__main__":
    main()
