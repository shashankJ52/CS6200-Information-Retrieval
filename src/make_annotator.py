"""
make_annotator.py — build a self-contained annotation page.

Embeds the pools and the narratives into one HTML file so the criteria
are on screen for every judgment. Across 634 documents and roughly
twenty hours, the thing that protects consistency is not remembering the
rules but never having to.

Progress lives in the browser's localStorage and is exported as JSON.
The peer build uses a different storage key and a sampled subset, so two
people can annotate on the same machine without seeing each other's
labels.

    python src/make_annotator.py
    python src/make_annotator.py --mode peer --fraction 0.2
"""

import argparse
import json
import random
from pathlib import Path

QUERIES = Path("data/queries/queries.json")
POOLS = Path("data/pools")
OUT_DIR = Path("annotation")

PEER_SEED = 20261119


def build_payload(fraction, seed):
    """Collect queries, narratives and pooled candidates into one object.

    For the peer build, sample within each query rather than across the
    whole collection, so the peer sees every query and agreement can be
    computed per query as well as overall.
    """
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    out = []

    for q in queries:
        pool_path = POOLS / f"{q['query_id']}.json"
        if not pool_path.exists():
            raise SystemExit(f"Missing pool: {pool_path}. Run pool.py first.")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))

        cands = pool["candidates"]
        if fraction < 1.0:
            k = max(1, round(len(cands) * fraction))
            rng = random.Random(f"{seed}-{q['query_id']}")
            cands = sorted(rng.sample(cands, k),
                           key=lambda c: cands.index(c))

        out.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "source": q["facet_source"],
            "facets": q["facets"],
            "narrative": q["narrative"],
            "rules": q["rules"],
            "docs": [{"doc_id": c["doc_id"],
                      "title": c["title"],
                      "abstract": c["abstract"]} for c in cands],
        })
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #F1F1F6; --card: #FFFFFF; --ink: #1C1C2E; --gray: #6B6B81;
  --line: #E4E4EE; --purple: #5B35D5; --purpleSoft: #EEE9FC;
  --green: #11914A; --greenSoft: #EAF7EF; --red: #CE2B24; --redSoft: #FBEEED;
  --amber: #B7791F; --amberSoft: #FDF6E7;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#app { display: grid; grid-template-columns: 380px 1fr; min-height: 100vh; }

/* criteria panel */
aside {
  background: var(--card); border-right: 2px solid var(--line);
  padding: 20px; overflow-y: auto; height: 100vh;
  position: sticky; top: 0;
}
aside h1 { font-size: 17px; margin: 0 0 4px; }
aside .src { font-size: 12px; color: var(--gray); margin-bottom: 16px; }
aside h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--gray); margin: 18px 0 8px; font-weight: 700;
}
.facetdef {
  border-left: 3px solid var(--purple); padding: 6px 0 6px 10px;
  margin-bottom: 10px;
}
.facetdef b { color: var(--purple); }
.facetdef p { margin: 3px 0 0; font-size: 13px; color: var(--ink); }
aside .narr { font-size: 13px; margin: 0 0 10px; }
aside .narr b { display: block; font-size: 11px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--gray); margin-bottom: 2px; }
aside .narr.no b { color: var(--red); }
aside ol { margin: 0; padding-left: 18px; font-size: 12.5px; color: var(--gray); }
aside ol li { margin-bottom: 7px; }

/* main */
main { padding: 24px 32px 120px; max-width: 900px; }
.bar { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
.bar select { font-size: 14px; padding: 6px 10px; border-radius: 8px;
  border: 2px solid var(--line); background: var(--card); }
.prog { flex: 1; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
.prog span { display: block; height: 100%; background: var(--purple); width: 0; }
.count { font-size: 13px; color: var(--gray); white-space: nowrap; }

.doc { background: var(--card); border: 2px solid var(--line);
  border-radius: 14px; padding: 22px 26px; }
.doc .id { font: 12px ui-monospace, Menlo, monospace; color: var(--gray); }
.doc h3 { margin: 6px 0 14px; font-size: 20px; line-height: 1.3; }
.doc p { margin: 0; font-size: 15px; line-height: 1.65; }

.sec { margin-top: 22px; }
.sec > label { display: block; font-size: 11px; text-transform: uppercase;
  letter-spacing: .09em; color: var(--gray); font-weight: 700; margin-bottom: 8px; }
.btns { display: flex; gap: 10px; flex-wrap: wrap; }
button.opt {
  font: inherit; padding: 11px 18px; border-radius: 10px; cursor: pointer;
  border: 2px solid var(--line); background: var(--card); color: var(--ink);
}
button.opt kbd { font: 11px ui-monospace, Menlo, monospace; color: var(--gray);
  margin-left: 7px; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; }
button.opt[aria-pressed="true"] { border-color: var(--purple);
  background: var(--purpleSoft); color: var(--purple); font-weight: 700; }
button.opt.g0[aria-pressed="true"] { border-color: var(--red);
  background: var(--redSoft); color: var(--red); }
button.opt.g2[aria-pressed="true"] { border-color: var(--green);
  background: var(--greenSoft); color: var(--green); }
button.opt:disabled { opacity: .4; cursor: not-allowed; }

.nav { position: fixed; bottom: 0; left: 380px; right: 0;
  background: var(--card); border-top: 2px solid var(--line);
  padding: 12px 32px; display: flex; gap: 12px; align-items: center; }
.nav button { font: inherit; padding: 10px 18px; border-radius: 10px;
  border: 2px solid var(--line); background: var(--card); cursor: pointer; }
.nav button.primary { background: var(--purple); border-color: var(--purple);
  color: #fff; font-weight: 700; }
.nav .spacer { flex: 1; }
.nav .hint { font-size: 12px; color: var(--gray); }
.flag { background: var(--amberSoft); border-color: #E6D5A8; color: var(--amber); }
.flag[aria-pressed="true"] { background: var(--amber); color: #fff; border-color: var(--amber); }
.saved { font-size: 12px; color: var(--green); font-weight: 700; }
.note { width: 100%; font: inherit; padding: 10px 12px; border-radius: 10px;
  border: 2px solid var(--line); resize: vertical; min-height: 54px; }
</style>
</head>
<body>
<div id="app">
  <aside id="side"></aside>
  <main>
    <div class="bar">
      <select id="qsel"></select>
      <div class="prog"><span id="bar"></span></div>
      <div class="count" id="count"></div>
    </div>
    <div class="doc">
      <div class="id" id="docid"></div>
      <h3 id="title"></h3>
      <p id="abstract"></p>
    </div>
    <div class="sec">
      <label>Relevance grade</label>
      <div class="btns" id="grades"></div>
    </div>
    <div class="sec" id="facetsec">
      <label>Facets &mdash; which does this paper discuss?</label>
      <div class="btns" id="facets"></div>
    </div>
    <div class="sec">
      <label>Note (optional)</label>
      <textarea class="note" id="note" placeholder="Why this call, if it was close"></textarea>
    </div>
  </main>
</div>
<div class="nav">
  <button id="prev">&larr; Prev</button>
  <button id="next" class="primary">Save &amp; next &rarr;</button>
  <button id="flag" class="opt flag" aria-pressed="false">Flag for review</button>
  <span class="saved" id="saved"></span>
  <div class="spacer"></div>
  <span class="hint">0 1 2 grade &middot; q w e r facets &middot; Enter next</span>
  <button id="export">Export JSON</button>
  <button id="import">Import</button>
  <input type="file" id="file" accept=".json" hidden>
</div>

<script>
const DATA = /*__DATA__*/;
const KEY  = "__KEY__";

let qi = 0, di = 0;
let store = JSON.parse(localStorage.getItem(KEY) || "{}");

const el = id => document.getElementById(id);
const Q  = () => DATA[qi];
const D  = () => Q().docs[di];
const slot = () => `${Q().query_id}|${D().doc_id}`;
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function persist() {
  localStorage.setItem(KEY, JSON.stringify(store));
  el("saved").textContent = "saved";
  setTimeout(() => el("saved").textContent = "", 900);
}

function renderSide() {
  const q = Q();
  const n = q.narrative || {};
  el("side").innerHTML = `
    <h1>${esc(q.query)}</h1>
    <div class="src">facets from ${esc(q.source.arxiv_id)} ${esc(q.source.sections || "")}</div>
    <h2>Facets</h2>
    ${q.facets.map(f => `<div class="facetdef"><b>${f.id} &middot; ${esc(f.label)}</b>
      <p>${esc(f.definition)}</p></div>`).join("")}
    <h2>Narrative</h2>
    ${n.user ? `<div class="narr"><b>User</b>${esc(n.user)}</div>` : ""}
    ${n.relevant ? `<div class="narr"><b>Relevant</b>${esc(n.relevant)}</div>` : ""}
    ${n.non_relevant ? `<div class="narr no"><b>Not relevant</b>${esc(n.non_relevant)}</div>` : ""}
    <h2>Rules</h2>
    <ol>${q.rules.map(r => `<li>${esc(r)}</li>`).join("")}</ol>`;
}

function renderDoc() {
  const q = Q(), d = D(), rec = store[slot()] || {};

  el("docid").textContent = d.doc_id;
  el("title").textContent = d.title;
  el("abstract").textContent = d.abstract;
  el("note").value = rec.note || "";
  el("flag").setAttribute("aria-pressed", rec.flag ? "true" : "false");

  el("grades").innerHTML = [
    [0, "Not relevant", "g0"], [1, "Narrow", "g1"], [2, "Substantive", "g2"],
  ].map(([g, label, cls]) =>
    `<button class="opt ${cls}" data-g="${g}"
      aria-pressed="${rec.grade === g}">${g} &middot; ${label}<kbd>${g}</kbd></button>`
  ).join("");

  // Grade 0 means no facets by construction, so the controls are disabled
  // rather than merely cleared — it removes a whole class of slip.
  const off = rec.grade === 0 || rec.grade === undefined;
  const keys = ["q", "w", "e", "r"];
  el("facets").innerHTML = q.facets.map((f, i) =>
    `<button class="opt" data-f="${i}" ${off ? "disabled" : ""}
      aria-pressed="${(rec.facets || [])[i] ? "true" : "false"}"
      >${f.id} &middot; ${esc(f.label)}<kbd>${keys[i]}</kbd></button>`
  ).join("");

  const done = q.docs.filter(x => store[`${q.query_id}|${x.doc_id}`]?.grade !== undefined).length;
  el("count").textContent = `${di + 1} / ${q.docs.length}  ·  ${done} judged`;
  el("bar").style.width = (100 * done / q.docs.length) + "%";
}

function setGrade(g) {
  const rec = store[slot()] || {};
  rec.grade = g;
  if (g === 0) rec.facets = Q().facets.map(() => 0);
  else if (!rec.facets) rec.facets = Q().facets.map(() => 0);
  rec.ts = Date.now();
  store[slot()] = rec;
  persist(); renderDoc();
}

function toggleFacet(i) {
  const rec = store[slot()];
  if (!rec || rec.grade === 0 || rec.grade === undefined) return;
  rec.facets = rec.facets || Q().facets.map(() => 0);
  rec.facets[i] = rec.facets[i] ? 0 : 1;
  store[slot()] = rec;
  persist(); renderDoc();
}

function saveNote() {
  const rec = store[slot()] || {};
  const v = el("note").value.trim();
  if (v) rec.note = v; else delete rec.note;
  store[slot()] = rec; persist();
}

function go(delta) {
  saveNote();
  const n = Q().docs.length;
  if (di + delta < 0) { if (qi > 0) { qi--; di = DATA[qi].docs.length - 1; renderSide(); } }
  else if (di + delta >= n) {
    if (qi < DATA.length - 1) { qi++; di = 0; renderSide(); el("qsel").value = qi; }
  } else di += delta;
  renderDoc(); window.scrollTo(0, 0);
}

el("grades").addEventListener("click", e => {
  const b = e.target.closest("[data-g]"); if (b) setGrade(+b.dataset.g);
});
el("facets").addEventListener("click", e => {
  const b = e.target.closest("[data-f]"); if (b) toggleFacet(+b.dataset.f);
});
el("flag").onclick = () => {
  const rec = store[slot()] || {};
  rec.flag = !rec.flag; store[slot()] = rec; persist(); renderDoc();
};
el("prev").onclick = () => go(-1);
el("next").onclick = () => go(1);

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key === "0" || e.key === "1" || e.key === "2") setGrade(+e.key);
  else if ("qwer".includes(e.key)) toggleFacet("qwer".indexOf(e.key));
  else if (e.key === "Enter") go(1);
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "f") el("flag").click();
});

el("export").onclick = () => {
  const rows = [];
  for (const q of DATA) for (const d of q.docs) {
    const r = store[`${q.query_id}|${d.doc_id}`];
    if (r && r.grade !== undefined) rows.push({
      query_id: q.query_id, doc_id: d.doc_id, grade: r.grade,
      facets: r.facets || [], flag: !!r.flag, note: r.note || "", ts: r.ts,
    });
  }
  const blob = new Blob([JSON.stringify(
    {annotator: KEY, exported: new Date().toISOString(), n: rows.length, judgments: rows},
    null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = KEY + "-" + new Date().toISOString().slice(0, 10) + ".json";
  a.click();
};

el("import").onclick = () => el("file").click();
el("file").onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  f.text().then(t => {
    const d = JSON.parse(t);
    for (const r of d.judgments) store[`${r.query_id}|${r.doc_id}`] =
      {grade: r.grade, facets: r.facets, flag: r.flag, note: r.note, ts: r.ts};
    persist(); renderDoc();
    alert(`Imported ${d.judgments.length} judgments.`);
  });
};

el("qsel").innerHTML = DATA.map((q, i) =>
  `<option value="${i}">${q.query_id} · ${esc(q.query).slice(0, 44)}</option>`).join("");
el("qsel").onchange = e => { saveNote(); qi = +e.target.value; di = 0; renderSide(); renderDoc(); };

renderSide(); renderDoc();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["self", "peer"], default="self")
    ap.add_argument("--fraction", type=float, default=0.2,
                    help="Share of each pool to sample for the peer build.")
    args = ap.parse_args()

    peer = args.mode == "peer"
    payload = build_payload(args.fraction if peer else 1.0, PEER_SEED)

    html = (TEMPLATE
            .replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False))
            .replace("__KEY__", "ir-peer" if peer else "ir-self")
            .replace("__TITLE__", "Annotation — peer sample" if peer
                     else "Annotation"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / ("annotate_peer.html" if peer else "annotate.html")
    out.write_text(html, encoding="utf-8")

    n = sum(len(q["docs"]) for q in payload)
    print(f"{out}  —  {len(payload)} queries, {n} documents"
          f"{f' ({args.fraction:.0%} sample)' if peer else ''}")
    print(f"size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()