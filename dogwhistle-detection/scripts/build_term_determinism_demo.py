"""Build the lookup that powers the Space's educational tab.

For each grouped-split TEST root (the ones Run A never saw during training),
record:
  - the gold ingroup (from the glossary / corpus annotation),
  - whether that same root appears in the ALT-split TRAIN (so Run C did see
    it — that's the term-determinism leak the report defends),
  - a representative test-set comment containing the root.

Output → data/manifests/term_determinism_demo.json. The Space's Tab 2
loads this and uses it to drive the side-by-side "Run A vs Run C"
demonstration without needing the parquets shipped into the Space repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPED = ROOT / "data" / "final" / "rq_b_multiclass"
ALT = ROOT / "data" / "final" / "rq_b_multiclass_random"
OUT = ROOT / "data" / "manifests" / "term_determinism_demo.json"


def _pick_sample(df: pd.DataFrame, root: str) -> dict:
    rows = df[df["dog_whistle_root"] == root]
    rows = rows[rows["content"].astype(str).str.len().between(40, 350)]
    if rows.empty:
        rows = df[df["dog_whistle_root"] == root].head(1)
    rows = rows.head(1)
    if rows.empty:
        return {}
    r = rows.iloc[0]
    return {
        "content": str(r["content"]),
        "surface_form": str(r["dog_whistle"]),
        "gold_ingroup": str(r["ingroup"]),
        "subreddit": str(r.get("subreddit", "")) or None,
    }


def main():
    test_grouped = pd.read_parquet(GROUPED / "test.parquet")
    train_alt = pd.read_parquet(ALT / "train.parquet")
    test_alt = pd.read_parquet(ALT / "test.parquet")

    test_grouped_roots = sorted(test_grouped["dog_whistle_root"].unique().tolist())
    alt_train_roots = set(train_alt["dog_whistle_root"].unique().tolist())
    alt_test_roots = set(test_alt["dog_whistle_root"].unique().tolist())

    entries = []
    for root in test_grouped_roots:
        sample = _pick_sample(test_grouped, root)
        if not sample:
            continue
        entries.append({
            "root": root,
            "gold_ingroup": sample["gold_ingroup"],
            "surface_form_in_sample": sample["surface_form"],
            "in_alt_train": root in alt_train_roots,
            "in_alt_test": root in alt_test_roots,
            "sample_comment": sample["content"],
            "sample_subreddit": sample.get("subreddit"),
        })

    out = {
        "schema_version": 1,
        "explainer": (
            "Each row is a dog-whistle root that lives in the grouped-split "
            "TEST set — Run A's training data never contained this root, so "
            "any RQB Run A prediction here is a cold-start generalisation. "
            "`in_alt_train` flags whether the same root was in the alt-split "
            "(ingroup-stratified) TRAIN data, which is what gave Run C the "
            "chance to memorise the term→ingroup glossary mapping. The "
            "0.996 macro-F1 of Run C is built on this leak."
        ),
        "n_grouped_test_roots": len(test_grouped_roots),
        "n_with_sample": len(entries),
        "n_in_alt_train": sum(1 for e in entries if e["in_alt_train"]),
        "n_in_alt_test_too": sum(1 for e in entries
                                 if e["in_alt_train"] and e["in_alt_test"]),
        "entries": entries,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Wrote {OUT}")
    print(f"  {len(entries)} grouped-test roots with sample comments")
    print(f"  {out['n_in_alt_train']} of them ALSO live in alt-split TRAIN "
          "(Run C saw these during training; Run A never did)")
    print(f"  {out['n_in_alt_test_too']} are in BOTH alt-train AND alt-test")


if __name__ == "__main__":
    main()
