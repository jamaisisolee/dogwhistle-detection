#!/usr/bin/env python3
"""
Parse Glossary_of_Dogwhistles.md and produce:
  1. allen_glossary.csv           — 340 rows, one per glossary term.
  2. dog_whistle_roots_enriched.csv — dog_whistle_roots.csv + joined glossary
                                      fields via a normalised match ladder.

No scraping, no network. Reads the markdown in place.
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
MD = HERE / "Glossary_of_Dogwhistles.md"
ROOTS = HERE / "dog_whistle_roots.csv"
OUT_GLOSSARY = HERE / "allen_glossary.csv"
OUT_ENRICHED = HERE / "dog_whistle_roots_enriched.csv"

GLOSSARY_FIELDS = [
    "term",
    "surface_forms",
    "ingroup",
    "covert_meaning",
    "type",
    "register",
    "description",
    "example",
    "speaker",
    "date",
]

HEADER_RE = re.compile(r"^\*\*(.+)\*\*\s*$")
FIELD_RE = {
    "ingroup": re.compile(r"_Persona/In-Group_:\s*([^\n\\]+)"),
    "covert_meaning": re.compile(r"_Covert \(in-group\) meaning_:\s*([^\n\\]+)"),
    "type": re.compile(r"_Type_:\s*([^\n\\]+)"),
    "register": re.compile(r"_Register_:\s*([^\n\\]+)"),
    "speaker": re.compile(r"_Speaker_:\s*([^\n\\]+)"),
    "date": re.compile(r"_Date_:\s*([^\n\\]+)"),
}


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().strip().lstrip("#")
    return re.sub(r"\s+", " ", s)


def alphanum(s: str) -> str:
    """Aggressive fallback: keep only [a-z0-9] after unicode+casefold."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[^a-z0-9]", "", s)


def clean_block(text: str) -> str:
    """Collapse indented multi-line markdown blocks into a single string."""
    lines = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        if ln.endswith("\\"):
            ln = ln[:-1]
        ln = ln.strip()
        if ln:
            lines.append(ln)
    return " ".join(lines).strip()


def parse_entry(block: list[str]) -> dict:
    term = HEADER_RE.match(block[0]).group(1).strip()
    body = "\n".join(block[1:])

    # Surface forms: from "_Surface forms_:" up to the blank line before "_Persona/..."
    sf_match = re.search(
        r"_Surface forms_:\s*(.+?)(?=\n\s*\n_Persona/In-Group_:)",
        body,
        re.DOTALL,
    )
    surface_raw = clean_block(sf_match.group(1)) if sf_match else ""
    # Separator is semicolon in the MD (a few use "; " — split on that).
    surface_list = [s.strip() for s in re.split(r"\s*;\s*", surface_raw) if s.strip()]

    fields = {"surface_forms": "; ".join(surface_list)}
    for key, rx in FIELD_RE.items():
        m = rx.search(body)
        fields[key] = m.group(1).strip() if m else ""

    # Description: between the "Description (from ...)" line and "Example context"
    desc_match = re.search(
        r"Description\s*\(from[^\n]*\n(.+?)(?=\n\s*Example context\b)",
        body,
        re.DOTALL,
    )
    fields["description"] = clean_block(desc_match.group(1)) if desc_match else ""

    # Example: between "Example context (...)" and "_Speaker_:"
    ex_match = re.search(
        r"Example context[^\n]*\n(.+?)(?=\n\s*_Speaker_:)",
        body,
        re.DOTALL,
    )
    fields["example"] = clean_block(ex_match.group(1)) if ex_match else ""

    return {"term": term, **fields}


def parse_glossary(text: str) -> list[dict]:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if HEADER_RE.match(ln)]
    entries = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        entries.append(parse_entry(lines[start:end]))
    return entries


def write_glossary(entries: list[dict]) -> None:
    with OUT_GLOSSARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GLOSSARY_FIELDS)
        w.writeheader()
        w.writerows(entries)


def build_lookups(entries: list[dict]):
    by_term = {e["term"]: e for e in entries}
    by_norm_term = {normalise(e["term"]): e for e in entries}
    by_alnum_term = {alphanum(e["term"]): e for e in entries}
    by_norm_surface: dict[str, dict] = {}
    for e in entries:
        for sf in e["surface_forms"].split(";"):
            key = normalise(sf)
            if key:
                by_norm_surface.setdefault(key, e)
    return by_term, by_norm_term, by_norm_surface, by_alnum_term


def match_root(root: str, by_term, by_norm_term, by_norm_surface, by_alnum_term):
    if root in by_term:
        return by_term[root], "exact"
    # casefold against all terms (small N)
    rcf = root.casefold()
    for term, e in by_term.items():
        if term.casefold() == rcf:
            return e, "casefold"
    n = normalise(root)
    if n in by_norm_term:
        return by_norm_term[n], "normalised"
    if n in by_norm_surface:
        return by_norm_surface[n], "surface_form"
    a = alphanum(root)
    if a and a in by_alnum_term:
        return by_alnum_term[a], "alphanum"
    return None, "UNRESOLVED"


def merge(entries: list[dict]) -> dict:
    by_term, by_norm_term, by_norm_surface, by_alnum_term = build_lookups(entries)

    with ROOTS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        roots = list(reader)

    glossary_cols = [f"glossary_{k}" for k in GLOSSARY_FIELDS]
    new_fields = orig_fields + glossary_cols + ["match_method"]

    stats = {"exact": 0, "casefold": 0, "normalised": 0, "surface_form": 0, "alphanum": 0, "UNRESOLVED": 0}
    unresolved: list[str] = []

    rows_out = []
    for row in roots:
        entry, method = match_root(
            row["dog_whistle_root"],
            by_term,
            by_norm_term,
            by_norm_surface,
            by_alnum_term,
        )
        stats[method] += 1
        if entry is None:
            unresolved.append(row["dog_whistle_root"])
            entry = {k: "" for k in GLOSSARY_FIELDS}
        merged = dict(row)
        for k in GLOSSARY_FIELDS:
            merged[f"glossary_{k}"] = entry.get(k, "")
        merged["match_method"] = method
        rows_out.append(merged)

    with OUT_ENRICHED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_fields)
        w.writeheader()
        w.writerows(rows_out)

    return {"n_rows": len(rows_out), "stats": stats, "unresolved": unresolved}


def main() -> int:
    text = MD.read_text(encoding="utf-8")
    entries = parse_glossary(text)

    if len(entries) != 340:
        print(f"WARN: expected 340 entries, parsed {len(entries)}", file=sys.stderr)

    # Sanity: required fields populated.
    missing = [
        e["term"]
        for e in entries
        if not (e["term"] and e["ingroup"] and e["covert_meaning"])
    ]
    if missing:
        print(f"WARN: {len(missing)} entries missing required fields: {missing[:5]}", file=sys.stderr)

    write_glossary(entries)
    print(f"wrote {OUT_GLOSSARY.name}: {len(entries)} rows")

    report = merge(entries)
    print(f"wrote {OUT_ENRICHED.name}: {report['n_rows']} rows")
    print(f"match stats: {report['stats']}")
    if report["unresolved"]:
        print(f"unresolved roots ({len(report['unresolved'])}):")
        for r in report["unresolved"]:
            print(f"  - {r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
