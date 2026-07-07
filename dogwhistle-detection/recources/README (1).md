# `dog_whistle_definitions/`

Working directory for enriching the `silent_signals` informal subset with the
Allen AI dog whistle glossary. The original `PLAN.md` assumed we would scrape
the glossary from the web; that step is no longer needed because the glossary
is already available as a local markdown dump
(`Glossary_of_Dogwhistles.md`). `extract_definitions.py` parses that markdown
directly.

---

## What happened in this directory

1. `Glossary_of_Dogwhistles.md` (340 entries) was dropped in alongside the
   dataset-side inventory files (`dog_whistle_roots.csv`,
   `dog_whistle_roots_list.txt`, `dog_whistle_canonical_forms.csv`).
2. `extract_definitions.py` was written to:
   - parse each entry into structured fields (term, surface forms,
     persona/ingroup, covert meaning, type, register, description, example,
     speaker, date);
   - write the parsed table to `allen_glossary.csv`;
   - left-join it onto `dog_whistle_roots.csv` via a normalisation ladder
     (exact → casefold → unicode/whitespace-normalised → surface-form lookup →
     alphanumeric-only fallback);
   - emit `dog_whistle_roots_enriched.csv` with every original column plus ten
     `glossary_*` columns and a `match_method` audit column.
3. The script was run once. Match outcome: **298 / 298 roots resolved**
   (297 exact, 1 alphanum — the roots CSV has `"majority-minorit y"` with a
   stray space; the glossary term is `"majority-minority"`). The existing
   `definition` column agrees with `glossary_covert_meaning` on 296 / 298
   rows; the 2 divergences are genuine content differences, not parse bugs.

---

## File inventory

### Inputs (pre-existing)

| File | What it is |
|---|---|
| `Glossary_of_Dogwhistles.md` | Allen AI glossary, 340 dog-whistle entries in markdown. Source of truth for definitions. |
| `dog_whistle_roots.csv` | Dataset-side inventory: 298 roots with `n_canonical`, `canonical_forms` (pipe-separated surface forms), `n_occurrences`, `ingroup`, `definition`. |
| `dog_whistle_roots_list.txt` | Plain list of the 298 roots, one per line. |
| `dog_whistle_canonical_forms.csv` | 696 surface forms (one per row), no metadata. |
| `PLAN.md` | Original scrape-and-merge plan. Kept for reference; the scraping steps are now obsolete. |

### Outputs (produced by `extract_definitions.py`)

| File | Rows | What it is |
|---|---|---|
| `allen_glossary.csv` | 340 | One row per glossary term. Columns: `term, surface_forms, ingroup, covert_meaning, type, register, description, example, speaker, date`. `surface_forms` is `; `-separated. |
| `dog_whistle_roots_enriched.csv` | 298 | `dog_whistle_roots.csv` + joined glossary fields. Adds: `glossary_term, glossary_surface_forms, glossary_ingroup, glossary_covert_meaning, glossary_type, glossary_register, glossary_description, glossary_example, glossary_speaker, glossary_date, match_method`. |

### Code

| File | What it does |
|---|---|
| `extract_definitions.py` | Parses the glossary markdown and writes both outputs above. Idempotent: rerun any time. No network, no dependencies beyond the Python stdlib. |

---

## How to use

### Regenerate the outputs

```bash
cd dog_whistle_definitions
python3 extract_definitions.py
```

Expected console output:

```
wrote allen_glossary.csv: 340 rows
wrote dog_whistle_roots_enriched.csv: 298 rows
match stats: {'exact': 297, 'casefold': 0, 'normalised': 0, 'surface_form': 0, 'alphanum': 1, 'UNRESOLVED': 0}
```

If `UNRESOLVED` is ever non-zero, the listed roots are printed and need to be
reconciled against the glossary by hand (add a manual override, or fix the
typo upstream).

### The link chain

```
canonical_form  ──┐
                  ├──► dog_whistle_root ──► glossary_term ──► glossary fields
dog_whistle_root ─┘         (join key,       (match_method
                             298 rows)        records how)
```

- `canonical_forms` column in `dog_whistle_roots.csv` lists the surface forms
  that collapse to each root (pipe-separated).
- `dog_whistle_root` is the join key.
- `match_method` in the enriched CSV tells you how the join was resolved for
  each row (`exact` for 297, `alphanum` for 1).

### Using the enriched CSV for RQ-A

Per `PLAN.md` §6, the two model variants are:

| Variant | `Candidate meaning` field |
|---|---|
| Baseline | `definition` (short gloss, already in `silent_signals`) |
| Enriched | `glossary_description` (paragraph, from this directory) |

To attach these columns to `silent_signals`, left-join the informal parquet
on `dog_whistle_root` against `dog_whistle_roots_enriched.csv`.

### Producing a surface-form-level long-format table

Not currently written. If needed, explode `canonical_forms` in the enriched
CSV — one row per `(surface_form, root, glossary_*)` triple. Two-line pandas
or csv snippet; ask if you want it as a permanent artefact.

---

## Notes and caveats

- **Not obsolete, just superseded:** `PLAN.md` still documents the reasoning
  for why we want these fields and how they feed RQ-A. Only the scraping
  mechanics are stale.
- **No manual overrides file exists.** The single imperfect match (the stray
  space in `majority-minorit y`) is absorbed by the `alphanum` fallback
  tier. If future glossary drift introduces new unresolved roots, consider
  adding a `manual_overrides.csv` (root → glossary_term) and having the
  script read it before the ladder.
- **Divergences between `definition` and `glossary_covert_meaning`** (2 rows)
  are worth eyeballing — they're the kind of edge case that makes good error
  analysis material, not a parser bug.
- **No HTML cache, no network calls.** Everything is reproduced from the
  local markdown, so the pipeline is deterministic and offline.
