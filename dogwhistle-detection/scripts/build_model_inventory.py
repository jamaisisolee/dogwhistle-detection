"""Walk results/{binary,multiclass,generation,generation_balanced}/ and emit
a manifest of every trained variant + headline metrics.

Output → data/manifests/model_inventory.json. The manifest is the single
source of truth for the HF Space dropdowns (steps 4–5) and the upload
script (step 2). Re-run any time results change.

What lives where (as of 2026-04-29):
  - RQA (binary):   2 arms × 3 seeds = 6 variants. Local + HPC.
  - RQB (multiclass): 6 variants across 4 conditions (term×3 seeds,
                      text_only×1, term_altsplit×1, term_weighted×1).
                      test_results.json local; best_model on HPC only.
  - RQC (generation): Path A (3 seeds) — local results, HPC checkpoints.
                      Path B (3 seeds) — HPC only, results not pulled yet.

Headline metrics (best variant chosen by these):
  - RQA: disambiguation_124_f1 (locked human eval — most defensible).
  - RQB: f1_macro on the grouped-split test (held-out roots).
  - RQC: ingroup.macro_f1 on the grouped-split test.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HPC_PROJECT = "~/dogwhistle_project"


# ---------- helpers --------------------------------------------------------

def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _local_to_hpc(local_path: Path) -> str:
    """Map a local results/... path to the HPC layout (results/... is
    relative to ~/dogwhistle_project on HPC; the prefix is the same)."""
    rel = local_path.relative_to(ROOT).as_posix()
    return f"{HPC_PROJECT}/{rel}"


def _round(v, digits=4):
    return round(float(v), digits) if isinstance(v, (int, float)) else v


# ---------- per-task variant builders --------------------------------------

def collect_rqa() -> list[dict]:
    """RQA = binary disambiguator. 2 arms × 3 seeds."""
    variants = []
    for arm_dir in sorted((RESULTS / "binary").glob("format_*")):
        arm = arm_dir.name.removeprefix("format_")
        for seed_dir in sorted(arm_dir.glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            r = _load(seed_dir / "test_results.json")
            if r is None:
                continue
            variants.append({
                "id": f"rqa_{arm}_seed{seed}",
                "task": "rqa",
                "arm": arm,
                "seed": seed,
                "artifact_local": str((seed_dir / "best_model").relative_to(ROOT)),
                "artifact_hpc": _local_to_hpc(seed_dir / "best_model"),
                "predictions_local": str((seed_dir / "predictions").relative_to(ROOT))
                                       if (seed_dir / "predictions").exists() else None,
                "hf_branch": f"{arm}-seed{seed}".replace("_", "-"),
                "metrics": {k: _round(r.get(k)) for k in (
                    "test_split_f1", "test_split_accuracy",
                    "detection_101_f1", "detection_101_accuracy",
                    "disambiguation_124_f1", "disambiguation_124_accuracy",
                ) if k in r},
            })
    return variants


def collect_rqb() -> list[dict]:
    """RQB = multiclass ingroup. Variants are nested under
    format_<arm>[_<runtag>]/seed_<n>/. Run-tags: '' (Run A/B), 'altsplit'
    (Run C), 'weighted' (Run D).
    """
    variants = []
    for fmt_dir in sorted((RESULTS / "multiclass").glob("format_*")):
        # Decompose 'format_term', 'format_text_only', 'format_term_altsplit',
        # 'format_term_weighted' into (arm, run_tag).
        name = fmt_dir.name.removeprefix("format_")
        if name.endswith("_altsplit"):
            arm, run_tag = name.removesuffix("_altsplit"), "altsplit"
        elif name.endswith("_weighted"):
            arm, run_tag = name.removesuffix("_weighted"), "weighted"
        else:
            arm, run_tag = name, ""
        for seed_dir in sorted(fmt_dir.glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            r = _load(seed_dir / "test_results.json")
            if r is None:
                continue
            id_parts = ["rqb", arm]
            if run_tag:
                id_parts.append(run_tag)
            id_parts.append(f"seed{seed}")
            variants.append({
                "id": "_".join(id_parts),
                "task": "rqb",
                "arm": arm,
                "run_tag": run_tag or None,
                "seed": seed,
                "artifact_local": str((seed_dir / "best_model").relative_to(ROOT)),
                "artifact_hpc": _local_to_hpc(seed_dir / "best_model"),
                "predictions_local": str((seed_dir / "predictions").relative_to(ROOT))
                                       if (seed_dir / "predictions").exists() else None,
                "hf_branch": ("-".join(id_parts[1:])).replace("_", "-"),
                "metrics": {
                    "f1_macro": _round(r.get("f1_macro")),
                    "f1_weighted": _round(r.get("f1_weighted")),
                    "accuracy": _round(r.get("accuracy")),
                    "n_test": int(r.get("n_test", 0)),
                },
            })
    return variants


def collect_rqc() -> list[dict]:
    """RQC = Flan-T5-XL + LoRA structured generator.
    Path A unbalanced: results/generation/seed_*/.
    Path B balanced:   results/generation_balanced/seed_*/ (HPC only — may be missing locally).
    """
    variants = []
    for path_label, subdir in (("A", "generation"), ("B", "generation_balanced")):
        base = RESULTS / subdir
        if not base.exists():
            # Stub the missing path so the manifest stays exhaustive even
            # when only Path A has been pulled locally.
            for seed in (42, 123, 7):
                variants.append({
                    "id": f"rqc_path{path_label}_seed{seed}",
                    "task": "rqc",
                    "path": path_label,
                    "seed": seed,
                    "is_lora_adapter": True,
                    "base_model": "google/flan-t5-xl",
                    "artifact_local": None,
                    "artifact_hpc": f"{HPC_PROJECT}/results/{subdir}/seed_{seed}",
                    "predictions_local": None,
                    "hf_branch": f"path{path_label.lower()}-seed{seed}",
                    "metrics": None,
                    "status": "hpc_only_pull_pending",
                })
            continue
        for seed_dir in sorted(base.glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            r = _load(seed_dir / "test_results.json")
            if r is None:
                continue
            ig = r.get("ingroup") or {}
            root = r.get("dog_whistle_root") or {}
            defn = r.get("definition") or {}
            # RQC artefact = the LoRA adapter dir specifically (not the seed
            # dir, which carries intermediate checkpoints + eval artefacts).
            # The Inference API expects adapter_config.json at the branch
            # root; nesting it under best_model/ would break adapter loading.
            adapter_dir = seed_dir / "best_model"
            variants.append({
                "id": f"rqc_path{path_label}_seed{seed}",
                "task": "rqc",
                "path": path_label,
                "seed": seed,
                "is_lora_adapter": True,
                "base_model": "google/flan-t5-xl",
                "artifact_local": str(adapter_dir.relative_to(ROOT)),
                "artifact_hpc": _local_to_hpc(adapter_dir),
                "predictions_local": (
                    str((seed_dir / "test_predictions.parquet").relative_to(ROOT))
                    if (seed_dir / "test_predictions.parquet").exists() else None
                ),
                "hf_branch": f"path{path_label.lower()}-seed{seed}",
                "metrics": {
                    "json_parse_rate": _round(r.get("json_parse_rate")),
                    "ingroup.accuracy": _round(ig.get("accuracy")),
                    "ingroup.macro_f1": _round(ig.get("macro_f1")),
                    "ingroup.weighted_f1": _round(ig.get("weighted_f1")),
                    "dog_whistle_root.macro_f1": _round(root.get("macro_f1")),
                    "definition.macro_f1": _round(defn.get("macro_f1")),
                    "explanation_cosine_mean": _round(r.get("explanation_cosine_mean")),
                    "explanation_rouge_l_mean": _round(r.get("explanation_rouge_l_mean")),
                    "n_samples": int(r.get("n_samples", 0)),
                },
            })
    return variants


# ---------- best-variant selection -----------------------------------------

HEADLINE = {
    "rqa": ("disambiguation_124_f1",
            "F1 on the 124-row locked human-eval disambiguation set "
            "(see DESIGN_DEFENSE.md D7)."),
    "rqb": ("f1_macro",
            "macro-F1 on the grouped-split test (held-out roots; "
            "see docs/rq_b_report.md § 4.1)."),
    "rqc": ("ingroup.macro_f1",
            "ingroup macro-F1 on the grouped-split test "
            "(see docs/rq_c_report.md)."),
}

# Trained label spaces. Saved configs were stored with generic LABEL_N
# names (training scripts didn't pass id2label to from_pretrained), so
# we surface the human labels here. RQB labels mirror the alphabetical
# ordering used by the training script (sorted(train_ingroups)).
TASK_LABELS = {
    "rqa": {
        "0": "literal",
        "1": "coded",
    },
    "rqb": {
        "0": "Islamophobic", "1": "anti-Asian", "2": "anti-GMO",
        "3": "anti-LGBTQ", "4": "anti-Latino", "5": "anti-liberal",
        "6": "anti-vax", "7": "antisemitic",
        "8": "climate change denier", "9": "conservative",
        "10": "homophobic", "11": "liberal", "12": "racist",
        "13": "religious", "14": "transphobic",
        "15": "white supremacist", "16": "xenophobic",
    },
    # RQC is generative — labels live inside the JSON output, not in
    # an id2label map.
}


def pick_best(variants: list[dict], metric_key: str) -> str | None:
    """Highest single-seed score across all variants. Raw leader."""
    candidates = [(v["metrics"].get(metric_key), v["id"])
                  for v in variants
                  if v.get("metrics") and v["metrics"].get(metric_key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


# `default_variant` ≠ `best_variant`. The raw leader is sometimes a sensitivity
# check (RQB altsplit), an unlucky-arm seed peak, or a single-seed ablation. The
# demo defaults must match what the per-RQ reports defend as the headline
# system. The gap between best and default is itself a research artifact the
# Space surfaces (Step 6 educational tab).

def pick_default_rqa(variants: list[dict]) -> tuple[str | None, str]:
    """Best-mean arm, highest seed within it.

    Single-seed peaks are noisy; per-arm means are what the report defends
    (term = 0.707 ± 0.015 vs enriched-def). Pick the arm whose mean is best,
    then the strongest seed inside that arm.
    """
    by_arm: dict[str, list[tuple[float, str]]] = {}
    for v in variants:
        m = (v.get("metrics") or {}).get("disambiguation_124_f1")
        if m is None:
            continue
        by_arm.setdefault(v["arm"], []).append((m, v["id"]))
    if not by_arm:
        return None, ""
    arm_mean = {a: sum(m for m, _ in s) / len(s) for a, s in by_arm.items()}
    best_arm = max(arm_mean, key=arm_mean.get)
    chosen = max(by_arm[best_arm], key=lambda t: t[0])[1]
    return chosen, (
        f"Picked the arm with the highest mean disambiguation_124_f1 across seeds "
        f"({best_arm} = {arm_mean[best_arm]:.4f}), then the best seed within. "
        f"Defends a per-arm comparison, not a single-seed peak. See "
        f"docs/rq_a_report.md."
    )


def pick_default_rqb(variants: list[dict]) -> tuple[str | None, str]:
    """Headline Run A only: term arm, grouped split, no run_tag.

    Run C (altsplit) hits 0.996 because every dog-whistle term in
    silent_signals deterministically maps to one ingroup in the Allen AI
    glossary; under the alt (ingroup-stratified) split, train roots leak
    into test, so the model memorises the glossary. Defaulting to Run C
    would invert the central finding of docs/rq_b_report.md § 4.1 — the
    GAP between Run A (0.353) and Run C (0.996) is the result.

    Run D (weighted-CE) underperforms Run A; Run B (text_only) is an input
    ablation. Default is the best of the three Run-A term/grouped seeds.
    """
    candidates = [
        (v["metrics"]["f1_macro"], v["id"])
        for v in variants
        if (v.get("metrics") or {}).get("f1_macro") is not None
        and v.get("run_tag") is None
        and v.get("arm") == "term"
    ]
    if not candidates:
        return None, ""
    chosen = max(candidates, key=lambda t: t[0])[1]
    return chosen, (
        "Restricted to Run A (term arm, grouped split, no run_tag) — the "
        "headline system in docs/rq_b_report.md. Run C (altsplit, 0.996) is "
        "excluded: it's a glossary-determinism artefact, not generalization, "
        "and defaulting to it would misrepresent RQ-B's central finding. The "
        "Run C model remains selectable in the dropdown for the educational "
        "tab (Step 6)."
    )


def pick_default_rqc(variants: list[dict]) -> tuple[str | None, str]:
    """Path A (unbalanced), highest seed by ingroup.macro_f1.

    Per docs/rq_c_report.md § 4, Path A is the more defensible system:
    Path B (class-balanced upsampling) overweights minority classes
    without enough signal to learn them.
    """
    candidates = [
        (v["metrics"]["ingroup.macro_f1"], v["id"])
        for v in variants
        if v.get("path") == "A"
        and (v.get("metrics") or {}).get("ingroup.macro_f1") is not None
    ]
    if not candidates:
        return None, ""
    chosen = max(candidates, key=lambda t: t[0])[1]
    return chosen, (
        "Restricted to Path A (unbalanced) — the defensible system per "
        "docs/rq_c_report.md § 4. Path B variants stay in the dropdown for "
        "comparison but aren't the default."
    )


# ---------- main -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/manifests/model_inventory.json")
    p.add_argument("--hf_user", default="calerio",
                   help="HF Hub username under which the dogwhistle-{rqa,rqb,rqc} "
                        "repos live.")
    args = p.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rqa = collect_rqa()
    rqb = collect_rqb()
    rqc = collect_rqc()

    rqa_default, rqa_default_why = pick_default_rqa(rqa)
    rqb_default, rqb_default_why = pick_default_rqb(rqb)
    rqc_default, rqc_default_why = pick_default_rqc(rqc)

    inventory = {
        "schema_version": 2,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "hf_user": args.hf_user,
        "hf_repo_template": f"{args.hf_user}/silent-signals-{{task}}",
        "field_doc": {
            "best_variant": "Raw single-seed leader by headline_metric. "
                            "Sometimes a sensitivity-check or lucky seed; "
                            "NOT necessarily what the demo ships.",
            "default_variant": "What the Space defaults to in the dropdown. "
                               "Chosen by the per-task rules in build_model_inventory.py "
                               "and defended in docs/hf_space_report.md.",
        },
        "tasks": {
            "rqa": {
                "description": "Binary disambiguator — coded vs literal use of a candidate dog-whistle term. RoBERTa-base.",
                "model_family": "FacebookAI/roberta-base",
                "headline_metric": HEADLINE["rqa"][0],
                "headline_metric_doc": HEADLINE["rqa"][1],
                "labels": TASK_LABELS["rqa"],
                "best_variant": pick_best(rqa, HEADLINE["rqa"][0]),
                "default_variant": rqa_default,
                "default_variant_rationale": rqa_default_why,
                "variants": rqa,
            },
            "rqb": {
                "description": "Multiclass ingroup classifier (17 classes; train-only label space). RoBERTa-base.",
                "model_family": "FacebookAI/roberta-base",
                "headline_metric": HEADLINE["rqb"][0],
                "headline_metric_doc": HEADLINE["rqb"][1],
                "labels": TASK_LABELS["rqb"],
                "best_variant": pick_best(rqb, HEADLINE["rqb"][0]),
                "default_variant": rqb_default,
                "default_variant_rationale": rqb_default_why,
                "variants": rqb,
            },
            "rqc": {
                "description": "Structured generator — produces (root, ingroup, definition, explanation) JSON. Flan-T5-XL + LoRA.",
                "model_family": "google/flan-t5-xl",
                "is_lora": True,
                "headline_metric": HEADLINE["rqc"][0],
                "headline_metric_doc": HEADLINE["rqc"][1],
                "best_variant": pick_best(rqc, HEADLINE["rqc"][0]),
                "default_variant": rqc_default,
                "default_variant_rationale": rqc_default_why,
                "variants": rqc,
            },
        },
        "totals": {
            "rqa": len(rqa), "rqb": len(rqb), "rqc": len(rqc),
            "all": len(rqa) + len(rqb) + len(rqc),
        },
    }

    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2)

    # Console summary
    print(f"Wrote {out_path}")
    print(f"\n=== Variant counts ===")
    for t in ("rqa", "rqb", "rqc"):
        info = inventory["tasks"][t]
        print(f"  {t}: {len(info['variants'])} variants  "
              f"best={info['best_variant']}  default={info['default_variant']}")

    print(f"\n=== Raw leader vs demo default ===")
    for t in ("rqa", "rqb", "rqc"):
        info = inventory["tasks"][t]
        metric_key = info["headline_metric"]
        for label, vid in (("best", info["best_variant"]),
                           ("default", info["default_variant"])):
            if vid is None:
                print(f"  {t} {label}: (none — all variants HPC-only?)")
                continue
            v = next(v for v in info["variants"] if v["id"] == vid)
            m = v["metrics"][metric_key]
            print(f"  {t} {label:7s}: {vid}  ({metric_key}={m})")
        print()


if __name__ == "__main__":
    main()
