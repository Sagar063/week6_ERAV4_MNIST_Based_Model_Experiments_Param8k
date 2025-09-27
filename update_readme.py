#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update README.md in-place by replacing placeholder tags with generated content.

Placeholders supported (case-sensitive):
  <!-- DATA_STATS -->
  <!-- MODEL1_RESULTS -->
  <!-- MODEL2_RESULTS -->
  <!-- MODEL3_RESULTS -->
  <!-- CONSOLIDATED_RESULTS -->
  <!-- OBSERVATIONS -->
  <!-- ENV_INFO -->

Behavior:
- If README contains <!-- TAG --> ... <!-- /TAG -->, only the inner content is replaced.
- If README contains a single <!-- TAG -->, it will be expanded to a wrapped block:
    <!-- TAG -->
    ...generated...
    <!-- /TAG -->

Usage:
  python update_readme.py \
      --template README.base.md \
      --out README.md \
      --results results/summary.csv \
      --plots-dir results/plots

Notes:
- Expects a CSV "results/summary.csv" with columns like:
    exp_name, model, variant, params, best_train_acc, best_test_acc, epochs, notes
  If model/variant are missing, they are derived from exp_name (e.g., m2_light_bn -> model2, light_bn).
"""

import argparse
import csv
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pandas as pd
except Exception:
    pd = None

# -----------------------------
# Configurable column aliases
# -----------------------------
DEFAULT_COLS = dict(
    model="model",
    variant="variant",
    params="params",
    best_train_acc="best_train_acc",
    best_val_acc="best_val_acc",   # optional if you have validation
    best_test_acc="best_test_acc",
    epochs="epochs",
    notes="notes",
)

# ----------- Briefs shown under each experiment card -----------
EXP_BRIEFS = {
    # model1
    "m1_big": {
        "target": [
            "Get the set-up right",
            "Set Transforms / DataLoader",
            "Set basic Training & Test loop",
        ],
        "analysis": [
            "Extremely heavy model for MNIST",
            "Over-fitting; will change model in next step",
        ],
    },
    "m1_skeleton": {
        "target": [
            "Get the basic skeleton right (avoid changing later)",
            "No fancy stuff",
        ],
        "analysis": [
            "Model is still large but working",
            "Some over-fitting visible",
        ],
    },
    "m1_light": {
        "target": [
            "Make the model lighter",
        ],
        "analysis": [
            "Good baseline",
            "No over-fitting; capable if pushed further",
        ],
    },
    # model2
    "m2_light_bn": {
        "target": ["Stabilize training with BatchNorm"],
        "analysis": ["Improved stability vs m1_light; check capacity/regularization balance"],
    },
    "m2_light_do": {
        "target": ["Reduce overfitting with Dropout"],
        "analysis": ["Better generalization; slightly slower convergence possible"],
    },
    "m2_light_bn_do": {
        "target": ["Combine BN + DO for stability + regularization"],
        "analysis": ["Often a sweet spot; monitor accuracy vs. params"],
    },
    "m2_light_bn_do_gap": {
        "target": ["Use GAP to replace FC and reduce parameters"],
        "analysis": ["Large parameter reduction with minimal accuracy loss expected"],
    },
    # model3
    "m3_capacity": {
        "target": ["Tune channels to approach target"],
        "analysis": ["Beware overfitting as capacity grows"],
    },
    "m3_poolfix": {
        "target": ["Fix pooling placement after sufficient convs"],
        "analysis": ["Cleaner feature hierarchy; can improve test stability"],
    },
    "m3_poolfix_aug": {
        "target": ["Add augmentations (rotation/affine/perspective/erasing)"],
        "analysis": ["Less overfitting; minor hit on train acc acceptable"],
    },
    "m3_poolfix_steplr": {
        "target": ["Use StepLR for late-epoch refinement"],
        "analysis": ["Improves convergence at end"],
    },
    "m3_poolfix_aug_steplr": {
        "target": ["Augmentations + StepLR together"],
        "analysis": ["Aiming for ≥99.4% within ≤15 epochs"],
    },
    "m3_depthwiseConv_aug_steplr": {
        "target": ["Depthwise conv + Aug + StepLR (tiny target)"],
        "analysis": ["Tiny yet capable model; verify target consistency"],
    },
}

PLACEHOLDERS = [
    "DATA_STATS",
    "MODEL1_RESULTS",
    "MODEL2_RESULTS",
    "MODEL3_RESULTS",
    "CONSOLIDATED_RESULTS",
    "OBSERVATIONS",
    "ENV_INFO",
]

BLOCK_HEADER = "<!-- {tag} -->"
BLOCK_FOOTER = "<!-- /{tag} -->"

# -----------------------------
# Small utilities
# -----------------------------
def _split_exp_name(exp: str):
    """
    'm1_big' -> ('model1', 'big')
    'm2_light_bn' -> ('model2', 'light_bn')
    'm3_aug_steplr' -> ('model3', 'aug_steplr')
    """
    m = re.match(r"^(m[123])_(.+)$", str(exp).strip())
    if not m:
        return "unknown", str(exp)
    fam, var = m.groups()
    model_map = {"m1": "model1", "m2": "model2", "m3": "model3"}
    return model_map.get(fam, "unknown"), var

def enrich_results_df(df, colmap):
    """
    - Create 'model' and 'variant' from 'exp_name' if missing
    - Convert accuracies from 0..1 to 0..100 (%)
    - Normalize boolean-like strings to bool (augment/use_steplr)
    """
    if df is None:
        return df
    df = df.copy()

    # derive model/variant if needed
    if "model" not in df.columns or "variant" not in df.columns:
        if "exp_name" in df.columns:
            models, variants = zip(*[_split_exp_name(e) for e in df["exp_name"]])
            if "model" not in df.columns:
                df["model"] = models
            if "variant" not in df.columns:
                df["variant"] = variants

    # scale accuracies if in 0..1
    for acc_col in ["best_train_acc", "best_val_acc", "best_test_acc", "final_test_acc"]:
        if acc_col in df.columns and pd is not None:
            df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
            med = df[acc_col].median(skipna=True)
            if pd.notna(med) and med <= 1.2:
                df[acc_col] = df[acc_col] * 100.0

    # normalize booleans stored as strings
    for bcol in ["augment", "use_steplr"]:
        if bcol in df.columns:
            df[bcol] = df[bcol].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(df[bcol])

    return df

def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)

def sanitize_markdown(s: str) -> str:
    return s.replace("\r\n", "\n")

def detect_block(template: str, tag: str) -> Tuple[Optional[re.Match], Optional[re.Match]]:
    """
    Return matches for opening <!-- TAG --> and closing <!-- /TAG --> markers.
    """
    open_pat = re.compile(rf"<!--\s*{re.escape(tag)}\s*-->", re.MULTILINE)
    close_pat = re.compile(rf"<!--\s*/\s*{re.escape(tag)}\s*-->", re.MULTILINE)
    open_m = open_pat.search(template)
    close_m = close_pat.search(template) if open_m else None
    return open_m, close_m

def replace_block(template: str, tag: str, content_md: str) -> str:
    """
    If <!-- TAG --> ... <!-- /TAG --> exists, replace the inner content.
    Else if only <!-- TAG --> exists, expand it to wrapped block with content.
    Else no-op (returns template unchanged).
    """
    content_md = content_md.strip("\n")
    open_m, close_m = detect_block(template, tag)

    if open_m and close_m:
        start = open_m.end()
        end = close_m.start()
        before = template[:start]
        after = template[end:]
        new_inner = "\n" + content_md + "\n"
        return before + new_inner + after

    elif open_m and not close_m:
        insert_at = open_m.end()
        wrapped = f"\n{content_md}\n{BLOCK_FOOTER.format(tag=tag)}\n"
        return template[:insert_at] + wrapped + template[insert_at:]

    else:
        appended = (
            "\n\n"
            + BLOCK_HEADER.format(tag=tag)
            + "\n"
            + content_md
            + "\n"
            + BLOCK_FOOTER.format(tag=tag)
            + "\n"
        )
        return template + appended

def df_to_markdown(df) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return "_No data available._"
    if pd is not None and hasattr(df, "to_markdown"):
        try:
            return df.to_markdown(index=False)
        except Exception:
            pass
    # Fallback simple pipe table
    headers = list(df.columns)
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(map(lambda x: f"{x}", row.values)) + " |")
    return "\n".join(lines)

def maybe_link_plot(plots_dir: Path, filename: str) -> Optional[str]:
    p = plots_dir / filename
    if p.exists():
        return f"![]({p.as_posix()})"
    return None

# -----------------------------
# Content builders
# -----------------------------
def load_summary(results_csv: Path, colmap: Dict[str, str]) -> Optional["pd.DataFrame"]:
    if not results_csv.exists():
        return None
    if pd is None:
        rows = []
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if not rows:
            return None
        try:
            import pandas as _pd
            return _pd.DataFrame(rows)
        except Exception:
            return None
    df = pd.read_csv(results_csv)
    for k in ["params", "best_train_acc", "best_val_acc", "best_test_acc", "epochs"]:
        c = colmap.get(k)
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def section_data_stats(df: Optional["pd.DataFrame"], colmap: Dict[str, str]) -> str:
    if df is None or df.empty:
        return "_No `summary.csv` found or it is empty._"
    model_col = colmap["model"]
    variant_col = colmap["variant"]
    by_model = df.groupby(model_col).size().reset_index(name="runs")
    by_variant = df.groupby([model_col, variant_col]).size().reset_index(name="runs")
    top = [
        "### Dataset & Runs Snapshot",
        "- Below reflects **counts of runs** captured in `results/summary.csv`.",
        "",
        "#### Runs per Model",
        df_to_markdown(by_model),
        "",
        "#### Runs per Model × Variant",
        df_to_markdown(by_variant),
    ]
    return "\n".join(top)

def section_model_results(df: Optional["pd.DataFrame"], colmap: Dict[str, str], model_name: str,
                          plots_dir: Path) -> str:
    """
    For model1/model2/model3:
      1) Show a consolidated table of variants.
      2) For each experiment row, render a 'card' with:
         - Targets (from EXP_BRIEFS)
         - Results (auto-filled: params/train/test/epochs)
         - Analysis (from EXP_BRIEFS)
         - Inline plots if present: acc_{exp}.png, loss_{exp}.png, cm_{exp}.png
      3) At the bottom, add a grouped gallery with all accuracy curves for quick scan.
    """
    title = f"### {model_name} Results"
    if df is None or df.empty:
        return f"{title}\n\n_No data._"
    mcol = colmap["model"]
    vcol = colmap["variant"]
    pcol = colmap["params"]
    trcol = colmap["best_train_acc"]
    tecol = colmap.get("best_test_acc")
    vacol = colmap.get("best_val_acc")
    epcol = colmap.get("epochs")

    sub = df[df[mcol].astype(str).str.lower() == model_name.lower()].copy()
    if sub.empty:
        return f"{title}\n\n_No entries for **{model_name}** in `summary.csv`._"

    # ---------- 1) summary table ----------
    cols = [c for c in [vcol, pcol, trcol, vacol, tecol, epcol, "notes"] if c and c in sub.columns]
    show = sub[cols].rename(columns={
        vcol: "Variant",
        pcol: "Params",
        trcol: "Best Train Acc",
        vacol: "Best Val Acc",
        tecol: "Best Test Acc",
        epcol: "Epochs",
    })

    if pd is not None:
        for c in ["Params", "Best Train Acc", "Best Val Acc", "Best Test Acc"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
                if c == "Params":
                    show[c] = show[c].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
                else:
                    show[c] = show[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    blocks = [title, df_to_markdown(show), ""]

    # Helpers
    def _fmt_pct(x):
        try:
            return f"{float(x):.2f}%"
        except Exception:
            return ""
    def _fmt_int(x):
        try:
            return f"{int(float(x)):,}"
        except Exception:
            return ""

    model_prefix = {"model1": "m1", "model2": "m2", "model3": "m3"}.get(model_name, "mx")
    exp_col = "exp_name" if "exp_name" in sub.columns else None

    per_variant_cards = []
    acc_gallery = []

    # ---------- 2) Per-experiment cards ----------
    for _, row in sub.iterrows():
        exp_slug = str(row[exp_col]) if exp_col else f"{model_prefix}_{str(row[vcol])}"

        # Plots
        acc_file  = plots_dir / f"acc_{exp_slug}.png"
        loss_file = plots_dir / f"loss_{exp_slug}.png"
        cm_file   = plots_dir / f"cm_{exp_slug}.png"

        brief = EXP_BRIEFS.get(exp_slug, {
            "target": ["(Customize targets for this experiment)"],
            "analysis": ["(Add analysis notes for this experiment)"],
        })

        params = _fmt_int(row.get(pcol, ""))
        btr    = _fmt_pct(row.get(trcol, "")) if trcol in row else ""
        bte    = _fmt_pct(row.get(tecol, "")) if tecol and tecol in row else ""
        eps    = _fmt_int(row.get(epcol, "")) if epcol in row else ""

        card_lines = [f"#### {exp_slug}", ""]
        card_lines.append("**Target:**")
        for t in brief.get("target", []):
            card_lines.append(f"- {t}")
        card_lines.append("")
        card_lines.append("**Results:**")
        card_lines.append(f"- Parameters: {params}")
        if btr: card_lines.append(f"- Best Training Accuracy: {btr}")
        if bte: card_lines.append(f"- Best Test Accuracy: {bte}")
        if eps: card_lines.append(f"- Epochs: {eps}")
        card_lines.append("")
        card_lines.append("**Analysis:**")
        for a in brief.get("analysis", []):
            card_lines.append(f"- {a}")
        card_lines.append("")

        had_plot = False
        if acc_file.exists():
            card_lines.append(f"![]({acc_file.as_posix()})")
            had_plot = True
            acc_gallery.append(f"![]({acc_file.as_posix()})")
        if loss_file.exists():
            card_lines.append(f"![]({loss_file.as_posix()})")
            had_plot = True
        if cm_file.exists():
            card_lines.append(f"![]({cm_file.as_posix()})")
            had_plot = True
        if not had_plot:
            card_lines.append("_No plots available for this experiment._")

        per_variant_cards.append("\n".join(card_lines))

    if per_variant_cards:
        blocks.append("#### Per-Experiment Details")
        blocks.append("\n\n".join(per_variant_cards))

    # ---------- 3) Grouped gallery (accuracy only) ----------
    if acc_gallery:
        blocks.append("")
        blocks.append("#### All Accuracy Curves")
        blocks.append("\n".join([f"- {img}" for img in acc_gallery]))

    return "\n".join(blocks).strip()

def section_consolidated(df: Optional["pd.DataFrame"], colmap: Dict[str, str]) -> str:
    title = "### Consolidated Leaderboard"
    if df is None or df.empty:
        return f"{title}\n\n_No data._"
    mcol = colmap["model"]
    vcol = colmap["variant"]
    tecol = colmap.get("best_test_acc") or colmap.get("best_val_acc") or colmap.get("best_train_acc")
    pcol = colmap["params"]

    use = df[[mcol, vcol, tecol, pcol]].copy()
    use = use.rename(columns={
        mcol: "Model",
        vcol: "Variant",
        tecol: "Score",
        pcol: "Params",
    })
    if pd is not None:
        use["Params"] = pd.to_numeric(use["Params"], errors="coerce")
        use["Score"] = pd.to_numeric(use["Score"], errors="coerce")
        use = use.sort_values(["Score", "Params"], ascending=[False, True], na_position="last")
        use["Params"] = use["Params"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
        use["Score"] = use["Score"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    return f"{title}\n\n" + df_to_markdown(use)

def section_observations(df: Optional["pd.DataFrame"]) -> str:
    lines = [
        "### Observations",
        "- *Add commentary here:* trends, overfitting signals, BN/Dropout impact, optimizer effects, etc.",
        "- This section is maintained by the script; you can edit the bullet points and the script will keep the block.",
    ]
    return "\n".join(lines)

def section_env_info() -> str:
    py = sys.version.split()[0]
    plat = platform.platform()
    cuda = ""
    try:
        import torch
        cuda = f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda += f", device: {torch.cuda.get_device_name(0)}"
    except Exception:
        cuda = "_PyTorch not installed in this environment._"
    return "\n".join([
        "### Environment Info",
        f"- Python: `{py}`",
        f"- Platform: `{plat}`",
        f"- {cuda}",
    ])

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fill README placeholders from results.")
    ap.add_argument("--template", type=Path, default=Path("README.base.md"),
                    help="Base README template with placeholders.")
    ap.add_argument("--out", type=Path, default=Path("README.md"),
                    help="Output README to write/update.")
    ap.add_argument("--results", type=Path, default=Path("results/summary.csv"),
                    help="CSV with consolidated results.")
    ap.add_argument("--plots-dir", type=Path, default=Path("results/plots"),
                    help="Directory containing plots to link.")
    # Optional column remaps
    for k, v in DEFAULT_COLS.items():
        ap.add_argument(f"--col-{k}", type=str, default=v, help=f"Column for {k} (default: {v})")

    args = ap.parse_args()

    # Build column map from args
    colmap = {k: getattr(args, f"col_{k}") for k in DEFAULT_COLS.keys()}

    # Load template source (if output exists and not same as template, start from current out)
    if args.out.exists() and args.out.resolve() != args.template.resolve():
        base_text = read_text(args.out)
    else:
        base_text = read_text(args.template)

    base_text = sanitize_markdown(base_text)

    # Load + normalize summary
    df = load_summary(args.results, colmap)
    if df is None or df.empty:
        print("No rows in results/summary.csv — nothing to update.")
        sys.exit(0)
    df = enrich_results_df(df, colmap)

    # Build sections
    data_stats_md = section_data_stats(df, colmap)
    model1_md = section_model_results(df, colmap, "model1", args.plots_dir)
    model2_md = section_model_results(df, colmap, "model2", args.plots_dir)
    model3_md = section_model_results(df, colmap, "model3", args.plots_dir)
    consolidated_md = section_consolidated(df, colmap)
    observations_md = section_observations(df)
    env_md = section_env_info()

    # Replace placeholders
    updated = base_text
    updated = replace_block(updated, "DATA_STATS", data_stats_md)
    updated = replace_block(updated, "MODEL1_RESULTS", model1_md)
    updated = replace_block(updated, "MODEL2_RESULTS", model2_md)
    updated = replace_block(updated, "MODEL3_RESULTS", model3_md)
    updated = replace_block(updated, "CONSOLIDATED_RESULTS", consolidated_md)
    updated = replace_block(updated, "OBSERVATIONS", observations_md)
    updated = replace_block(updated, "ENV_INFO", env_md)

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, updated)
    print(f"README updated: {args.out}")

if __name__ == "__main__":
    main()
