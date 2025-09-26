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
    model, variant, params, best_train_acc, best_test_acc, epochs, notes
  You can adapt the column names via --col-* flags below.
"""

import argparse
import csv
import os
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
    df = df.copy()

    # derive model/variant if needed
    if "model" not in df.columns or "variant" not in df.columns:
        if "exp_name" not in df.columns:
            raise ValueError("summary.csv missing 'model'/'variant' and also 'exp_name' to derive from.")
        models, variants = zip(*[_split_exp_name(e) for e in df["exp_name"]])
        if "model" not in df.columns:
            df["model"] = models
        if "variant" not in df.columns:
            df["variant"] = variants

    # scale accuracies if in 0..1
    for acc_col in ["best_train_acc", "best_val_acc", "best_test_acc", "final_test_acc"]:
        if acc_col in df.columns:
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
        # Upgrade single tag to wrapped block
        insert_at = open_m.end()
        wrapped = f"\n{content_md}\n{BLOCK_FOOTER.format(tag=tag)}\n"
        return template[:insert_at] + wrapped + template[insert_at:]

    else:
        # Tag not present: optionally append at end
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
    if df is None or df.empty:
        return "_No data available._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: simple pipe table
        headers = list(df.columns)
        lines = ["| " + " | ".join(map(str, headers)) + " |",
                 "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(map(lambda x: f"{x}", row.values)) + " |")
        return "\n".join(lines)

def maybe_link_plot(plots_dir: Path, filename: str) -> str:
    p = plots_dir / filename
    if p.exists():
        # relative link makes GitHub render
        return f"![]({p.as_posix()})"
    return f"_Plot not found: `{filename}`_"

# -----------------------------
# Content builders
# -----------------------------
def load_summary(results_csv: Path, colmap: Dict[str, str]) -> Optional["pd.DataFrame"]:
    if not results_csv.exists():
        return None
    if pd is None:
        # Very small CSV reader fallback
        rows = []
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        if not rows:
            return None
        # best-effort normalize
        try:
            import pandas as _pd
            return _pd.DataFrame(rows)
        except Exception:
            return None
    df = pd.read_csv(results_csv)
    # Normalize known numeric fields if present
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
    params_col = colmap["params"]
    # Simple counts by model / variant
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

    # Select & format
    cols = [c for c in [vcol, pcol, trcol, vacol, tecol, epcol, "notes"] if c and c in sub.columns]
    show = sub[cols].rename(columns={
        vcol: "Variant",
        pcol: "Params",
        trcol: "Best Train Acc",
        vacol: "Best Val Acc",
        tecol: "Best Test Acc",
        epcol: "Epochs",
    })

    # Nicify numeric fields
    for c in ["Params", "Best Train Acc", "Best Val Acc", "Best Test Acc"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce")
            if c == "Params":
                show[c] = show[c].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
            else:
                show[c] = show[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

    blocks = [title, df_to_markdown(show), ""]
    # Optional: attach standard plot thumbnails if present
    # You can adapt filenames to your convention
    maybe_plots = [
        ("Top-5 Train Accuracy", "combined_train_accuracy_top5.png"),
        ("Top-5 Test Accuracy", "combined_test_accuracy_top5.png"),
        ("Top-5 Val Accuracy", "combined_val_accuracy_top5.png"),
    ]
    found_any = False
    for label, fname in maybe_plots:
        md = maybe_link_plot(plots_dir, fname)
        if not md.startswith("_Plot not found"):
            if not found_any:
                blocks.append("#### Plots")
                found_any = True
            blocks.append(f"- **{label}**\n\n{md}\n")
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
    use["Params"] = pd.to_numeric(use["Params"], errors="coerce")
    use["Score"] = pd.to_numeric(use["Score"], errors="coerce")
    use = use.sort_values(["Score", "Params"], ascending=[False, True], na_position="last")
    use["Params"] = use["Params"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "")
    use["Score"] = use["Score"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    return f"{title}\n\n" + df_to_markdown(use)

def section_observations(df: Optional["pd.DataFrame"]) -> str:
    # Stub you can edit later; keeps automation idempotent
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

    # Load summary
    # df = load_summary(args.results, colmap)
    # Load summary
    df = load_summary(args.results, colmap)
    if df is None or df.empty:
        print("No rows in results/summary.csv — nothing to update.")
        sys.exit(0)
    df = enrich_results_df(df, colmap)

    if df is None or df.empty:
        print("No rows in results/summary.csv — nothing to update.")
        sys.exit(0)

    # Add model/variant if missing; fix accuracy scales
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
