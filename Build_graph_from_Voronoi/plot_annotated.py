#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_voronota_contacts.py

Read Voronota 'annotated_contacts.txt' and save plots & summaries:
  - Histogram of contact areas
  - Scatter plot (distance vs area)
  - Top residues by total contact area (bar)
  - Top residue-residue pairs by total contact area (bar)
Also writes CSVs with per-residue and per-residue-pair summaries.

Usage:
  python plot_voronota_contacts.py --input annotated_contacts.txt --outdir ./plots_contacts
"""

import argparse
import os
import re
import sys
import math
import json
from typing import Tuple, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure no GUI required
import matplotlib.pyplot as plt


# ------------------------- Parsing helpers -------------------------

# Example descriptor: c<A>r<23>a<45>R<GUA>A<N1>
# We’ll extract residue name and an identifier that combines chain + resseq(+icode) for grouping.
RE_RESNAME = re.compile(r'R<([^>]+)>')
RE_CHAIN   = re.compile(r'c<([^>]+)>')
RE_RESSEQ  = re.compile(r'r<([^>]+)>')   # residue number (can have insertion codes in PDB, mmCIF mapping varies)
RE_ATOM    = re.compile(r'A<([^>]+)>')   # atom name (optional for summaries)

def parse_descriptor(desc: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Parse an atom descriptor string into (resname, residue_id, chain, atom_name).
    residue_id := "{chain}:{resseq}" if chain present else "{resseq}"
    """
    resname = RE_RESNAME.search(desc)
    chain   = RE_CHAIN.search(desc)
    resseq  = RE_RESSEQ.search(desc)
    atom    = RE_ATOM.search(desc)

    resname_val = resname.group(1) if resname else "UNK"
    chain_val   = chain.group(1) if chain else ""
    resseq_val  = resseq.group(1) if resseq else "?"

    if chain_val:
        residue_id = f"{chain_val}:{resseq_val}"
    else:
        residue_id = f"{resseq_val}"

    atom_val = atom.group(1) if atom else None
    return resname_val, residue_id, chain_val if chain_val else None, atom_val


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_contacts_table(path: str) -> pd.DataFrame:
    """
    Load the first 4 whitespace-separated fields:
      col0: descriptor1
      col1: descriptor2
      col2: area
      col3: distance
    Ignore lines that don't have >= 4 tokens.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            a1, a2, area_str, dist_str = toks[0], toks[1], toks[2], toks[3]
            area = safe_float(area_str)
            dist = safe_float(dist_str)
            if area is None or dist is None:
                continue
            rows.append((a1, a2, area, dist))
    df = pd.DataFrame(rows, columns=["atom1", "atom2", "area", "distance"])
    return df


# ------------------------- Plotting helpers -------------------------

def save_hist(series: pd.Series, out_png: str, title: str, xlabel: str, bins: int = 50):
    fig = plt.figure(figsize=(7, 5))
    plt.hist(series.dropna().values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_scatter(x: pd.Series, y: pd.Series, out_png: str, title: str, xlabel: str, ylabel: str, alpha: float = 0.3):
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(x.values, y.values, alpha=alpha, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_bar(labels, values, out_png: str, title: str, xlabel: str, ylabel: str, rotation: int = 45):
    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot Voronota annotated contacts to files.")
    ap.add_argument("--input", "-i", required=True, help="Path to annotated_contacts.txt (Voronota)")
    ap.add_argument("--outdir", "-o", required=True, help="Output directory for plots and CSVs")
    ap.add_argument("--topk", type=int, default=20, help="Top K items to plot in bar charts (default: 20)")
    args = ap.parse_args()

    in_path = os.path.abspath(args.input)
    outdir  = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(in_path):
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Loading contacts from: {in_path}")
    df = load_contacts_table(in_path)

    if df.empty:
        print("[ERROR] No valid rows found (need columns: atom1 atom2 area distance).", file=sys.stderr)
        sys.exit(3)

    # Extract residue info
    print("[INFO] Parsing atom descriptors...")
    r1, rid1, c1, a1 = zip(*df["atom1"].map(parse_descriptor))
    r2, rid2, c2, a2 = zip(*df["atom2"].map(parse_descriptor))

    df["res1"] = r1
    df["res2"] = r2
    df["rid1"] = rid1  # chain:resseq
    df["rid2"] = rid2

    # Basic summaries
    n_rows = len(df)
    n_solvent_rows = int((df["atom2"].str.contains("solvent", na=False)).sum())
    area_total = float(df["area"].sum())
    area_mean = float(df["area"].mean())
    dist_mean = float(df["distance"].mean())

    summary = {
        "n_contacts_rows": n_rows,
        "n_solvent_rows": n_solvent_rows,
        "total_area": area_total,
        "mean_area": area_mean,
        "mean_distance": dist_mean,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("[INFO] Saving plots...")
    save_hist(
        df["area"],
        os.path.join(outdir, "hist_contact_area.png"),
        "Distribution of Voronoi contact areas",
        "Contact area (Å²)",
        bins=60
    )

    save_scatter(
        df["distance"],
        df["area"],
        os.path.join(outdir, "scatter_distance_vs_area.png"),
        "Voronoi contact area vs atom–atom distance",
        "Distance (Å)",
        "Area (Å²)",
        alpha=0.25
    )

    # Per-residue total areas (for residue of atom1)
    res_area = (
        df.groupby(["res1", "rid1"], as_index=False)["area"]
        .sum()
        .sort_values("area", ascending=False)
    )
    res_area.to_csv(os.path.join(outdir, "per_residue_total_area.csv"), index=False)

    top_res = res_area.head(args.topk)
    save_bar(
        labels=[f"{r} ({rid})" for r, rid in zip(top_res["res1"], top_res["rid1"])],
        values=top_res["area"].values,
        out_png=os.path.join(outdir, "bar_top_residues_by_area.png"),
        title=f"Top {len(top_res)} residues by total contact area (using atom1 residues)",
        xlabel="Residue (resname chain:resseq)",
        ylabel="Total area (Å²)",
        rotation=60
    )

    # Per residue-residue pair (unordered)
    # Make canonical pair key to avoid double-counting A-B and B-A
    pair_keys = []
    pair_labels = []
    for rname1, rid_1, rname2, rid_2 in zip(df["res1"], df["rid1"], df["res2"], df["rid2"]):
        # build labels like GUA A:23 — CYS B:15
        label1 = f"{rname1} {rid_1}"
        label2 = f"{rname2} {rid_2}"
        if label1 <= label2:
            key = (label1, label2)
        else:
            key = (label2, label1)
        pair_keys.append(key)
        pair_labels.append(key)

    df_pairs = df.copy()
    df_pairs["pair"] = pair_keys

    pair_area = (
        df_pairs.groupby("pair")["area"]
        .sum()
        .reset_index()
        .sort_values("area", ascending=False)
    )
    pair_area[["residue1", "residue2"]] = pd.DataFrame(pair_area["pair"].tolist(), index=pair_area.index)
    pair_area = pair_area[["residue1", "residue2", "area"]]
    pair_area.to_csv(os.path.join(outdir, "per_residue_pair_total_area.csv"), index=False)

    top_pairs = pair_area.head(args.topk)
    labels = [f"{r1} — {r2}" for r1, r2 in zip(top_pairs["residue1"], top_pairs["residue2"])]
    save_bar(
        labels=labels,
        values=top_pairs["area"].values,
        out_png=os.path.join(outdir, "bar_top_residue_pairs_by_area.png"),
        title=f"Top {len(top_pairs)} residue–residue pairs by total contact area",
        xlabel="Residue pair",
        ylabel="Total area (Å²)",
        rotation=60
    )

    print(f"[DONE] Wrote outputs to: {outdir}")
    print("       - hist_contact_area.png")
    print("       - scatter_distance_vs_area.png")
    print("       - bar_top_residues_by_area.png")
    print("       - bar_top_residue_pairs_by_area.png")
    print("       - per_residue_total_area.csv")
    print("       - per_residue_pair_total_area.csv")
    print("       - summary.json")


if __name__ == "__main__":
    main()
