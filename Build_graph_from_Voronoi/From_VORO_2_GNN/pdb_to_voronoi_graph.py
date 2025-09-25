#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a GNN-ready graph DIRECTLY from Voronota outputs
  - nodes: Voronota balls (with coords, element, residue info from annotations)
  - edges: Voronota annotated contacts (face area, center distance)
This avoids any fragile remapping to PDB atom IDs.

Usage:
  python voronota_to_pyg.py receptor_balls_nosolv.txt receptor_contacts_annot.txt --out receptor

Outputs:
  receptor_graph.pt
  receptor_ball_index.csv      (node table)
  receptor_residue_index.csv   (atom→residue map)
"""
import argparse, re, csv
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

VDW = {
    "H":1.20, "C":1.70, "N":1.55, "O":1.52, "P":1.80, "S":1.80,
    "F":1.47, "CL":1.75, "BR":1.85, "I":1.98, "MG":1.73, "NA":2.27, "K":2.75
}

def vdw_radius(elem):
    e = (elem or "").upper()
    return VDW.get(e, 1.8)

def parse_kv_line(line):
    """Parse a Voronota 'key=value' line into a dict (safe for extra tokens)."""
    d = {}
    for tok in line.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d

def load_balls(path):
    """
    Expect lines with keys like:
      id=A:11:O5'  chain=A  residue_type=U  residue_number=11  atom_name=O5'  x=.. y=.. z=.. element=O  bfactor=..
    Exact field names can vary slightly; we fall back when missing.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            kv = parse_kv_line(s)
            _id   = kv.get("id", "")
            chain = kv.get("chain", "")
            resn  = kv.get("residue_type", kv.get("resname",""))
            resi  = kv.get("residue_number", kv.get("resid",""))
            atom  = kv.get("atom_name", kv.get("atom",""))
            elem  = kv.get("element", "")
            # coords
            x = float(kv.get("x","0")); y = float(kv.get("y","0")); z = float(kv.get("z","0"))
            # b-factor (optional)
            try:
                b = float(kv.get("bfactor","0"))
            except Exception:
                b = 0.0
            rows.append({
                "id": _id, "chain": chain, "resname": resn, "resid": resi, "atom": atom,
                "element": elem, "x": x, "y": y, "z": z, "bfactor": b
            })
    df = pd.DataFrame(rows)
    # normalize
    df["resid"] = df["resid"].astype(str)
    df["atom_norm"] = df["atom"].str.replace(r"\*","'", regex=True).str.replace(" ","", regex=False)
    return df

def load_contacts(path):
    """
    Expect keys at least: id1, id2, area, center_distance (non-solvent; we assume you filtered earlier)
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            kv = parse_kv_line(s)
            # skip solvent contacts if any slipped through
            if kv.get("annotation1","") == "solvent" or kv.get("annotation2","") == "solvent":
                continue
            try:
                area = float(kv.get("area","0"))
                cd   = float(kv.get("center_distance","0"))
            except Exception:
                continue
            if area <= 0:
                continue
            id1 = (kv.get("id1","") or "").replace("*","'").replace(" ","")
            id2 = (kv.get("id2","") or "").replace("*","'").replace(" ","")
            if id1 and id2:
                rows.append((id1, id2, cd, area))
    return rows

def node_features(df):
    # element one-hot (C,N,O,P,S,H) + base one-hot (A,C,G,U) + VDW + scaled B
    el = df["element"].str.upper().fillna("")
    elem = np.stack([
        (el=="C").values, (el=="N").values, (el=="O").values,
        (el=="P").values, (el=="S").values, (el=="H").values
    ], axis=1).astype(float)
    base = df["resname"].str.upper().fillna("")
    base_vec = np.stack([
        (base=="A").values, (base=="C").values, (base=="G").values,
        base.isin(["U","URA","U5P"]).values
    ], axis=1).astype(float)
    vdw = np.array([vdw_radius(e) for e in el], dtype=float).reshape(-1,1)
    bsc = (df["bfactor"].fillna(0.0).astype(float)/50.0).values.reshape(-1,1)
    X = np.concatenate([elem, base_vec, vdw, bsc], axis=1)
    pos = df[["x","y","z"]].to_numpy(dtype=float)
    # residue mapping
    res_keys = (df["chain"].astype(str)+":"+df["resid"].astype(str)+":"+df["resname"].astype(str)).tolist()
    uniq = sorted(set(res_keys)); rmap = {k:i for i,k in enumerate(uniq)}
    res_idx = np.array([rmap[k] for k in res_keys], dtype=int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(res_idx, dtype=torch.long), uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("balls_annot")
    ap.add_argument("contacts_annot")
    ap.add_argument("--out", default="receptor")
    args = ap.parse_args()

    balls = load_balls(args.balls_annot)
    contacts = load_contacts(args.contacts_annot)

    # Build node index by Voronota's own 'id' (exactly what contacts use)
    id2idx = {i: k for k, i in enumerate(balls["id"].astype(str).str.replace(r"\*","'", regex=True).str.replace(" ","", regex=False))}
    rows, cols, eattr = [], [], []
    missing = 0
    for id1, id2, cd, area in contacts:
        i = id2idx.get(id1); j = id2idx.get(id2)
        if i is None or j is None:
            missing += 1
            continue
        rows.extend([i,j]); cols.extend([j,i])
        eattr.extend([[cd,area],[cd,area]])

    if not rows:
        raise RuntimeError("No edges matched even when using Voronota IDs. Check that both files are from the same run and not empty.")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    x, pos, res_idx, res_keys = node_features(balls)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=x.size(0), reduce='mean')

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.res_idx = res_idx
    torch.save(data, f"{args.out}_graph.pt")

    balls.to_csv(f"{args.out}_ball_index.csv", index=False)
    pd.DataFrame({"res_key": res_keys, "res_idx": list(range(len(res_keys)))}).to_csv(f"{args.out}_residue_index.csv", index=False)

    print(f"[OK] Saved {args.out}_graph.pt | nodes={x.size(0)} edges={edge_index.size(1)} missing_pairs={missing}")

if __name__ == "__main__":
    main()
