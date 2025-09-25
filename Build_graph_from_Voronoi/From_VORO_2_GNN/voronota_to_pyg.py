#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Voronota *plain-format* balls + contacts into a PyTorch Geometric graph.

Expected formats:
  contacts (plain):  id1  id2  <area>  <center_distance>  .  .
  balls    (plain):  id  x  y  z  radius  el=N  ... (other kv pairs)

Example IDs look like: c<A>r<13>a<1>R<THR>A<N>

Usage:
  python voronota_plain_to_pyg.py receptor_balls_nosolv.txt receptor_contacts_annot.txt --out receptor
"""
import argparse, re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

# --- simple VDW table (used only if you want it as a node feature) ---
VDW = {
    "H":1.20, "C":1.70, "N":1.55, "O":1.52, "P":1.80, "S":1.80,
    "F":1.47, "CL":1.75, "BR":1.85, "I":1.98, "MG":1.73, "NA":2.27, "K":2.75
}
def vdw_radius(elem: str) -> float:
    e = (elem or "").upper()
    return VDW.get(e, 1.8)

# --- regex helpers to extract fields from the Voronota ID string ---
RE_CHAIN = re.compile(r'c<([^>]+)>')
RE_RESID = re.compile(r'r<([^>]+)>')
RE_RNAME = re.compile(r'R<([^>]+)>')
RE_ANAME = re.compile(r'A<([^>]+)>')

def parse_id(idstr: str):
    """Return (chain, resid, resname, atomname) parsed from an ID like c<A>r<13>...R<THR>A<N>."""
    chain   = RE_CHAIN.search(idstr)
    resid   = RE_RESID.search(idstr)
    resname = RE_RNAME.search(idstr)
    atom    = RE_ANAME.search(idstr)
    return (chain.group(1) if chain else "",
            resid.group(1) if resid else "",
            (resname.group(1) if resname else "").upper(),
            atom.group(1) if atom else "")

def load_balls_plain(path):
    """Read plain balls: id x y z radius [kv...] → DataFrame with id, x,y,z, radius, element, chain,resid,resname,atom."""
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            toks = s.split()
            if len(toks) < 5: 
                continue
            _id = toks[0]
            try:
                x, y, z, rad = map(float, toks[1:5])
            except Exception:
                continue
            # parse trailing kv (e.g., el=N)
            kv = {}
            for t in toks[5:]:
                if "=" in t:
                    k,v = t.split("=",1)
                    kv[k] = v
            el = kv.get("el", kv.get("element", ""))  # element symbol if present
            chain, resid, resname, atom = parse_id(_id)
            rows.append({
                "id": _id, "x": x, "y": y, "z": z, "radius": rad,
                "element": el, "chain": chain, "resid": resid, "resname": resname, "atom": atom
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No balls parsed from {path}. Is it the correct file?")
    return df

def load_contacts_plain(path):
    """Read plain contacts: id1 id2 area center_dist ... → list of (id1, id2, area, center_dist) excluding solvent lines."""
    edges = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            toks = s.split()
            if len(toks) < 4:
                continue
            id1, id2 = toks[0], toks[1]
            try:
                area = float(toks[2])
                cd   = float(toks[3])
            except Exception:
                # sometimes order can be swapped; try best-effort
                try:
                    cd = float(toks[2]); area = float(toks[3])
                except Exception:
                    continue
            # skip solvent rows (they usually have 'c<solvent>' as one id)
            if id1.startswith("c<solvent>") or id2.startswith("c<solvent>"):
                continue
            if area <= 0:
                continue
            edges.append((id1, id2, area, cd))
    if not edges:
        raise RuntimeError(f"No contacts parsed from {path}.")
    return edges

def build_graph(balls_df, contacts):
    # map ball id -> node index
    ids = balls_df["id"].astype(str).tolist()
    id2idx = {s:i for i,s in enumerate(ids)}

    # build edges + edge_attr
    rows, cols, eattr = [], [], []
    missing = 0
    for id1, id2, area, cd in contacts:
        i = id2idx.get(id1); j = id2idx.get(id2)
        if i is None or j is None:
            missing += 1
            continue
        # undirected
        rows += [i,j]; cols += [j,i]
        eattr += [[cd, area], [cd, area]]

    if not rows:
        raise RuntimeError("All contacts failed to map to balls (id mismatch). Ensure files come from the same Voronota run.")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    # node features
    el = balls_df["element"].fillna("").str.upper()
    # if element missing, try derive from atom name's first letter
    guess_el = balls_df["atom"].fillna("").str.upper().str.replace(r"[^A-Z]", "", regex=True).str[:1]
    el = el.mask(el.eq(""), guess_el)

    elem_oh = np.stack([
        (el=="C").values, (el=="N").values, (el=="O").values,
        (el=="P").values, (el=="S").values, (el=="H").values
    ], axis=1).astype(float)

    base = balls_df["resname"].fillna("").str.upper()
    base_oh = np.stack([
        (base=="A").values, (base=="C").values, (base=="G").values,
        base.isin(["U","URA","U5P"]).values
    ], axis=1).astype(float)

    # you can choose either Voronota's radius (from file) or VDW table; we include *both*
    rad_voronota = balls_df["radius"].astype(float).to_numpy().reshape(-1,1)
    rad_vdw = np.array([vdw_radius(e) for e in el], dtype=float).reshape(-1,1)

    X = np.concatenate([elem_oh, base_oh, rad_voronota, rad_vdw], axis=1)
    x = torch.tensor(X, dtype=torch.float32)

    # positions
    pos = torch.tensor(balls_df[["x","y","z"]].to_numpy(dtype=float), dtype=torch.float32)

    # residue mapping (for pooling later)
    res_keys = (balls_df["chain"].astype(str)+":"+balls_df["resid"].astype(str)+":"+balls_df["resname"].astype(str)).tolist()
    uniq = sorted(set(res_keys)); rmap = {k:i for i,k in enumerate(uniq)}
    res_idx = torch.tensor([rmap[k] for k in res_keys], dtype=torch.long)

    # coalesce duplicates
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=x.size(0), reduce='mean')

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.res_idx = res_idx
    return data, uniq, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("balls_plain")
    ap.add_argument("contacts_plain")
    ap.add_argument("--out", default="receptor")
    args = ap.parse_args()

    balls = load_balls_plain(args.balls_plain)
    contacts = load_contacts_plain(args.contacts_plain)

    data, res_keys, missing = build_graph(balls, contacts)
    torch.save(data, f"{args.out}_graph.pt")
    balls.to_csv(f"{args.out}_ball_index.csv", index=False)
    pd.DataFrame({"res_key": res_keys, "res_idx": list(range(len(res_keys)))}).to_csv(f"{args.out}_residue_index.csv", index=False)

    print(f"[OK] {args.out}_graph.pt  | nodes={data.num_nodes}  edges={data.edge_index.size(1)}  unmatched_contacts={missing}")

if __name__ == "__main__":
    main()
