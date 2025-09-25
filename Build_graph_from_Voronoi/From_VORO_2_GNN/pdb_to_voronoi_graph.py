#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Voronota contacts -> atom-level graph (PyTorch Geometric).
Handles multiple Voronota id formats:
  A:11:O5' | A:U:11:O5' | 1:A:11:O5' | A:11A:O3' | (spaces/asterisks primes)
Usage:
  python pdb_to_voronoi_graph.py receptor.pdb receptor_contacts.txt --out receptor [--graphml] [--debug]
Outputs:
  receptor_graph.pt, receptor_atom_index.csv, receptor_residue_index.csv, (optional) receptor_graph.graphml
"""
import argparse, os, re, math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from Bio.PDB import PDBParser

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

VDW = {
    "H":1.20, "C":1.70, "N":1.55, "O":1.52, "P":1.80, "S":1.80,
    "F":1.47, "CL":1.75, "BR":1.85, "I":1.98, "MG":1.73, "NA":2.27, "K":2.75
}

def vdw_radius(elem: str) -> float:
    e = (elem or "").upper()
    return VDW.get(e, 1.8)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("contacts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--graphml", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def load_pdb_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("st", pdb_path)
    atoms = []
    idx = 0
    for model in st:
        model_id = str(model.id)
        for chain in model:
            chain_id = chain.id
            for res in chain:
                het = res.id[0].strip() != ""
                resseq = res.id[1]
                icode = res.id[2].strip() if res.id[2] != " " else ""
                resname = res.get_resname().strip()
                for atom in res:
                    name = atom.get_name().strip()
                    element = (atom.element or name[:1]).strip()
                    x, y, z = atom.coord
                    b = float(atom.get_bfactor() or 0.0)
                    atoms.append({
                        "idx": idx,
                        "model": model_id,
                        "chain": chain_id,
                        "resid": int(resseq),
                        "icode": icode,
                        "resname": resname,
                        "atom": name,
                        "element": element,
                        "het": het,
                        "x": float(x), "y": float(y), "z": float(z),
                        "bfactor": b
                    })
                    idx += 1
    return atoms

def normalize_atom_name(s: str) -> str:
    # Map PDB-style primes and spaces consistently
    return s.replace("*", "'").replace(" ", "")

def build_multi_keys(a):
    """
    Build multiple candidate keys for this atom so we can match different Voronota id patterns.
    Returns list of strings.
    """
    chain = a["chain"]
    resid = str(a["resid"]) + (a["icode"] or "")
    atom = normalize_atom_name(a["atom"])
    resname = (a["resname"] or "").upper()
    model = a["model"]

    # candidates (ordered by likelihood)
    keys = [
        f"{chain}:{resid}:{atom}",             # A:11:O5'
        f"{chain}:{resname}:{resid}:{atom}",   # A:U:11:O5'
        f"{model}:{chain}:{resid}:{atom}",     # 1:A:11:O5'
        f"{model}:{chain}:{resname}:{resid}:{atom}",  # 1:A:U:11:O5'
    ]
    return keys

def index_atoms(atoms):
    # map each candidate key to atom index
    idx_map = {}
    for a in atoms:
        for k in build_multi_keys(a):
            if k not in idx_map:
                idx_map[k] = a["idx"]
    return idx_map

def parse_contacts(path, debug=False, sample=20):
    """
    Parse Voronota annotated contacts; keep non-solvent atom-atom with positive area.
    Returns list of (id1, id2, cd, area) and a small sample of raw ids (for debugging).
    """
    edges = []
    raw_ids = set()
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t for t in s.split() if "=" in t]
            rec = {}
            for t in toks:
                k, v = t.split("=", 1)
                rec[k] = v
            if rec.get("annotation1","") == "solvent" or rec.get("annotation2","") == "solvent":
                continue
            try:
                area = float(rec.get("area","0") or 0.0)
                cd = float(rec.get("center_distance","0") or 0.0)
            except Exception:
                continue
            if area <= 0: 
                continue
            id1 = rec.get("id1","")
            id2 = rec.get("id2","")
            # normalize prime/space
            id1n = normalize_atom_name(id1)
            id2n = normalize_atom_name(id2)
            edges.append((id1n, id2n, cd, area))
            if len(raw_ids) < sample:
                raw_ids.add(id1)
                raw_ids.add(id2)
    return edges, sorted(list(raw_ids))[:sample]

def try_match(id_str, key2idx):
    """
    Try to map an id string like:
      A:11:O5' | A:U:11:O5' | 1:A:11:O5' | A:11A:O3' | 1:A:U:11:O5'
    to an atom index by progressively relaxing/transforming patterns.
    """
    # fast path: exact
    if id_str in key2idx:
        return key2idx[id_str]

    parts = id_str.split(":")
    # General strategy:
    # - last part is atom name
    # - first is chain or model
    # - somewhere there is resid (digits + optional insertion code letter)
    # - optional resname token (A/C/G/U/...); optional model token at start

    # Build candidates
    cand = []
    if len(parts) >= 3:
        atom = parts[-1]
        # pick first token that looks like chain (1-2 chars typical)
        # chain is often parts[0] or parts[1] if model present
        # resid token: contains digits, maybe trailing letter
        # resname token: 1-3 letters (A,C,G,U or other)
        toks = parts[:-1]  # without atom
        # Guess model+chain
        if len(parts) >= 4 and parts[0].isdigit():  # model present
            model = parts[0]; chain = parts[1]
            rest = parts[2:-0] if len(parts)>3 else []
        else:
            model = None; chain = parts[0]
            rest = parts[1:]
        # Find resid (first token containing a digit)
        resid_tok = next((t for t in rest if re.search(r"\d", t)), None)
        # Find resname (A/C/G/U/..), prefer tokens of letters only, length<=3
        resname_tok = next((t for t in rest if re.fullmatch(r"[A-Za-z]+", t) and len(t)<=3), None)

        if resid_tok is not None:
            resid = resid_tok
            if model:
                cand.append(f"{chain}:{resid}:{atom}")
                if resname_tok:
                    cand.append(f"{chain}:{resname_tok}:{resid}:{atom}")
                cand.append(f"{model}:{chain}:{resid}:{atom}")
                if resname_tok:
                    cand.append(f"{model}:{chain}:{resname_tok}:{resid}:{atom}")
            else:
                cand.append(f"{chain}:{resid}:{atom}")
                if resname_tok:
                    cand.append(f"{chain}:{resname_tok}:{resid}:{atom}")

    # Try candidates
    for k in cand:
        if k in key2idx:
            return key2idx[k]

    # As a last resort, try to strip potential insertion letter from resid (e.g., 11A -> 11)
    if len(parts) >= 3:
        atom = parts[-1]
        if len(parts) >= 4 and parts[0].isdigit():
            model = parts[0]; chain = parts[1]; rest = parts[2:-0] if len(parts)>3 else []
        else:
            model = None; chain = parts[0]; rest = parts[1:]
        resid_tok = next((t for t in rest if re.search(r"\d", t)), None)
        resname_tok = next((t for t in rest if re.fullmatch(r"[A-Za-z]+", t) and len(t)<=3), None)
        if resid_tok:
            resid_num = re.search(r"(\d+)", resid_tok).group(1)
            cands2 = []
            if model:
                cands2 += [f"{chain}:{resid_num}:{atom}", f"{model}:{chain}:{resid_num}:{atom}"]
                if resname_tok:
                    cands2 += [f"{chain}:{resname_tok}:{resid_num}:{atom}",
                               f"{model}:{chain}:{resname_tok}:{resid_num}:{atom}"]
            else:
                cands2 += [f"{chain}:{resid_num}:{atom}"]
                if resname_tok:
                    cands2 += [f"{chain}:{resname_tok}:{resid_num}:{atom}"]
            for k in cands2:
                if k in key2idx:
                    return key2idx[k]

    return None

def make_features(atoms):
    X, pos, res_keys = [], [], []
    for a in atoms:
        el = (a["element"] or "").upper()
        elem = np.array([el=="C", el=="N", el=="O", el=="P", el=="S", el=="H"], dtype=float)
        base = (a["resname"] or "").upper()
        base_vec = np.array([base=="A", base=="C", base=="G", base in ("U","URA","U5P")], dtype=float)
        vdw = vdw_radius(el)
        bscaled = (a["bfactor"]/50.0) if a["bfactor"] is not None else 0.0
        het = 1.0 if a["het"] else 0.0
        X.append(np.concatenate([elem, base_vec, [vdw, bscaled, het]], axis=0))
        pos.append([a["x"], a["y"], a["z"]])
        res_keys.append(f"{a['chain']}:{a['resid']}:{a['icode'] or ''}:{a['resname']}")
    X = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
    pos = torch.tensor(np.asarray(pos, dtype=float), dtype=torch.float32)
    uniq = sorted(set(res_keys)); rmap = {k:i for i,k in enumerate(uniq)}
    res_idx = torch.tensor([rmap[k] for k in res_keys], dtype=torch.long)
    return X, pos, res_idx, uniq

def main():
    args = get_args()
    out = args.out or os.path.splitext(os.path.basename(args.pdb))[0]

    atoms = load_pdb_atoms(args.pdb)
    atom_df = pd.DataFrame(atoms)
    atom_df.to_csv(f"{out}_atom_index.csv", index=False)

    key2idx = index_atoms(atoms)

    # Parse contacts and capture a small sample of raw ids for debugging
    edges_raw, raw_id_sample = parse_contacts(args.contacts, debug=args.debug)

    rows, cols, eattr = [], [], []
    missing_pairs = 0
    missing_ids = {}

    for k1, k2, cd, area in edges_raw:
        i = try_match(k1, key2idx)
        j = try_match(k2, key2idx)
        if i is None or j is None:
            missing_pairs += 1
            if i is None:
                missing_ids[k1] = missing_ids.get(k1, 0) + 1
            if j is None:
                missing_ids[k2] = missing_ids.get(k2, 0) + 1
            continue
        rows.extend([i, j]); cols.extend([j, i])
        eattr.extend([[cd, area], [cd, area]])

    if len(rows) == 0:
        msg = "No edges matched."
        if args.debug:
            msg += "\n[DEBUG] Example Voronota ids:\n  - " + "\n  - ".join(raw_id_sample)
            msg += "\n[DEBUG] Top unmapped ids:\n  - " + "\n  - ".join([f"{k} (n={v})" for k,v in list(sorted(missing_ids.items(), key=lambda x: -x[1]))[:10]])
        raise RuntimeError(msg + "\nHint: ensure contacts were generated from the SAME PDB; check chain IDs and residue numbering.")
    if args.debug and missing_pairs > 0:
        print(f"[WARN] Unmatched contact endpoints: {missing_pairs} (showing top 10)")
        for k,v in list(sorted(missing_ids.items(), key=lambda x: -x[1]))[:10]:
            print("  ", k, "→ unmatched", f"(n={v})")

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    x, pos, res_idx, res_keys = make_features(atoms)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=x.size(0), reduce='mean')

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.res_idx = res_idx
    torch.save(data, f"{out}_graph.pt")

    pd.DataFrame({"res_key": res_keys, "res_idx": list(range(len(res_keys)))}).to_csv(f"{out}_residue_index.csv", index=False)

    if args.graphml:
        if not HAVE_NX:
            print("[WARN] networkx not installed; skipping GraphML.")
        else:
            G = nx.Graph()
            for i, a in atom_df.iterrows():
                G.add_node(int(i), element=a["element"], atom=a["atom"], chain=a["chain"],
                           resid=int(a["resid"]), resname=a["resname"])
            # add undirected unique edges with mean attrs
            ei = edge_index.numpy().T
            ea = edge_attr.numpy()
            seen = set()
            for k,(u,v) in enumerate(ei):
                if u>v: u,v = v,u
                if (u,v) in seen: continue
                mask = ((ei[:,0]==u)&(ei[:,1]==v)) | ((ei[:,0]==v)&(ei[:,1]==u))
                area_mean = float(np.mean(ea[mask,1])); cd_mean = float(np.mean(ea[mask,0]))
                G.add_edge(int(u), int(v), center_distance=cd_mean, face_area=area_mean)
                seen.add((u,v))
            nx.write_graphml(G, f"{out}_graph.graphml")

    print(f"[OK] Saved {out}_graph.pt | atoms={x.size(0)} edges={edge_index.size(1)}")
    if args.debug:
        print(f"[DEBUG] Sample Voronota ids:\n  - " + "\n  - ".join(raw_id_sample))

if __name__ == "__main__":
    main()
