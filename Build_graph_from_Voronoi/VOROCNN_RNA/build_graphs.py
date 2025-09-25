#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From an RNA mmCIF:
  1) run Voronota core pipeline to get annotated contacts
  2) parse contacts + CIF to make an atom-level graph
  3) pool map (atom -> residue) for residue-level targets
Outputs:
  - <prefix>_contacts.txt        (Voronota contacts)
  - <prefix>_graph.pt            (PyG Data object with atom graph)
  - <prefix>_atom_index.csv      (atom index mapping)
  - <prefix>_residue_index.csv   (residue index mapping)
Usage:
  python build_voronoi_graph.py 1AJU.cif --probe 1.4 --keep-h --keep-het
"""
import argparse, os, subprocess, json, math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

def run(cmd, check=True):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}\nSTDERR:\n{p.stderr}")
    return p.stdout

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif")
    ap.add_argument("--probe", type=float, default=1.4)      # SAS probe, Å
    ap.add_argument("--keep-h", action="store_true")
    ap.add_argument("--keep-het", action="store_true")
    ap.add_argument("--prefix", default=None)
    return ap.parse_args()

def vdw_radius(element:str)->float:
    # Minimal set; expand as needed
    table = {
        "H":1.20, "C":1.70, "N":1.55, "O":1.52, "P":1.80, "S":1.80,
        "F":1.47, "Cl":1.75, "BR":1.85, "Br":1.85, "I":1.98, "MG":1.73, "Mg":1.73, "NA":2.27, "Na":2.27, "K":2.75
    }
    return table.get(element.capitalize(), 1.8)

def load_cif_atoms(cif_path):
    """Return list of atom dicts with keys: idx, chain, resname, resid, icode, atom, element, het, x,y,z, bfactor."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("st", cif_path)
    atoms = []
    idx = 0
    for model in structure:
        for chain in model:
            for res in chain:
                het = res.id[0].strip() != ""  # HETATM if True
                resseq = res.id[1]
                icode = res.id[2].strip() if res.id[2] != " " else ""
                for atom in res:
                    element = atom.element.strip() or atom.get_name()[0]
                    atoms.append({
                        "idx": idx,
                        "chain": chain.id, "resname": res.get_resname(), "resid": int(resseq),
                        "icode": icode, "atom": atom.get_name(), "element": element,
                        "het": het, "x": atom.coord[0], "y": atom.coord[1], "z": atom.coord[2],
                        "bfactor": float(atom.get_bfactor())
                    })
                    idx += 1
    return atoms

def write_temp_contacts(cif_path, prefix, keep_h, keep_het, probe):
    # atoms -> balls
    flags = []
    if keep_h: flags.append("--include-hydrogens")
    if keep_het: flags.append("--include-heteroatoms")
    cmd1 = f"voronota get-balls-from-atoms-file --annotated {' '.join(flags)} < {cif_path} > {prefix}_balls.txt"
    run(cmd1)

    # optional pre-filter: drop solvent & common ions
    cmdf = f"voronota query-balls --match-tags-not 'solvent,ion' < {prefix}_balls.txt > {prefix}_balls_nosolv.txt"
    run(cmdf)

    # balls -> contacts
    cmd2 = f"voronota calculate-contacts --annotated --probe {probe} < {prefix}_balls_nosolv.txt > {prefix}_contacts.txt"
    run(cmd2)
    return f"{prefix}_contacts.txt"

def parse_contacts(path):
    """
    Voronota annotated contacts lines. We’ll parse minimal fields:
    - atomA, atomB identifiers (contain chain/res/atom)
    - center distance (cd)
    - face area (area)
    - flags: solvent etc. (we already filtered)
    """
    edges = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): 
                continue
            # Lines are space-separated 'key=value' tokens; keep robust parsing.
            toks = [t for t in line.strip().split() if "=" in t]
            rec = {}
            for t in toks:
                k,v = t.split("=",1)
                rec[k]=v
            # We only keep non-solvent atom-atom contacts
            if rec.get("annotation1","")=="solvent" or rec.get("annotation2","")=="solvent":
                continue
            # Extract identifiers like chain/resname/resid/atom
            id1 = rec.get("id1","")
            id2 = rec.get("id2","")
            # numeric features
            cd = float(rec.get("center_distance","0"))
            area = float(rec.get("area","0"))
            if area<=0: 
                continue
            edges.append((id1,id2,cd,area))
    return edges

def build_mappings(atoms):
    # Build lookup by a stable key similar to Voronota annotation (chain:resid:atom)
    # We'll allow loose matching by chain/resid/atom name.
    by_key = {}
    for a in atoms:
        key = f"{a['chain']}:{a['resid']}:{a['atom']}"
        by_key[key]=a["idx"]
    return by_key

def atom_features(atom):
    # Element one-hot (C,N,O,P) + generic bins, residue type (RNA bases), VDW radius, B-factor
    el = atom["element"].upper()
    base = atom["resname"].strip().upper()
    elem_vec = np.array([el=="C", el=="N", el=="O", el=="P"], dtype=float)
    # RNA base one-hot: A,C,G,U + others
    base_vec = np.array([base=="A", base=="C", base=="G", base in ("U","URA","U5P")], dtype=float)
    vdw = vdw_radius(el)
    b = atom["bfactor"]/50.0  # rough scale
    het = 1.0 if atom["het"] else 0.0
    return np.concatenate([elem_vec, base_vec, [vdw, b, het]], axis=0)

def main():
    args = parse_args()
    prefix = args.prefix or os.path.splitext(os.path.basename(args.cif))[0]
    contacts_path = write_temp_contacts(args.cif, prefix, args.keep_h, args.keep_het, args.probe)

    atoms = load_cif_atoms(args.cif)
    atom_df = pd.DataFrame(atoms)
    atom_df.to_csv(f"{prefix}_atom_index.csv", index=False)

    # residue index map
    res_keys = []
    for (_, chain, resid, icode, resname) in atom_df[['chain','resid','icode','resname']].drop_duplicates().itertuples():
        key = f"{chain}:{resid}:{icode or ''}:{resname}"
        res_keys.append(key)
    res_keys = sorted(set(res_keys))
    res_index = {k:i for i,k in enumerate(res_keys)}
    pd.DataFrame({"res_key":res_keys, "res_idx":list(range(len(res_keys)))}).to_csv(f"{prefix}_residue_index.csv", index=False)

    key2idx = build_mappings(atoms)
    edges_raw = parse_contacts(contacts_path)

    # Build edge_index and edge_attr
    rows, cols = [], []
    eattr = []  # [center_distance, area]
    missing = 0
    for k1,k2,cd,area in edges_raw:
        # Voronota ids are like 'A:123:ATOMNAME' or similar depending on build; try several fallbacks
        def try_keys(vkey):
            if vkey in key2idx: return key2idx[vkey]
            # Sometimes atom names have spacing (e.g., O5'), unify common RNA primes
            vkey2 = vkey.replace("*","'")  # normalize
            return key2idx.get(vkey2, None)

        i = try_keys(k1)
        j = try_keys(k2)
        if i is None or j is None:
            missing += 1
            continue
        rows.append(i); cols.append(j)
        eattr.append([cd, area])
        # undirected graph
        rows.append(j); cols.append(i)
        eattr.append([cd, area])

    if not rows:
        raise RuntimeError("No edges constructed; check contact parsing or identifiers.")
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32)

    # Node features
    X = np.vstack([atom_features(a) for a in atoms])
    x = torch.tensor(X, dtype=torch.float32)

    # Positions (for potential geometric ops)
    pos = torch.tensor(atom_df[['x','y','z']].values, dtype=torch.float32)

    # Atom->residue pooling map
    res_idx = []
    for a in atoms:
        rkey = f"{a['chain']}:{a['resid']}:{a['icode'] or ''}:{a['resname']}"
        res_idx.append(res_index[rkey])
    res_idx = torch.tensor(res_idx, dtype=torch.long)

    # Coalesce duplicates
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=x.size(0), reduce='mean')

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.res_idx = res_idx  # atom->residue map
    torch.save(data, f"{prefix}_graph.pt")

    print(f"[OK] Graph saved: {prefix}_graph.pt")
    print(f"[INFO] Atoms: {x.size(0)}, Edges: {edge_index.size(1)}, Residues: {len(res_index)} (missing edges mapped: {missing})")

if __name__ == "__main__":
    main()
