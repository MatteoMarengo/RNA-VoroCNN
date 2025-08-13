#!/usr/bin/env python3
import argparse, re, itertools, json
from collections import defaultdict
import numpy as np
import networkx as nx

RNA_RESNAMES = set([
    "A","U","G","C","DA","DT","DG","DC",  # in case of DNA/RNA hybrids
    # common modified nucleotides (extend as needed)
    "PSU","H2U","M2G","7MG","OMG","1MA","5MC","5MU","OMC","MIA","QUE","Y1G","Y2G","YMP"
])

ann_re = re.compile(r"^(?P<ann>\S+)\s+(?P<x>-?\d+\.\d+)\s+(?P<y>-?\d+\.\d+)\s+(?P<z>-?\d+\.\d+)\s+(?P<r>-?\d+\.\d+)\s+(?P<tags>.*)$")

def parse_annotation(ann):
    # Voronota annotation looks like: c<A>r<42>a<1>R<GLY>A<N> ...
    # We'll extract chain, resSeq, resName, atomName if present.
    d = {"chain": None, "resseq": None, "resname": None, "atom": None}
    # chain
    m = re.search(r"c<([^>]+)>", ann)
    if m: d["chain"] = m.group(1)
    # residue index
    m = re.search(r"r<([^>]+)>", ann)
    if m: d["resseq"] = m.group(1)
    # residue name
    m = re.search(r"R<([^>]+)>", ann)
    if m: d["resname"] = m.group(1)
    # atom name
    m = re.search(r"A<([^>]+)>", ann)
    if m: d["atom"] = m.group(1)
    # hetero/solvent tags may appear in tags; we’ll keep raw ann too
    return d

def load_annotated_balls(path, rna_only=False):
    atoms = []
    with open(path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            if not line or line.startswith("#"): continue
            m = ann_re.match(line)
            if not m:
                continue
            ann = m.group("ann")
            x,y,z,r = map(float,[m.group("x"), m.group("y"), m.group("z"), m.group("r")])
            tags = m.group("tags")
            meta = parse_annotation(ann)
            if rna_only:
                if not meta["resname"] or meta["resname"] not in RNA_RESNAMES:
                    # skip non-RNA atoms (ligands/proteins/solvent) when requested
                    atoms.append(None)  # placeholder to keep indexing stable
                    continue
            atoms.append({"idx": i, "ann": ann, "x":x, "y":y, "z":z, "r":r,
                          "chain": meta["chain"], "resseq": meta["resseq"],
                          "resname": meta["resname"], "atom": meta["atom"], "tags": tags})
    return atoms

def load_vertices(path):
    verts=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=line.split()
            if len(parts)<8: continue
            q = tuple(map(int, parts[0:4]))
            x,y,z,r = map(float, parts[4:8])
            verts.append({"atoms": q, "x":x, "y":y, "z":z, "r":r})
    return verts

def build_graph(atoms, vertices):
    G = nx.Graph()
    # add nodes
    for vid, v in enumerate(vertices):
        q = v["atoms"]
        G.add_node(vid,
                   pos=(v["x"], v["y"], v["z"]),
                   r=v["r"],
                   atoms=q)
        # Attach per-atom metadata (chain/res/atom/resname). Keep None if missing.
        meta=[]
        for aidx in q:
            if 0 <= aidx < len(atoms) and atoms[aidx] is not None:
                a=atoms[aidx]
                meta.append({"idx": aidx, "chain": a["chain"], "resseq": a["resseq"],
                             "resname": a["resname"], "atom": a["atom"]})
            else:
                meta.append({"idx": aidx, "chain": None, "resseq": None, "resname": None, "atom": None})
        G.nodes[vid]["atom_meta"]=meta

        # convenience: residue set (e.g., for pooling)
        residues=set()
        for m in meta:
            if m["chain"] is not None and m["resseq"] is not None:
                residues.add((m["chain"], m["resseq"]))
        G.nodes[vid]["residues"]=sorted(list(residues))

    # build adjacency via shared triplets
    triplet_map=defaultdict(list)
    for vid, v in enumerate(vertices):
        a,b,c,d = v["atoms"]
        for comb in itertools.combinations(sorted([a,b,c,d]),3):
            triplet_map[comb].append(vid)
    # connect vertices that share a triplet
    for comb, vids in triplet_map.items():
        if len(vids) > 1:
            for u,v in itertools.combinations(vids,2):
                # optional: weight edge by something (e.g., mean of radii)
                if not G.has_edge(u,v):
                    G.add_edge(u,v)

    return G

def dump_npz(G, out_npz):
    # Minimal arrays example for DL: node positions, node radii, edge index
    N = G.number_of_nodes()
    node_xyz = np.zeros((N,3), dtype=np.float32)
    node_r   = np.zeros((N,), dtype=np.float32)
    for i,(nid, data) in enumerate(G.nodes(data=True)):
        node_xyz[i,:] = np.array(data["pos"], dtype=np.float32)
        node_r[i]     = data["r"]
        G.nodes[nid]["_row"]=i
    # edges
    edges=[]
    for u,v in G.edges():
        edges.append([G.nodes[u]["_row"], G.nodes[v]["_row"]])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2,0), dtype=np.int64)
    np.savez(out_npz, node_xyz=node_xyz, node_r=node_r, edge_index=edge_index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balls", required=True, help="annotated_balls.txt from Voronota (annotated mode)")
    ap.add_argument("--verts", required=True, help="vertices.txt from Voronota calculate-vertices")
    ap.add_argument("--out",   required=True, help="output NetworkX gpickle path, e.g., graph.gpickle")
    ap.add_argument("--npz",   default=None, help="optional npz dump for DL")
    ap.add_argument("--rna-only", action="store_true", help="keep only RNA atoms when building graph")
    args = ap.parse_args()

    atoms = load_annotated_balls(args.balls, rna_only=args.rna_only)
    vertices = load_vertices(args.verts)

    # If we filtered atoms (rna-only), some atom indices point to None; we still build the graph,
    # but node features may be partially missing. That mirrors “RNA-only” focus with neighbors present.
    G = build_graph(atoms, vertices)

    nx.write_gpickle(G, args.out)
    if args.npz:
        dump_npz(G, args.npz)
    print(f"Wrote graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    main()
