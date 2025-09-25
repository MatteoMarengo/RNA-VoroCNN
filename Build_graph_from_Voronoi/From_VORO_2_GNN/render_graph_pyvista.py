#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data import Data
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def load_pyg(path):
    add_safe_globals([Data])           # allowlist for PyTorch 2.6+
    return torch.load(path, map_location="cpu", weights_only=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_pt")
    ap.add_argument("--out", default="graph.png")
    ap.add_argument("--point_size", type=float, default=6.0)
    ap.add_argument("--line_width", type=float, default=0.5)
    ap.add_argument("--max_edges", type=int, default=100000, help="downsample if graph is huge")
    args = ap.parse_args()

    data = load_pyg(args.graph_pt)
    pos = data.pos.cpu().numpy()              # [N,3]
    ei  = data.edge_index.cpu().numpy().T     # [E,2]

    # unique undirected edges
    edges = {tuple(sorted(map(int, e))) for e in ei if e[0] != e[1]}
    edges = sorted(edges)
    if len(edges) > args.max_edges:
        # uniform downsample to keep rendering fast
        step = max(1, len(edges)//args.max_edges)
        edges = edges[::step]

    fig = plt.figure(figsize=(8,8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    # draw nodes
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=args.point_size, alpha=0.9)

    # draw edges
    lw = args.line_width
    for a,b in edges:
        xs = [pos[a,0], pos[b,0]]
        ys = [pos[a,1], pos[b,1]]
        zs = [pos[a,2], pos[b,2]]
        ax.plot(xs, ys, zs, linewidth=lw, alpha=0.6)

    ax.set_axis_off()
    # auto-zoom
    mins = pos.min(axis=0); maxs = pos.max(axis=0)
    ctr = (mins + maxs) / 2.0
    size = (maxs - mins).max() * 0.6
    ax.set_xlim(ctr[0]-size, ctr[0]+size)
    ax.set_ylim(ctr[1]-size, ctr[1]+size)
    ax.set_zlim(ctr[2]-size, ctr[2]+size)

    plt.tight_layout(pad=0)
    plt.savefig(args.out, bbox_inches='tight', pad_inches=0)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
