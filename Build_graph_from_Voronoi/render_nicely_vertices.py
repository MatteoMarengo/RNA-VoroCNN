#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xs, ys, zs = [], [], []

with open("/root/workdir/RNA-VoroCNN/Build_graph_from_Voronoi/voronota_1.29.4415/vertices.txt") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            parts = dict(tok.split("=") for tok in line.split() if "=" in tok)
            if "x" in parts and "y" in parts and "z" in parts:
                xs.append(float(parts["x"]))
                ys.append(float(parts["y"]))
                zs.append(float(parts["z"]))

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, s=5, c="blue", alpha=0.6)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("vertices.png", dpi=300)
