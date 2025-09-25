#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter

class EdgeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, eattr):
        return self.net(eattr)

class MPNNLayer(MessagePassing):
    def __init__(self, x_dim, e_dim, out_dim):
        super().__init__(aggr='mean')
        self.edge_mlp = EdgeMLP(e_dim, out_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(x_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.upd = nn.GRUCell(out_dim, x_dim)

    def forward(self, x, edge_index, edge_attr):
        eemb = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, eemb=eemb)

    def message(self, x_j, eemb):
        m = torch.cat([x_j, eemb], dim=-1)
        return self.msg_mlp(m)

    def update(self, aggr_out, x):
        # project to x size via GRUCell-like update
        x_new = self.upd(aggr_out, x)
        return x_new

class VoroCNNLike(nn.Module):
    """
    Atom-level message passing, then pool to residues for per-residue lDDT.
    """
    def __init__(self, x_dim, e_dim, hidden=128, n_layers=3):
        super().__init__()
        self.enc = nn.Linear(x_dim, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden, e_dim, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)  # predict lDDT per residue after pooling
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        res_idx = data.res_idx  # [num_atoms] -> residue ids [0..R-1]
        x = torch.relu(self.enc(x))
        for layer, ln in zip(self.layers, self.norms):
            x = ln(layer(x, edge_index, edge_attr))
        # pool atoms to residues (mean)
        res_x = scatter(x, res_idx, dim=0, reduce='mean')  # [R, hidden]
        pred_res = self.head(res_x).squeeze(-1)            # [R]
        return pred_res
