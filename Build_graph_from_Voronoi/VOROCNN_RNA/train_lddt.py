#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train per-residue lDDT predictor.
- Expects <prefix>_graph.pt from build_voronoi_graph.py
- Optional labels CSV: columns [res_key, lDDT] where res_key matches <prefix>_residue_index.csv
Usage:
  python train_lDDT.py data_dir --labels labels.csv --epochs 50
"""
import argparse, os, glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from model_vorocnn_like import VoroCNNLike

class VoronoiGraphSet(Dataset):
    def __init__(self, data_dir, labels_csv=None):
        self.graphs = sorted(glob.glob(os.path.join(data_dir, "*_graph.pt")))
        self.res_maps = {g: g.replace("_graph.pt", "_residue_index.csv") for g in self.graphs}
        self.labels = None
        if labels_csv and os.path.exists(labels_csv):
            # Expected: file, res_key, lDDT  (file without suffix should match prefix)
            df = pd.read_csv(labels_csv)
            # normalize prefix field
            if "file" not in df.columns:
                raise ValueError("labels.csv needs a 'file' column with structure prefix (e.g., 1AJU)")
            self.labels = df
        print(f"[INFO] Loaded {len(self.graphs)} graphs.")

    def __len__(self): return len(self.graphs)

    def __getitem__(self, i):
        gpath = self.graphs[i]
        data = torch.load(gpath)
        # Optional labels
        y = None
        if self.labels is not None:
            prefix = os.path.basename(gpath).replace("_graph.pt","")
            res_map = pd.read_csv(self.res_maps[gpath])  # res_key,res_idx
            sub = self.labels[self.labels["file"]==prefix]
            if len(sub):
                mapping = dict(zip(res_map["res_key"], res_map["res_idx"]))
                y = torch.full((res_map["res_idx"].max()+1,), float("nan"))
                for _, row in sub.iterrows():
                    rk = row["res_key"]
                    if rk in mapping and pd.notna(row["lDDT"]):
                        y[mapping[rk]] = float(row["lDDT"])
        return data, y

def collate(batch):
    datas, ys = zip(*batch)
    # No batching across structures for simplicity
    return datas, ys

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0; n = 0
    for datas, ys in loader:
        for data, y in zip(datas, ys):
            data = data.to(device)
            pred = model(data)
            if y is None or y.numel()==0 or torch.isnan(y).all():
                continue
            y = y.to(device)
            mask = ~torch.isnan(y)
            if mask.sum()==0: 
                continue
            loss = torch.nn.functional.mse_loss(pred[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
    return total / max(n,1)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0; n = 0
    for datas, ys in loader:
        for data, y in zip(datas, ys):
            data = data.to(device)
            pred = model(data)
            if y is None or y.numel()==0 or torch.isnan(y).all():
                continue
            y = y.to(device)
            mask = ~torch.isnan(y)
            if mask.sum()==0:
                continue
            loss = torch.nn.functional.mse_loss(pred[mask], y[mask])
            total += loss.item(); n += 1
    return total / max(n,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--labels", default=None, help="CSV with columns: file,res_key,lDDT")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    ds = VoronoiGraphSet(args.data_dir, labels_csv=args.labels)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)

    # Peek a sample to get dims
    first = torch.load(ds.graphs[0])
    x_dim = first.x.size(-1)
    e_dim = first.edge_attr.size(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoroCNNLike(x_dim=x_dim, e_dim=e_dim, hidden=128, n_layers=3).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, dl, opt, device)
        ev = eval_epoch(model, dl, device)
        print(f"[E{ep:03d}] train_mse={tr:.4f}  val_mse={ev:.4f}")

    torch.save(model.state_dict(), os.path.join(args.data_dir, "model_lddt.pt"))
    print("[OK] Saved model to", os.path.join(args.data_dir, "model_lddt.pt"))

if __name__ == "__main__":
    main()
