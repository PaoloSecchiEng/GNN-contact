import os
import ast
import random
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from torch_geometric.nn import TransformerConv
import sys

from models import *
from utils import *


sys.stdout = TeeStdout("terminal.txt")
Graph = namedtuple('Graph', ['x', 'y', 'edge_index', 'edge_features', 'global_features'])
data_set_graphs = torch.load("./../cal_ai/dataset.pth")
print(f"Loaded {len(data_set_graphs)} simulations from data_set")

labeled_graphs = []
for sim in data_set_graphs:
    labeled_graphs.append((sim, 'data_set'))
    

# -----------------------------
# Data Normalization
# -----------------------------

all_x = torch.cat([g.x for sim in data_set_graphs for g in sim], dim=0)
x_mean = all_x[:, :2].mean(dim=0)
x_std  = all_x[:, :2].std(dim=0)

all_y = torch.cat([g.y for sim in data_set_graphs for g in sim], dim=0)
_, y_mean, y_std = normalize_tensor(all_y)

all_edge = torch.cat([g.edge_features for sim in data_set_graphs for g in sim], dim=0)
_, edge_mean, edge_std = normalize_tensor(all_edge)

all_global = torch.stack([g.global_features for sim in data_set_graphs for g in sim], dim=0)
global_mean = all_global.mean(dim=0)
global_std  = all_global.std(dim=0)

normalization_params = {
    "x_mean": x_mean.tolist(),
    "x_std": x_std.tolist(),
    "y_mean": y_mean.tolist()[0],
    "y_std": y_std.tolist()[0],
    "edge_mean": edge_mean.tolist()[0],
    "edge_std": edge_std.tolist()[0],
    "global_mean": global_mean.tolist(),
    "global_std": global_std.tolist()
}
with open("utils/normalization_params_base.txt", "w") as f:
    for k, v in normalization_params.items():
        f.write(f"{k}: {v}\n")
print("Normalization parameters saved to normalization_params_base.txt")

normalized_labeled = []
for sim, src in labeled_graphs:
    sim_norm = []
    for g in sim:
        norm_x = g.x.clone()
        norm_x[:, :2] = (g.x[:, :2] - x_mean) / (x_std + 1e-8)
        norm_y = (g.y - y_mean) / (y_std + 1e-8)
        norm_edge = (g.edge_features - edge_mean) / (edge_std + 1e-8)
        norm_global = (g.global_features - global_mean) / (global_std + 1e-8)
        sim_norm.append(Graph(
            x=norm_x,
            y=norm_y,
            edge_index=g.edge_index,
            edge_features=norm_edge,
            global_features=norm_global
        ))
    normalized_labeled.append((sim_norm, src))

# -----------------------------
# Precompute top-surface mask & r_top
# -----------------------------
# Use first simulation to compute mask/r_top once
first_sim, _ = normalized_labeled[0]
sample_graph = first_sim[0]
node_pos_ref = (sample_graph.x[:, :2] * x_std + x_mean)
z_max = torch.max(node_pos_ref[:, 1])
tol = 1e-3
top_mask = (torch.abs(node_pos_ref[:, 1] - z_max) < tol)
r_top = node_pos_ref[top_mask, 0]


# -----------------------------
# Training utilities
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resume training flag (default True): if True, attempt to load checkpoints/last.pt
check = True

# Kendall parameters
s1 = nn.Parameter(torch.tensor(0.0, device=device))
s2 = nn.Parameter(torch.tensor(0.0, device=device))
def train_model(model, data, val_data, epochs, checkpoint_dir="checkpoints"):
    optimizer = optim.Adam(list(model.parameters()) + [s1, s2], lr=1e-4)
    loss_fn = nn.MSELoss()
    loss_history, val_loss_history = [], []
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    start_epoch = 0

    # Conditional resume from last.pt if check flag is True
    if check:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        if os.path.exists(last_path):
            try:
                ckpt = torch.load(last_path, map_location=device)
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
                best_val_loss = ckpt.get('val_loss', float('inf'))
                print(f"Resumed training from {last_path} at epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to resume from {last_path}: {e}")

    for ep in range(start_epoch, epochs):
        model.train()
        tot, cnt = 0.0, 0
        for sim, src in data:
            memory = None
            for g in sim:
                gi = Graph(
                    x=g.x.to(device), y=g.y.to(device),
                    edge_index=g.edge_index.to(device),
                    edge_features=g.edge_features.to(device),
                    global_features=g.global_features.to(device)
                )
                y_pred, memory = model(gi, memory)
                mse = loss_fn(y_pred, gi.y.to(device))
                # load loss (denormalized sigma_zz integral difference)
                y_pred_den = denormalize(y_pred, y_mean, y_std)
                y_gt_den   = denormalize(gi.y.to(device), y_mean, y_std)
                tm = top_mask.to(y_pred_den.device)
                if tm.sum() > 0:
                    sp = y_pred_den[tm,2]; sg = y_gt_den[tm,2]; rt = r_top.to(sp.device)
                    lp = torch.trapz(sp*(2*math.pi*rt), rt)
                    lg = torch.trapz(sg*(2*math.pi*rt), rt)
                    load_loss = (lp - lg)**2
                else:
                    load_loss = torch.tensor(0.0, device=device)
                loss = (1/(2*torch.exp(2*s1))) * mse \
                     + (1/(2*torch.exp(2*s2))) * load_loss \
                     + s1 + s2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += loss.item()
                cnt += 1
                memory = memory.detach()
        avg = tot / cnt
        loss_history.append(avg)
        print(f"Epoch {ep+1}/{epochs}, Avg Loss: {avg:.6f}, s1={s1.item():.4f}, s2={s2.item():.4f}")

        model.eval()
        with torch.no_grad():
            tot_v, cnt_v = 0.0, 0
            for sim, src in val_data:
                memory = None
                for g in sim:
                    gi = Graph(
                        x=g.x.to(device), y=g.y.to(device),
                        edge_index=g.edge_index.to(device),
                        edge_features=g.edge_features.to(device),
                        global_features=g.global_features.to(device)
                    )
                    y_pred, memory = model(gi, memory)
                    y_pred_den = denormalize(y_pred, y_mean, y_std)
                    y_gt_den   = denormalize(gi.y, y_mean, y_std)
                    tm = top_mask.to(y_pred_den.device)
                    if tm.sum() > 0:
                        sp = y_pred_den[tm,2]; sg = y_gt_den[tm,2]; rt = r_top.to(sp.device)
                        lp = torch.trapz(sp*(2*math.pi*rt), rt)
                        lg = torch.trapz(sg*(2*math.pi*rt), rt)
                        load_loss = (lp - lg)**2
                    else:
                        load_loss = torch.tensor(0.0, device=device)

                    mse = F.mse_loss(y_pred, gi.y.to(device))
                    loss = (1/(2*torch.exp(2*s1))) * mse \
                        + (1/(2*torch.exp(2*s2))) * load_loss \
                        + s1 + s2
                    tot_v += loss.item()
                    cnt_v += 1
                    memory = memory.detach()
            avg_v = tot_v / cnt_v
            val_loss_history.append(avg_v)
            print(f"           Val Loss: {avg_v:.6f}")
            
            # Save best checkpoint
            if avg_v < best_val_loss:
                best_val_loss = avg_v
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save({
                    'epoch': ep + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_v,
                    'train_loss': avg
                }, best_path)
                print(f"           Saved best checkpoint: {best_path}")
            
            # Save last.pt every 10 epochs
            if (ep + 1) % 10 == 0:
                last_path = os.path.join(checkpoint_dir, "last.pt")
                torch.save({
                    'epoch': ep + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_v,
                    'train_loss': avg
                }, last_path)
                print(f"           Saved last checkpoint: {last_path}")

    return loss_history, val_loss_history



# -----------------------------
# Train/Val Split (simple 80/20 split)
# -----------------------------
n_total = len(normalized_labeled)
n_train = int(0.9 * n_total)
data_train = normalized_labeled[:n_train]
data_val = normalized_labeled[n_train:]

print(f"\nTrain samples: {n_train}, Val samples: {n_total - n_train}")

# Initialize model and move to device
model = ElasticityTGN_MLP().to(device)

# Train
t0 = time.time()
loss_history, val_loss_history = train_model(model, data_train, data_val, epochs=1000)
elapsed = time.time() - t0
print(f"Training took {elapsed:.2f}s")

# Inference & error computation on validation set
# Precompute node areas & total area
gt_dir = "../cal_dataset/data_paper"
with open(os.path.join(gt_dir, "connectivity_sim_3_cp0_0.0800_cp1_-0.0297.txt"), "r") as f:
    conn = ast.literal_eval(f.read())
pos = node_pos_ref.numpy()
N_nodes = pos.shape[0]

def polygon_area(pts):
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

node_areas = np.zeros(N_nodes, dtype=np.float32)
A_e_list = []
for elem in conn:
    idx = [n-1 for n in elem]
    pts = pos[idx]
    area = polygon_area(pts)
    A_e_list.append(area)
    share = area / float(len(idx))
    for i in idx:
        node_areas[i] += share
A_Omega = float(sum(A_e_list))
node_areas = torch.tensor(node_areas, device=device)

mean_rel_sigma_z = []
mean_rel_dr      = []
mean_rel_dz      = []
mean_rel_load    = []

model.eval()
with torch.no_grad():
    for sim, src in data_val:
        abs_sigma_ts, max_sigma_ts, min_sigma_ts = [], [], []
        abs_dr_ts,    max_dr_ts,    min_dr_ts    = [], [], []
        abs_dz_ts,    max_dz_ts,    min_dz_ts    = [], [], []
        load_pred_ts, load_gt_ts               = [], []

        memory = None
        for g in sim:
            gi = Graph(
                x=g.x.to(device), y=g.y.to(device),
                edge_index=g.edge_index.to(device),
                edge_features=g.edge_features.to(device),
                global_features=g.global_features.to(device)
            )
            y_pred, memory = model(gi, memory)
            y_pred_den = denormalize(y_pred, y_mean, y_std)
            y_gt_den   = denormalize(gi.y,   y_mean, y_std)

            # σzz
            σ_pred, σ_gt = y_pred_den[:,2], y_gt_den[:,2]
            diffσ = torch.abs(σ_pred - σ_gt)
            Eσ_t = (diffσ * node_areas).sum().item() / A_Omega
            abs_sigma_ts.append(Eσ_t)
            max_sigma_ts.append(σ_gt.max().item())
            min_sigma_ts.append(σ_gt.min().item())

            # dr
            dr_pred, dr_gt = y_pred_den[:,0], y_gt_den[:,0]
            diff_r = torch.abs(dr_pred - dr_gt)
            Edr_t = (diff_r * node_areas).sum().item() / A_Omega
            abs_dr_ts.append(Edr_t)
            max_dr_ts.append(dr_gt.max().item())
            min_dr_ts.append(dr_gt.min().item())

            # dz
            dz_pred, dz_gt = y_pred_den[:,1], y_gt_den[:,1]
            diff_z = torch.abs(dz_pred - dz_gt)
            Edz_t = (diff_z * node_areas).sum().item() / A_Omega
            abs_dz_ts.append(Edz_t)
            max_dz_ts.append(dz_gt.max().item())
            min_dz_ts.append(dz_gt.min().item())

            # load
            tm = top_mask.to(y_pred_den.device)
            if tm.sum() > 0:
                sp, sg = σ_pred[tm], σ_gt[tm]
                rt = r_top.to(sp.device)
                Lp = torch.trapz(sp * (2 * math.pi * rt), rt).item()
                Lg = torch.trapz(sg * (2 * math.pi * rt), rt).item()
            else:
                Lp, Lg = 0.0, 0.0
            load_pred_ts.append(Lp)
            load_gt_ts.append(Lg)

            memory = memory.detach()

        mσ  = np.mean(abs_sigma_ts)
        md_r= np.mean(abs_dr_ts)
        md_z= np.mean(abs_dz_ts)
        mL  = np.mean(np.abs(np.array(load_pred_ts) - np.array(load_gt_ts)))

        Δσ = max(max_sigma_ts) - min(min_sigma_ts)
        Δr = max(max_dr_ts)    - min(min_dr_ts)
        Δz = max(max_dz_ts)    - min(min_dz_ts)
        ΔL = max(load_gt_ts)   - min(load_gt_ts)

        mean_rel_sigma_z.append(mσ/Δσ if Δσ>0 else 0.0)
        mean_rel_dr.append(   md_r/Δr if Δr>0 else 0.0)
        mean_rel_dz.append(   md_z/Δz if Δz>0 else 0.0)
        mean_rel_load.append( mL/ΔL  if ΔL>0 else 0.0)


