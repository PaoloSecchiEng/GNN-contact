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
from sklearn.model_selection import KFold


class ElasticityTGN_MLP(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=2,
                 latent_dim=64, num_messages=3, kernels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # node & edge encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )

        # per-step message & update MLPs
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim),  
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim),  
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])

        # global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(3, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # edge-augmentation MLP 
        self.edge_infer_mlp = nn.Sequential(
            nn.Linear(2*latent_dim + 2, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 1), nn.Sigmoid()
        )

        # persistent memory buffer & GRUCell
        self.register_buffer("memory", torch.zeros(1, latent_dim))
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, graph, memory=None, return_aug_edges=False):
        # initialize memory if first timestep
        if memory is None:
            memory = self.memory.repeat(graph.x.size(0), 1)

        # encode nodes + add global context
        x_enc = self.node_encoder(graph.x)  # [N, latent_dim]
        glob = self.global_encoder(
            graph.global_features.view(1, -1)
        ).expand_as(x_enc)                 # [N, latent_dim]
        x_enc = x_enc + glob

        # encode edges
        e_enc = self.edge_encoder(graph.edge_features)  # [E, latent_dim]

        # message passing
        src, tgt = graph.edge_index
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            h_src = x_enc[src]                                 # [E, latent_dim]
            m_ij = msg_mlp(torch.cat([h_src, e_enc], dim=1))   # [E, latent_dim]
            m_v = torch.zeros_like(x_enc)                      # [N, latent_dim]
            m_v.index_add_(0, tgt, m_ij)                       # sum messages
            h_v = x_enc + upd_mlp(torch.cat([x_enc, m_v], dim=1))  # [N, latent_dim]
            x_enc = F.relu(h_v)

        # update memory & decode
        memory = self.gru(x_enc, memory)   # [N, latent_dim]
        out = self.decoder(memory)         

        if return_aug_edges:
            # placeholder for future edge-augmentation output
            return out, memory, (None, None)
        return out, memory


class ElasticityTGN_TC(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=2,
                 latent_dim=64, num_messages=3, heads=4):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            TransformerConv(latent_dim, latent_dim // heads,
                            heads=heads, edge_dim=edge_in_dim,
                            concat=True, dropout=0.1)
            for _ in range(num_messages)
        ])
        self.global_encoder = nn.Sequential(
            nn.Linear(3, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.edge_infer_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim + 2, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 1), nn.Sigmoid()
        )
        self.register_buffer("memory", torch.zeros(1, latent_dim))
        self.gru = nn.GRUCell(latent_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, graph, memory=None):
        if memory is None:
            memory = self.memory.repeat(graph.x.size(0), 1)

        x_enc = self.node_enc(graph.x)
        glob = self.global_encoder(graph.global_features.view(1, -1))
        glob = glob.expand(x_enc.size(0), -1)
        x_enc = x_enc + glob

        e = graph.edge_features  # [E, 2]

        for conv in self.convs:
            x_enc = F.relu(conv(x_enc, graph.edge_index, e))

        memory = self.gru(x_enc, memory)
        out = self.decoder(memory)

        return out, memory


class ElasticityTGN_FR(nn.Module):
    def __init__(self,
                 node_in_dim=3,
                 edge_in_dim=2,
                 latent_dim=64,
                 num_messages=3,
                 kernels=4,      # kept for compatibility, unused
                 k_extra=8):     # kept for compatibility, unused
        super().__init__()
        self.latent_dim = latent_dim
        self.k_extra = k_extra

        # node & edge encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )

        # per-step message & update MLPs
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim),  
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim), 
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])

        # global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(3, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # memory + GRU
        self.register_buffer("memory", torch.zeros(1, latent_dim))
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        # final decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, graph, memory=None, return_aug_edges=False):
        if memory is None:
            memory = self.memory.repeat(graph.x.size(0), 1)

        # encode node features + add global context
        x_enc = self.node_encoder(graph.x)
        glob = self.global_encoder(graph.global_features.view(1, -1))
        glob = glob.expand(x_enc.size(0), -1)
        x_enc = x_enc + glob

        # fixed-radius edge augmentation
        coords = graph.x[:, :2]
        N = x_enc.size(0)
        with torch.no_grad():
            dist = torch.cdist(coords, coords)
            r_extra = 0.1
            mask = (dist < r_extra) & (~torch.eye(N, dtype=torch.bool, device=coords.device))
            src_extra, tgt_extra = mask.nonzero(as_tuple=True)
            delta = coords[tgt_extra] - coords[src_extra]

        # build augmented graph
        edge_index = torch.cat([graph.edge_index,
                                torch.stack([src_extra, tgt_extra], dim=0)],
                               dim=1)
        edge_features = torch.cat([graph.edge_features, delta], dim=0)
        e_enc = self.edge_encoder(edge_features)

        # message passing without torch_scatter
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            src, tgt = edge_index
            h_src = x_enc[src]
            # compute per-edge messages
            m_ij = msg_mlp(torch.cat([h_src, e_enc], dim=1))
            # sum-aggregate using index_add_
            m_v = torch.zeros_like(x_enc)
            m_v.index_add_(0, tgt, m_ij)
            # update with residual
            h_v = x_enc + upd_mlp(torch.cat([x_enc, m_v], dim=1))
            x_enc = F.relu(h_v)

        # update memory and decode
        memory = self.gru(x_enc, memory)
        out = self.decoder(memory)

        if return_aug_edges:
            return out, memory, (src_extra, tgt_extra)
        return out, memory

class ElasticityTGN_LE(nn.Module):
    def __init__(self,
                 node_in_dim=3,
                 edge_in_dim=2,
                 latent_dim=64,
                 num_messages=3,
                 kernels=4,          # unused but kept for compatibility
                 k_extra=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_extra = k_extra

        # node & edge input encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )

        # message & update MLPs per step ---
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim),  # h_u + e_uv
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])
        self.upd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, latent_dim),  # h_v + m_v
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            )
            for _ in range(num_messages)
        ])

        # global context encoder 
        self.global_encoder = nn.Sequential(
            nn.Linear(3, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # edge augmentation inference 
        self.edge_infer_mlp = nn.Sequential(
            nn.Linear(2*latent_dim + 2, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 1), nn.Sigmoid()
        )

        # memory & GRU 
        self.register_buffer("memory", torch.zeros(1, latent_dim))
        self.gru = nn.GRUCell(latent_dim, latent_dim)

        # decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(self, graph, memory=None, return_aug_edges=False):
        if memory is None:
            memory = self.memory.repeat(graph.x.size(0), 1)

        # encode node features + global context
        x_enc = self.node_encoder(graph.x)
        glob = self.global_encoder(graph.global_features.view(1, -1))
        glob = glob.expand(x_enc.size(0), -1)
        x_enc = x_enc + glob

        
        coords = graph.x[:, :2]
        N = x_enc.size(0)
        with torch.no_grad():
            dist = torch.cdist(coords, coords)
            _, idx = dist.topk(self.k_extra+1, largest=False)
        src_extra = torch.arange(N, device=coords.device).unsqueeze(1).repeat(1, self.k_extra+1)
        tgt_extra = idx
        mask = tgt_extra != src_extra
        src_extra = src_extra[mask]
        tgt_extra = tgt_extra[mask]
        h_i = x_enc[src_extra]
        h_j = x_enc[tgt_extra]
        delta = coords[tgt_extra] - coords[src_extra]
        infer_in = torch.cat([h_i, h_j, delta], dim=1)
        weights = self.edge_infer_mlp(infer_in).squeeze()
        keep = weights > 0.5
        src_extra = src_extra[keep]
        tgt_extra = tgt_extra[keep]
        delta = delta[keep]

        edge_index = torch.cat([graph.edge_index,
                                torch.stack([src_extra, tgt_extra], dim=0)],
                               dim=1)
        edge_features = torch.cat([graph.edge_features, delta], dim=0)
        e_enc = self.edge_encoder(edge_features)

        #  message passing without torch_scatter ---
        for msg_mlp, upd_mlp in zip(self.msg_mlps, self.upd_mlps):
            src, tgt = edge_index  # src -> tgt
            h_src = x_enc[src]
            # compute per-edge messages
            m_ij = msg_mlp(torch.cat([h_src, e_enc], dim=1))
            # sum-aggregate into m_v using index_add_
            m_v = torch.zeros_like(x_enc)
            m_v.index_add_(0, tgt, m_ij)
            # update with residual
            h_v = x_enc + upd_mlp(torch.cat([x_enc, m_v], dim=1))
            x_enc = F.relu(h_v)

        # update memory via GRUCell and decode
        memory = self.gru(x_enc, memory)
        out = self.decoder(memory)

        if return_aug_edges:
            return out, memory, (src_extra, tgt_extra)
        return out, memory