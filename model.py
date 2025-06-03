#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool,LayerNorm
from torch_scatter import  scatter_mean

from layers import GETLayer, EquivariantLayerNorm, EquivariantFFN



class GET(nn.Module):
    '''Generalist Equivariant Transformer'''

    def __init__(self, d_hidden, n_rbf, d_edge, hid_edge, n_layers=3, n_head=4,
                 act_fn=nn.SiLU(), residual=True, dropout=0.5, z_requires_grad=True, pre_norm=True, cutoff=7.0):
        super().__init__()

        self.pre_norm = pre_norm
        self.n_layers = n_layers
        self.d_edge = d_edge
        if self.d_edge != 0:
            self.hid_edge = hid_edge
            self.edge_norm = LayerNorm(d_edge)
            self.edge_linear = nn.Linear(d_edge, self.hid_edge)

        if self.pre_norm:
            self.pre_layernorm = EquivariantLayerNorm(d_hidden, d_edge, hid_edge, n_rbf, cutoff, act_fn)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', GETLayer(d_hidden, d_edge, hid_edge, n_head, act_fn, n_rbf, residual))
            self.add_module(f'layernorm0_{i}', EquivariantLayerNorm(d_hidden, d_edge, hid_edge, n_rbf, cutoff, act_fn))
            self.add_module(f'ffn_{i}', EquivariantFFN(d_hidden, 2 * d_hidden, d_hidden, d_edge, hid_edge, n_rbf, act_fn, residual,
                                                       dropout))
            self.add_module(f'layernorm1_{i}', EquivariantLayerNorm(d_hidden, d_edge, hid_edge, n_rbf, cutoff, act_fn))


    def recover_scale(self, Z, block_id, record_scale):
        with torch.no_grad():
            unit_batch_id = block_id
        Z_c = scatter_mean(Z, unit_batch_id, dim=0)
        Z_c = Z_c[unit_batch_id]
        Z_centered = Z - Z_c

        Z = Z_c + Z_centered / record_scale[unit_batch_id]

        return Z

    def forward(self, drug):

        record_scale = torch.ones((drug.num_graphs, 3), dtype=torch.float, device=drug.pos.device)

        if self.d_edge != 0:
            edge_id = drug.batch[drug.edge_index[0]]
            drug.edge_attr = self.edge_norm(drug.edge_attr, edge_id)
            drug.edge_attr = self.edge_linear(drug.edge_attr)

        if self.pre_norm:
            drug.x, drug.pos, drug.edge_attr, rescale = self.pre_layernorm(drug.x, drug.pos, drug.edge_attr, drug.batch, drug.edge_index)
            record_scale *= rescale

        for i in range(self.n_layers):
            # old_drug = drug
            drug.x, drug.pos, drug.edge_attr = self._modules[f'layer_{i}'](drug.x, drug.pos, drug.edge_attr, drug.batch, drug.edge_index)
            drug.x, drug.pos, drug.edge_attr, rescale = self._modules[f'layernorm0_{i}'](drug.x, drug.pos, drug.edge_attr, drug.batch, drug.edge_index)
            record_scale *= rescale
            drug.x, drug.pos, drug.edge_attr = self._modules[f'ffn_{i}'](drug.x, drug.pos, drug.edge_attr, drug.batch, drug.edge_index)
            drug.x, drug.pos, drug.edge_attr, rescale = self._modules[f'layernorm1_{i}'](drug.x, drug.pos, drug.edge_attr, drug.batch, drug.edge_index)
            record_scale *= rescale
            # drug.x = old_drug.x + drug.x            # 使用残差连接大法
            # drug.pos = old_drug.pos + drug.pos

        drug.pos = self.recover_scale(drug.pos, drug.batch, record_scale)

        maxp = global_max_pool(drug.x, drug.batch)
        meanp = global_mean_pool(drug.x, drug.batch)
        drug = torch.concat((maxp, meanp), dim=-1).unsqueeze(-2)
        return drug



class pre_chemBERTa(nn.Module):
    def __init__(self, d_hidden, pre_hidden, act_fn, dropout):
        super().__init__()

        self.pre_linear = nn.Linear(pre_hidden, d_hidden)
        self.gru = nn.GRU(input_size=d_hidden * 2, hidden_size=d_hidden * 2, num_layers=3, dropout=dropout, batch_first=True)

    def forward(self, batch_smiles):

        batch_smiles.x = self.pre_linear(batch_smiles.x)

        max_pool = global_max_pool(batch_smiles.x, batch_smiles.batch)
        mean_pool = global_mean_pool(batch_smiles.x, batch_smiles.batch)

        all_outputs = torch.concat((max_pool, mean_pool), dim=-1)
        all_outputs = self.gru(all_outputs.unsqueeze(1))
        outputs = all_outputs[0]

        return outputs


class PTETDDI(nn.Module):
    def __init__(self, d_atom, d_hidden, n_rbf, d_edge, n_layers=3, n_head=4, dropout=0.1,
                 act_fn=nn.SiLU(), residual=True, z_requires_grad=True, pre_norm=True, sparse_k=3, pre_hidden=879,
                 rel_total=86):
        super().__init__()
        self.init_norm = LayerNorm(d_atom)
        self.init_linear = nn.Linear(d_atom, d_hidden)

        # 加入键特征
        self.d_edge = d_edge
        self.hid_edge = d_hidden // 2

        self.GET = GET(d_hidden, n_rbf, d_edge=self.d_edge, hid_edge=self.hid_edge, n_layers=n_layers, n_head=4, act_fn=act_fn, residual=True, dropout=dropout)

        self.pre_tmodel = pre_chemBERTa(d_hidden, pre_hidden, act_fn, dropout)


        self.mlp_score = nn.Sequential(
            nn.Linear(d_hidden * 8, d_hidden * 8 * 2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 8 * 2, 1)
        )

    def forward(self, triples):
        h_data, t_data, chem_h, chem_t, rels = triples

        h_data.x = self.init_norm(h_data.x, h_data.batch)
        t_data.x = self.init_norm(t_data.x, t_data.batch)
        h_data.x = self.init_linear(h_data.x)
        t_data.x = self.init_linear(t_data.x)

        get_h = self.GET(h_data)
        get_t = self.GET(t_data)


        chem_h = self.pre_tmodel(chem_h)
        chem_t = self.pre_tmodel(chem_t)


        final_h = torch.concat((get_h, chem_h), dim=-1)
        final_t = torch.concat((get_t, chem_t), dim=-1)

        scores = self.mlp_score(torch.cat((final_h, final_t), dim=-1))
        scores = scores.squeeze(-1).squeeze(-1)
        return scores  # 不要瞎猜

