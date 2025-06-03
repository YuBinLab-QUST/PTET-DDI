import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv
from radial_basis import RadialBasis
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std
from tools import stable_norm
from torch_geometric.nn import LayerNorm

class GETLayer(nn.Module):
    '''
    Generalist Equivariant Transformer layer
    '''

    def __init__(self, d_hidden, d_edge, hid_edge, n_head, act_fn=nn.SiLU(), n_rbf=16, residual=True, cutoff=7.0):
        super(GETLayer, self).__init__()

        self.residual = residual
        self.d_head = d_hidden

        self.n_head = n_head
        self.d_edge = d_edge
        self.hid_edge = 0

        self.head_dim = d_hidden // self.n_head


        self.q_linear = nn.Linear(d_hidden, d_hidden)
        self.k_linear = nn.Linear(d_hidden, d_hidden)
        self.v_linear = nn.Linear(d_hidden, d_hidden)

        if self.d_edge != 0:
            self.hid_edge = hid_edge

        self.att_mlp = nn.Sequential(
            nn.Linear((d_hidden * 2 + n_rbf + self.hid_edge) // self.n_head, d_hidden * 4),
            act_fn,
            nn.Linear(d_hidden * 4, self.n_head),
        )

        self.ed_l = nn.Linear(n_rbf // self.n_head, self.head_dim)
        self.ind_l = nn.Linear(n_rbf // self.n_head, self.head_dim)

        self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

    def attention(self, H, Z, edge_attr, edges, block_id):

        unit_row, unit_col = edges[0], edges[1]

        H_q = H[unit_col]
        H_k = H[unit_row]

        H_q = self.q_linear(H_q).contiguous().view(H_q.shape[0], self.n_head, -1)
        H_k = self.k_linear(H_k).contiguous().view(H_k.shape[0], self.n_head, -1)


        dZ = Z[unit_row] - Z[unit_col]
        # 计算距离嵌入
        D = stable_norm(dZ, dim=-1)
        D = self.rbf(D).contiguous().view(D.shape[0], self.n_head, -1)

        if self.d_edge != 0:
            edge_attr = edge_attr.contiguous().view(edge_attr.shape[0], self.n_head, -1)
            R_repr = torch.concat([H_q, H_k, D, edge_attr], dim=-1)
        else:
            R_repr = torch.concat([H_q, H_k, D], dim=-1)

        R_repr = self.att_mlp(R_repr)

        alpha = scatter_softmax(R_repr, unit_col, dim=0)
        return alpha, (D, dZ)

    def invariant_update(self, H_v, H, alpha, D, edge_attr, edges):
        unit_row, unit_col = edges

        H_v = H_v * self.ind_l(D)

        H_agg = alpha @ H_v
        if self.d_edge != 0:
            edge_agg = (alpha @ (edge_attr.contiguous().view(edge_attr.shape[0], self.n_head, -1)))
            edge_H_agg = (edge_agg).contiguous().view(edge_attr.shape[0], -1)

            edge_attr = edge_attr + edge_H_agg if self.residual else edge_H_agg

        H_agg = scatter_sum(H_agg, unit_row, dim=0)
        H_agg = H_agg.contiguous().view(H_agg.shape[0], -1)

        H = H + H_agg if self.residual else H_agg

        return H, edge_attr

    def equivariant_update(self, H_v, Z, alpha, D, dZ, block_id, edges):
        unit_row, unit_col = edges

        H_v = H_v * self.ed_l(D)

        Z_H_agg = alpha @ H_v

        Z_agg = (Z_H_agg.unsqueeze(1) * dZ.unsqueeze(-1).unsqueeze(-1))


        Z_agg = scatter_sum((Z_H_agg.unsqueeze(1) * Z_agg), unit_row, dim=0)

        Z_agg = torch.sum(Z_agg, dim=(-2, -1))
        Z = Z + Z_agg if self.residual else Z_agg

        return Z

    def forward(self, H, Z, edge_attr, block_id, edges):

        alpha, (D, dZ) = self.attention(H, Z, edge_attr, edges, block_id)

        H_v = self.v_linear(H[edges[1]])
        H_v = H_v.contiguous().view(H_v.shape[0], self.n_head, -1)

        H, edge_attr = self.invariant_update(H_v, H, alpha, D, edge_attr, edges)

        Z = self.equivariant_update(H_v, Z, alpha, D, dZ, block_id, edges)

        return H, Z, edge_attr


class EquivariantFFN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, d_edge, hid_edge, n_rbf=16, act_fn=nn.SiLU(),
                 residual=True, dropout=0.1):
        super().__init__()
        self.residual = residual
        self.n_rbf = n_rbf
        self.d_edge = d_edge
        self.hid_edge = 0
        if self.d_edge:
            self.hid_edge = hid_edge
            self.mlp_edge = nn.Sequential(
                nn.Linear(d_in * 2 + n_rbf + self.hid_edge, d_hidden * 4),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(d_hidden * 4, self.hid_edge),
            )

        self.mlp_h = nn.Sequential(
            nn.Linear(d_in * 2 + n_rbf + self.hid_edge, d_hidden * 4),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 4, d_out),
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_in * 2 + n_rbf + self.hid_edge, d_hidden * 4),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 4, 1),
        )

        self.rbf = RadialBasis(num_radial=n_rbf, cutoff=7.0)

    def forward(self, H, Z, edge_attr, block_id, edge_id):

        radial, (Z_c, Z_o) = self._radial(Z, block_id)

        H_c = scatter_mean(H, block_id, dim=0)[block_id]
        if self.d_edge != 0:
            inputs = torch.cat([H, H_c, radial, scatter_mean(edge_attr, edge_id[0], dim=0)], dim=-1)
            edge_update = self.mlp_edge(inputs)[edge_id[0]] * edge_attr
            edge_attr = edge_attr + edge_update if self.residual else edge_update
        else:
            inputs = torch.cat([H, H_c, radial], dim=-1)

        H_update = self.mlp_h(inputs)

        H = H + H_update if self.residual else H_update

        Z_update = self.mlp_z(inputs) * Z_o

        Z = Z + Z_update if self.residual else Z_update

        return H, Z, edge_attr

    def _radial(self, Z, block_id):
        Z_c = scatter_mean(Z, block_id, dim=0)
        Z_c = Z_c[block_id]
        Z_o = Z - Z_c

        D = stable_norm(Z_o, dim=-1)
        radial = self.rbf(D)

        return radial, (Z_c, Z_o)


class EquivariantLayerNorm(nn.Module):

    def __init__(self, d_hidden, d_edge, hid_edge, n_rbf=16, cutoff=7.0, act_fn=nn.SiLU()):
        super().__init__()

        self.d_edge = d_edge
        if self.d_edge:
            self.hid_edge = hid_edge
            self.layernorm_edge = LayerNorm(self.hid_edge)

        # invariant
        self.fuse_scale_ffn = nn.Sequential(
            nn.Linear(n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden),
            # act_fn
        )
        self.layernorm_H = LayerNorm(d_hidden)

        sigma = torch.ones((1, 3))
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

    def forward(self, H, Z, edge_attr, block_id, edge_id):
        _, n_axis = Z.shape
        unit_batch_id = block_id
        unit_axis_batch_id = unit_batch_id.unsqueeze(-1).repeat(1, n_axis).flatten()  # [N * 3]

        Z_c = scatter_mean(Z, unit_batch_id, dim=0)
        Z_c = Z_c[unit_batch_id]
        Z_C_dis = Z - Z_c

        var_Ez = scatter_std(
            Z_C_dis.unsqueeze(-1).contiguous().reshape(-1, 1),
            unit_axis_batch_id, dim=0) + 1e-8

        rescale = (1 / var_Ez) * self.sigma
        Z = Z_c + Z_C_dis * rescale[unit_batch_id]

        if torch.any(torch.isnan(Z)):
            print("rescale_rbf contains NaN values！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！.")
        if torch.any(torch.isnan(H)):
            print("rescale_rbf contains NaN values！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！.")

        H = self.layernorm_H(H, block_id)
        if self.d_edge != 0:
            edge_attr = self.layernorm_edge(edge_attr, block_id[edge_id[0]])

        return H, Z, edge_attr, rescale


