import torch
import torch.nn as nn
from torch_scatter import scatter

from diffcsp.pl_modules.cspnet import CSPNet, MAX_ATOMIC_NUM


class CSPNetIMF(CSPNet):
    """Shared CSPNet trunk with dual u/v heads for iMF."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_dim = self.atom_latent_emb.out_features
        lattice_dim = self.lattice_out.out_features

        self.u_coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.u_lattice_out = nn.Linear(hidden_dim, lattice_dim, bias=False)
        self.v_coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.v_lattice_out = nn.Linear(hidden_dim, lattice_dim, bias=False)

        if self.pred_type:
            type_out_dim = MAX_ATOMIC_NUM if self.type_encoding is None else self.type_encoding.out_dim
            self.u_type_out = nn.Linear(hidden_dim, type_out_dim)
            self.v_type_out = nn.Linear(hidden_dim, type_out_dim)

        if self.pred_scalar:
            self.u_scalar_out = nn.Linear(hidden_dim, 1)
            self.v_scalar_out = nn.Linear(hidden_dim, 1)

    def _encode_features(
        self,
        t,
        atom_types,
        frac_coords,
        lattices_rep,
        num_atoms,
        node2graph,
        lattices_mat=None,
        cemb=None,
        guide_indicator=None,
    ):
        if lattices_mat is None:
            lattices_mat = lattices_rep

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices_mat, node2graph)
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            if cemb is not None:
                cemb_mixin = self._modules[f"cemb_mixin_{i}"]
                cemb_adapter = self._modules[f"cemb_adapter_{i}"]
                cemb_bias = (cemb_mixin(cemb_adapter(cemb)) * guide_indicator[:, None]).repeat_interleave(num_atoms, dim=0)
                node_features = node_features + cemb_bias

            csp_layer = self._modules[f"csp_layer_{i}"]
            node_features = csp_layer(
                node_features,
                frac_coords,
                lattices_rep,
                edges,
                edge2graph,
                num_atoms=num_atoms,
                frac_diff=frac_diff,
                lattices_mat=lattices_mat,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')
        return node_features, graph_features

    def _project_head(self, prefix, node_features, graph_features, lattices_rep):
        coord_out = getattr(self, f"{prefix}_coord_out")(node_features)

        if self.pred_scalar:
            return getattr(self, f"{prefix}_scalar_out")(graph_features)

        lattice_out = getattr(self, f"{prefix}_lattice_out")(graph_features)
        lattice_out = lattice_out.view(lattices_rep.shape)
        if self.ip:
            lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices_rep)

        if self.pred_type:
            type_out = getattr(self, f"{prefix}_type_out")(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices_rep,
        num_atoms,
        node2graph,
        lattices_mat=None,
        cemb=None,
        guide_indicator=None,
        return_aux=True,
    ):
        node_features, graph_features = self._encode_features(
            t=t,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices_rep=lattices_rep,
            num_atoms=num_atoms,
            node2graph=node2graph,
            lattices_mat=lattices_mat,
            cemb=cemb,
            guide_indicator=guide_indicator,
        )
        u_out = self._project_head("u", node_features, graph_features, lattices_rep)
        if not return_aux:
            return u_out
        v_out = self._project_head("v", node_features, graph_features, lattices_rep)
        return u_out, v_out

    def forward_u(self, *args, **kwargs):
        kwargs["return_aux"] = False
        return self.forward(*args, **kwargs)