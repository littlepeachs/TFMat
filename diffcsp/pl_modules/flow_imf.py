import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import jvp
from torch_scatter import scatter
from tqdm import tqdm

import hydra
from diffcsp.common.data_utils import lattice_params_to_matrix_torch, lattice_polar_build_torch
from diffcsp.pl_modules.conditioning import MultiEmbedding
from diffcsp.pl_modules.flow import BaseModule, DirectUnsqueezeTime, SinusoidalTimeEmbeddings
from diffcsp.pl_modules.hungarian import HungarianMatcher
from diffcsp.pl_modules.lattice_utils import LatticeDecompNN
from diffcsp.pl_modules.symmetrize import SymmetrizeRotavg
from diffcsp.pl_modules.time_scheduler import TimeScheduler
from diffcsp.pl_modules.type_module import TypeTableModule

MAX_ATOMIC_NUM = 100
metriclogger = logging.getLogger("metrics")


class CSPFlowIMF(BaseModule):
    """Orthodox iMF adaptation built on CrystalFlow crystal states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.time_dim == 0:
            self.time_dim = 1
            self.time_embedding_r = DirectUnsqueezeTime()
            self.time_embedding_t = DirectUnsqueezeTime()
        else:
            self.time_dim = self.hparams.time_dim
            self.time_embedding_r = SinusoidalTimeEmbeddings(self.time_dim)
            self.time_embedding_t = SinusoidalTimeEmbeddings(self.time_dim)

        self.time_scheduler = TimeScheduler(self.hparams.get("time_scheduler", ""))

        self.norm_p = self.hparams.get("norm_p", 1.0)
        self.norm_eps = self.hparams.get("norm_eps", 0.01)
        self.data_proportion = self.hparams.get("data_proportion", 0.5)
        self.P_mean = self.hparams.get("P_mean", -0.4)
        self.P_std = self.hparams.get("P_std", 1.0)
        self.loss_u_weight = self.hparams.get("loss_u_weight", 1.0)
        self.loss_v_weight = self.hparams.get("loss_v_weight", 1.0)

        self.guide_threshold = self.hparams.get("guide_threshold", None)
        if self.guide_threshold is not None:
            self.cond_emb = MultiEmbedding(**self.hparams.conditions)
            cemb_dim = self.cond_emb.n_out
        else:
            self.cond_emb = None
            cemb_dim = 1

        self.pred_type = self.hparams.decoder.get('pred_type', False)
        self.type_encoding = self.hparams.get('type_encoding', None)
        if self.type_encoding == "table":
            self.type_encoding = TypeTableModule()

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            type_encoding=self.type_encoding,
            latent_dim=self.hparams.latent_dim + 2 * self.time_dim,
            cemb_dim=cemb_dim,
            _recursive_=False,
        )

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)

        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.ot = self.hparams.get("ot", False)
        self.permute_l = HungarianMatcher("norm")
        self.permute_f = HungarianMatcher("norm_mic")
        self.lattice_polar = self.hparams.get("lattice_polar", False)
        self.lattice_polar_sigma = self.hparams.get("lattice_polar_sigma", 1.0)
        self.latticedecompnn = LatticeDecompNN()
        self.from_cubic = self.hparams.get("from_cubic", False)
        self.lattice_teacher_forcing = self.hparams.get("lattice_teacher_forcing", -1)

        self.symmetrize_anchor = self.hparams.get("symmetrize_anchor", False)
        self.symmetrize_rotavg = self.hparams.get("symmetrize_rotavg", False)
        self.post_symmetrize = self.hparams.get("post_symmetrize", True)
        self.symm_rotavg = SymmetrizeRotavg()
        self.use_symmetrize_loss = self.hparams.get("use_symmetrize_loss", False)
        self.cost_sym_lattice = self.hparams.get("cost_sym_lattice", self.hparams.cost_lattice)
        self.cost_sym_coord = self.hparams.get("cost_sym_coord", self.hparams.cost_coord)

        if self.ot:
            hydra.utils.log.info("Using optimal transport")
        if self.lattice_polar:
            hydra.utils.log.info(f"Using lattice polar decomposition with sigma={self.lattice_polar_sigma}")
        hydra.utils.log.info(
            f"iMF: norm_p={self.norm_p}, norm_eps={self.norm_eps}, data_proportion={self.data_proportion}"
        )

    def embed_time(self, r, t):
        return torch.cat([self.time_embedding_r(r), self.time_embedding_t(t)], dim=-1)

    def sample_lattice_polar(self, batch_size):
        l0 = torch.randn([batch_size, 6], device=self.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        if self.from_cubic:
            l0[:, :5] = 0
        return l0

    def logit_normal_dist(self, batch_size):
        return torch.sigmoid(torch.randn(batch_size, device=self.device) * self.P_std + self.P_mean)

    def sample_tr(self, batch_size):
        t = self.logit_normal_dist(batch_size)
        r = self.logit_normal_dist(batch_size)
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        fm_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        fm_count = int(round(batch_size * self.data_proportion))
        if fm_count > 0:
            fm_mask[torch.randperm(batch_size, device=self.device)[:fm_count]] = True
        r = torch.where(fm_mask, t, r)
        return self.time_scheduler(t), self.time_scheduler(r), fm_mask

    def adaptive_weight(self, loss_per_graph):
        adaptive = (loss_per_graph + self.norm_eps).pow(self.norm_p)
        return loss_per_graph / adaptive.detach()

    def graph_mse(self, pred, target):
        return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=-1)

    def node_mse(self, pred, target, node2graph, batch_size):
        loss_per_node = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=-1)
        return scatter(loss_per_node, node2graph, dim=0, dim_size=batch_size, reduce='mean')

    def symmetrize_velocity(self, pred_l, pred_f, batch, tar_f=None):
        loss_sym_l = pred_l.new_tensor(0.0)
        loss_sym_f = pred_f.new_tensor(0.0)
        if not self.post_symmetrize:
            return pred_l, pred_f, tar_f, loss_sym_l, loss_sym_f

        if self.symmetrize_anchor:
            if not self.lattice_polar:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_l_sym = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
            if tar_f is not None:
                tar_f = torch.einsum('bij,bj->bi', batch.ops_inv, tar_f)
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_sym)
                loss_sym_f = F.mse_loss(pred_f, torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor))
            return pred_l_sym, pred_f_anchor, tar_f, loss_sym_l, loss_sym_f

        if self.symmetrize_rotavg:
            if not self.lattice_polar:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_l_sym = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            pred_f_sym = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=pred_f,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            )
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_sym)
                loss_sym_f = F.mse_loss(pred_f, pred_f_sym)
            return pred_l_sym, pred_f_sym, tar_f, loss_sym_l, loss_sym_f

        return pred_l, pred_f, tar_f, loss_sym_l, loss_sym_f

    def build_noisy_state(self, batch, t):
        batch_size = batch.num_graphs

        if self.lattice_polar:
            lattices_rep_T = batch.lattice_polar
            lattices_rep_0 = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                lattices_rep_T = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_T, batch.spacegroup)
                lattices_rep_0 = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_0, batch.spacegroup)
            lattices_mat_T = lattice_polar_build_torch(lattices_rep_T)
        else:
            lattices_rep_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            lattices_rep_0 = torch.randn([batch_size, 3, 3], device=self.device)
            lattices_mat_T = lattices_rep_T

        lattice_teacher_forcing = self.current_epoch < self.lattice_teacher_forcing
        if lattice_teacher_forcing:
            lattices_rep_0 = lattices_rep_T

        frac_coords = batch.frac_coords
        f0 = torch.rand_like(frac_coords)
        if self.symmetrize_anchor:
            f0_anchor = f0[batch.anchor_index]
            f0_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index], f0_anchor)
            f0 = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f0_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f0 = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f0,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        if self.pred_type:
            if self.type_encoding is None:
                gt_atom_types = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
                rd_atom_types = torch.randn_like(gt_atom_types)
            else:
                gt_atom_types = self.type_encoding(batch.atom_types)
                rd_atom_types = self.type_encoding.get_rd_encoded_types(batch.num_nodes, device=self.device)
        else:
            gt_atom_types = None
            rd_atom_types = None

        if self.ot:
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                raise ValueError("OT is forbidden in symmetrize.")
            _, lattices_rep_0 = self.permute_l(lattices_rep_T, lattices_rep_0)
            _, f0 = self.permute_f(frac_coords, f0)

        tar_l = lattices_rep_T - lattices_rep_0
        tar_f = (frac_coords - f0 - 0.5) % 1 - 0.5
        tar_t = gt_atom_types - rd_atom_types if self.pred_type else None

        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        input_lattice_rep = lattices_rep_0 + t[l_expand_dim] * tar_l
        input_frac_coords = f0 + t.repeat_interleave(batch.num_atoms)[:, None] * tar_f
        input_lattice_mat = lattice_polar_build_torch(input_lattice_rep) if self.lattice_polar else input_lattice_rep
        input_atom_types = (
            rd_atom_types + t.repeat_interleave(batch.num_atoms)[:, None] * tar_t if self.pred_type else batch.atom_types
        )

        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice_rep = lattices_rep_T
            input_lattice_mat = lattices_mat_T

        return {
            'lattice_teacher_forcing': lattice_teacher_forcing,
            'tar_l': tar_l,
            'tar_f': tar_f,
            'tar_t': tar_t,
            'input_lattice_rep': input_lattice_rep,
            'input_lattice_mat': input_lattice_mat,
            'input_frac_coords': input_frac_coords,
            'input_atom_types': input_atom_types,
        }

    def forward(self, batch, guide_threshold=None):
        batch_size = batch.num_graphs
        t, r, fm_mask = self.sample_tr(batch_size)

        guide_threshold = self.guide_threshold if guide_threshold is None else guide_threshold
        if guide_threshold is None:
            cemb = None
            guide_indicator = None
        else:
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = torch.rand(batch_size, device=self.device) - guide_threshold
            guide_indicator = guide_indicator.heaviside(torch.tensor(1.0, device=self.device))

        state = self.build_noisy_state(batch, t)
        input_lattice_rep = state['input_lattice_rep']
        input_lattice_mat = state['input_lattice_mat']
        input_frac_coords = state['input_frac_coords']
        input_atom_types = state['input_atom_types']
        tar_l = state['tar_l']
        tar_f = state['tar_f']
        tar_t = state['tar_t']

        u_pred, v_pred = self.decoder(
            t=self.embed_time(r, t),
            atom_types=input_atom_types,
            frac_coords=input_frac_coords,
            lattices_rep=input_lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=input_lattice_mat,
            cemb=cemb,
            guide_indicator=guide_indicator,
        )

        if self.pred_type:
            u_l_raw, u_f_raw, u_t_raw = u_pred
            v_l_raw, v_f_raw, v_t_raw = v_pred
        else:
            u_l_raw, u_f_raw = u_pred
            v_l_raw, v_f_raw = v_pred
            u_t_raw = v_t_raw = None

        u_l, u_f, tar_f_u, sym_u_l, sym_u_f = self.symmetrize_velocity(u_l_raw, u_f_raw, batch, tar_f=tar_f)
        v_l, v_f, tar_f_v, sym_v_l, sym_v_f = self.symmetrize_velocity(v_l_raw, v_f_raw, batch, tar_f=tar_f)
        tar_f = tar_f_v if tar_f_v is not None else tar_f_u
        loss_sym_l = sym_u_l + sym_v_l
        loss_sym_f = sym_u_f + sym_v_f
        u_t, v_t = u_t_raw, v_t_raw

        if self.pred_type:
            def u_only_fn(lattice_rep, frac_coords, atom_type_state, t_cur, r_cur):
                lattices_mat = lattice_polar_build_torch(lattice_rep) if self.lattice_polar else lattice_rep
                return self.decoder.forward_u(
                    t=self.embed_time(r_cur, t_cur),
                    atom_types=atom_type_state,
                    frac_coords=frac_coords,
                    lattices_rep=lattice_rep,
                    num_atoms=batch.num_atoms,
                    node2graph=batch.batch,
                    lattices_mat=lattices_mat,
                    cemb=cemb,
                    guide_indicator=guide_indicator,
                )

            primals = (input_lattice_rep, input_frac_coords, input_atom_types, t, r)
            tangents = (v_l.detach(), v_f.detach(), v_t.detach(), torch.ones_like(t), torch.zeros_like(r))
            u_out, du_dt = jvp(u_only_fn, primals, tangents)
            u_l_jvp, u_f_jvp, u_t_jvp = u_out
            du_dt_l, du_dt_f, du_dt_t = du_dt
        else:
            def u_only_fn(lattice_rep, frac_coords, t_cur, r_cur):
                lattices_mat = lattice_polar_build_torch(lattice_rep) if self.lattice_polar else lattice_rep
                return self.decoder.forward_u(
                    t=self.embed_time(r_cur, t_cur),
                    atom_types=input_atom_types,
                    frac_coords=frac_coords,
                    lattices_rep=lattice_rep,
                    num_atoms=batch.num_atoms,
                    node2graph=batch.batch,
                    lattices_mat=lattices_mat,
                    cemb=cemb,
                    guide_indicator=guide_indicator,
                )

            primals = (input_lattice_rep, input_frac_coords, t, r)
            tangents = (v_l.detach(), v_f.detach(), torch.ones_like(t), torch.zeros_like(r))
            u_out, du_dt = jvp(u_only_fn, primals, tangents)
            u_l_jvp, u_f_jvp = u_out
            du_dt_l, du_dt_f = du_dt
            u_t_jvp = du_dt_t = None

        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        dt_graph = (t - r)[l_expand_dim]
        dt_node = (t - r).repeat_interleave(batch.num_atoms)[:, None]

        # NOTE:
        # This implementation uses the forward parameterization
        #   z_t = z_0 + t * (z_1 - z_0)
        # where t=0 is noise and t=1 is data, and sampling integrates forward
        # with x <- x + dt * u.
        # Under this convention, the meanflow compound field is
        #   V = u - (t - r) * du/dt,
        # not the reverse-time + sign used in the original image-space iMF code.
        V_l = u_l_jvp - dt_graph * du_dt_l.detach()
        V_f = u_f_jvp - dt_node * du_dt_f.detach()
        V_t = u_t_jvp - dt_node * du_dt_t.detach() if self.pred_type else None

        loss_u_lattice_graph = self.graph_mse(V_l, tar_l)
        loss_u_coord_graph = self.node_mse(V_f, tar_f, batch.batch, batch_size)
        loss_v_lattice_graph = self.graph_mse(v_l, tar_l)
        loss_v_coord_graph = self.node_mse(v_f, tar_f, batch.batch, batch_size)

        if self.pred_type:
            loss_u_type_graph = self.node_mse(V_t, tar_t, batch.batch, batch_size)
            loss_v_type_graph = self.node_mse(v_t, tar_t, batch.batch, batch_size)
        else:
            loss_u_type_graph = torch.zeros(batch_size, device=self.device)
            loss_v_type_graph = torch.zeros(batch_size, device=self.device)

        cost_coord = self.hparams.cost_coord
        cost_lattice = 0.0 if state['lattice_teacher_forcing'] else self.hparams.cost_lattice
        cost_type = 0.0 if not self.pred_type else self.hparams.cost_type

        loss_u_graph = cost_lattice * loss_u_lattice_graph + cost_coord * loss_u_coord_graph + cost_type * loss_u_type_graph
        loss_v_graph = cost_lattice * loss_v_lattice_graph + cost_coord * loss_v_coord_graph + cost_type * loss_v_type_graph

        raw_loss_u = loss_u_graph.mean()
        raw_loss_v = loss_v_graph.mean()
        raw_loss = (
            self.loss_u_weight * raw_loss_u
            + self.loss_v_weight * raw_loss_v
            + self.cost_sym_lattice * loss_sym_l
            + self.cost_sym_coord * loss_sym_f
        )

        obj_loss_u = self.adaptive_weight(loss_u_graph).mean()
        obj_loss_v = self.adaptive_weight(loss_v_graph).mean()
        loss = (
            self.loss_u_weight * obj_loss_u
            + self.loss_v_weight * obj_loss_v
            + self.cost_sym_lattice * loss_sym_l
            + self.cost_sym_coord * loss_sym_f
        )

        return {
            'loss': loss,
            'loss_raw': raw_loss,
            'loss_u': raw_loss_u,
            'loss_v': raw_loss_v,
            'loss_u_obj': obj_loss_u,
            'loss_v_obj': obj_loss_v,
            'loss_u_lattice': loss_u_lattice_graph.mean(),
            'loss_u_coord': loss_u_coord_graph.mean(),
            'loss_u_type': loss_u_type_graph.mean(),
            'loss_v_lattice': loss_v_lattice_graph.mean(),
            'loss_v_coord': loss_v_coord_graph.mean(),
            'loss_v_type': loss_v_type_graph.mean(),
            'loss_sym_lattice': loss_sym_l,
            'loss_sym_coord': loss_sym_f,
            'fm_mask_ratio': fm_mask.float().mean(),
        }

    @torch.no_grad()
    def sample(self, batch, step_lr=None, N=None, **kwargs):
        if N is None:
            N = round(1 / step_lr) if step_lr is not None else 1

        batch_size = batch.num_graphs
        if self.lattice_polar:
            l_t = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                l_t = self.latticedecompnn.proj_k_to_spacegroup(l_t, batch.spacegroup)
            lattices_mat_t = lattice_polar_build_torch(l_t)
        else:
            l_t = torch.randn([batch_size, 3, 3], device=self.device)
            lattices_mat_t = l_t

        f_t = torch.rand([batch.num_nodes, 3], device=self.device)
        if self.symmetrize_anchor:
            f_t_anchor = f_t[batch.anchor_index]
            f_t_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_t_anchor)
            f_t = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_t_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f_t = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f_t,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        if self.pred_type:
            if self.type_encoding is None:
                atom_type_state = torch.randn((batch.num_nodes, MAX_ATOMIC_NUM), device=self.device)
                atom_types = torch.argmax(atom_type_state, dim=-1) + 1
            else:
                atom_type_state = self.type_encoding.get_rd_encoded_types(batch.num_nodes, device=self.device)
                atom_types = self.type_encoding.decode_types(atom_type_state)
        else:
            atom_type_state = None
            atom_types = batch.atom_types

        traj = {0: {'num_atoms': batch.num_atoms, 'atom_types': atom_types, 'frac_coords': f_t % 1.0, 'lattices': lattices_mat_t}}
        t_steps = torch.linspace(0.0, 1.0, N + 1, device=self.device)

        for step in tqdm(range(N)):
            r_i = t_steps[step].repeat(batch_size)
            t_i = t_steps[step + 1].repeat(batch_size)
            pred = self.decoder.forward_u(
                t=self.embed_time(r_i, t_i),
                atom_types=atom_type_state if self.pred_type else atom_types,
                frac_coords=f_t,
                lattices_rep=l_t,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattice_polar_build_torch(l_t) if self.lattice_polar else l_t,
                cemb=None,
                guide_indicator=None,
            )

            if self.pred_type:
                pred_l, pred_f, pred_t = pred
            else:
                pred_l, pred_f = pred

            if self.symmetrize_anchor:
                if self.lattice_polar:
                    pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
                pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
                pred_f_anchor = torch.scatter_reduce(
                    torch.zeros_like(pred_f_anchor),
                    0,
                    batch.anchor_index.unsqueeze(-1).expand_as(pred_f_anchor),
                    pred_f_anchor,
                    reduce='mean'
                )[batch.anchor_index]
                pred_f = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor)
            elif self.symmetrize_rotavg:
                if self.lattice_polar:
                    pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
                pred_f = self.symm_rotavg.symmetrize_rank1_scaled(
                    scaled_forces=pred_f,
                    num_atoms=batch.num_atoms,
                    general_ops=batch.general_ops,
                    symm_map=batch.symm_map,
                    num_general_ops=batch.num_general_ops,
                )

            dt = (t_i - r_i)[0]
            l_t = l_t + dt * pred_l
            f_t = (f_t + dt * pred_f) % 1.0
            if self.pred_type:
                atom_type_state = atom_type_state + dt * pred_t
                if self.type_encoding is None:
                    atom_types = torch.argmax(atom_type_state, dim=-1) + 1
                else:
                    atom_types = self.type_encoding.decode_types(atom_type_state)

            lattices_mat_t = lattice_polar_build_torch(l_t) if self.lattice_polar else l_t
            traj[step + 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types,
                'frac_coords': f_t,
                'lattices': lattices_mat_t,
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': torch.stack([traj[i]['atom_types'] for i in range(N + 1)]) if self.pred_type else batch.atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(N + 1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(N + 1)]),
        }
        return traj[N], traj_stack

    def training_step(self, batch, batch_idx: int, dataloader_idx=0):
        output_dict = self(batch)
        loss = output_dict['loss']
        self.log_dict(
            {
                'train_loss': output_dict['loss_raw'],
                'train_loss_obj': loss,
                'train_loss_u': output_dict['loss_u'],
                'train_loss_v': output_dict['loss_v'],
                'train_loss_u_obj': output_dict['loss_u_obj'],
                'train_loss_v_obj': output_dict['loss_v_obj'],
                'train_u_lattice_loss': output_dict['loss_u_lattice'],
                'train_u_coord_loss': output_dict['loss_u_coord'],
                'train_v_lattice_loss': output_dict['loss_v_lattice'],
                'train_v_coord_loss': output_dict['loss_v_coord'],
                'train_sym_lattice_loss': output_dict['loss_sym_lattice'],
                'train_sym_coord_loss': output_dict['loss_sym_coord'],
                'train_fm_mask_ratio': output_dict['fm_mask_ratio'],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        if loss.isnan():
            return None
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='test')
        self.log_dict(log_dict, batch_size=batch.num_graphs)
        return loss

    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']
        return {
            f'{prefix}_loss': output_dict['loss_raw'],
            f'{prefix}_loss_obj': loss,
            f'{prefix}_loss_u': output_dict['loss_u'],
            f'{prefix}_loss_v': output_dict['loss_v'],
            f'{prefix}_loss_u_obj': output_dict['loss_u_obj'],
            f'{prefix}_loss_v_obj': output_dict['loss_v_obj'],
            f'{prefix}_u_lattice_loss': output_dict['loss_u_lattice'],
            f'{prefix}_u_coord_loss': output_dict['loss_u_coord'],
            f'{prefix}_u_type_loss': output_dict['loss_u_type'],
            f'{prefix}_v_lattice_loss': output_dict['loss_v_lattice'],
            f'{prefix}_v_coord_loss': output_dict['loss_v_coord'],
            f'{prefix}_v_type_loss': output_dict['loss_v_type'],
            f'{prefix}_sym_lattice_loss': output_dict['loss_sym_lattice'],
            f'{prefix}_sym_coord_loss': output_dict['loss_sym_coord'],
            f'{prefix}_fm_mask_ratio': output_dict['fm_mask_ratio'],
        }, loss

    def on_train_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_validation_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_test_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")
