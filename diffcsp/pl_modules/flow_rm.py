import logging
import math
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


class CSPFlowRM(BaseModule):
    """RMFlow-style one-step MeanFlow training for crystal generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.time_dim == 0:
            self.time_dim = 1
            self.time_embedding_dt = DirectUnsqueezeTime()
            self.time_embedding_t = DirectUnsqueezeTime()
        else:
            self.time_dim = self.hparams.time_dim
            self.time_embedding_dt = SinusoidalTimeEmbeddings(self.time_dim)
            self.time_embedding_t = SinusoidalTimeEmbeddings(self.time_dim)

        self.time_scheduler = TimeScheduler(self.hparams.get("time_scheduler", ""))

        self.adaptive_gamma = self.hparams.get("adaptive_gamma", 0.5)
        self.adaptive_c = self.hparams.get("adaptive_c", 1e-3)
        self.time_sample_p = self.hparams.get("time_sample_p", 0.5)
        self.lambda_nll = self.hparams.get("lambda_nll", 5e-2)
        self.lambda_cond_reg = self.hparams.get("lambda_cond_reg", 0.0)

        self.sigma_min_lattice = self.hparams.get("sigma_min_lattice", 1e-3)
        self.sigma_min_coord = self.hparams.get("sigma_min_coord", 1e-3)
        self.transport_sigma_lattice = self.hparams.get("transport_sigma_lattice", 0.0)
        self.transport_sigma_coord = self.hparams.get("transport_sigma_coord", 0.0)
        if self.transport_sigma_lattice > self.sigma_min_lattice:
            raise ValueError("transport_sigma_lattice must be <= sigma_min_lattice")
        if self.transport_sigma_coord > self.sigma_min_coord:
            raise ValueError("transport_sigma_coord must be <= sigma_min_coord")
        self.refine_sigma_lattice = math.sqrt(max(self.sigma_min_lattice ** 2 - self.transport_sigma_lattice ** 2, 0.0))
        self.refine_sigma_coord = math.sqrt(max(self.sigma_min_coord ** 2 - self.transport_sigma_coord ** 2, 0.0))
        self.sample_refine = self.hparams.get("sample_refine", True)

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

        hydra.utils.log.info(
            "RMFlow: gamma=%s c=%s time_sample_p=%s lambda_nll=%s sigma_min(l,c)=(%s,%s)",
            self.adaptive_gamma,
            self.adaptive_c,
            self.time_sample_p,
            self.lambda_nll,
            self.sigma_min_lattice,
            self.sigma_min_coord,
        )

    def embed_time(self, r, t):
        dt = t - r
        return torch.cat([self.time_embedding_dt(dt), self.time_embedding_t(t)], dim=-1)

    def sample_lattice_polar(self, batch_size):
        l0 = torch.randn([batch_size, 6], device=self.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        if self.from_cubic:
            l0[:, :5] = 0
        return l0

    def sample_tr(self, batch_size):
        t = torch.sqrt(torch.rand(batch_size, device=self.device))
        r = torch.rand(batch_size, device=self.device) * t
        meanflow_mask = torch.rand(batch_size, device=self.device) < self.time_sample_p
        r = torch.where(meanflow_mask, r, t)
        return self.time_scheduler(t), self.time_scheduler(r), (~meanflow_mask)

    def adaptive_weight(self, loss_per_graph):
        p = 1.0 - self.adaptive_gamma
        weight = 1.0 / (loss_per_graph + self.adaptive_c).pow(p)
        return weight.detach() * loss_per_graph

    def graph_mse(self, pred, target):
        return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=-1)

    def node_mse(self, pred, target, node2graph, batch_size):
        loss_per_node = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=-1)
        return scatter(loss_per_node, node2graph, dim=0, dim_size=batch_size, reduce='mean')

    def node_wrapped_mse(self, pred, target, node2graph, batch_size):
        diff = (target - pred - 0.5) % 1.0 - 0.5
        loss_per_node = (diff ** 2).reshape(diff.shape[0], -1).mean(dim=-1)
        return scatter(loss_per_node, node2graph, dim=0, dim_size=batch_size, reduce='mean')

    def apply_coord_symmetry(self, frac_coords, batch):
        if self.symmetrize_anchor:
            f_anchor = frac_coords[batch.anchor_index]
            f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_anchor)
            return torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_anchor) + batch.ops[:, :3, 3]
        if self.symmetrize_rotavg:
            return self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=frac_coords,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]
        return frac_coords

    def apply_lattice_symmetry(self, lattice_rep, batch):
        if self.symmetrize_anchor or self.symmetrize_rotavg:
            if not self.lattice_polar:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            return self.latticedecompnn.proj_k_to_spacegroup(lattice_rep, batch.spacegroup)
        return lattice_rep

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
            clean_lattice_rep = batch.lattice_polar
            source_lattice_rep = self.sample_lattice_polar(batch_size)
            clean_lattice_rep = self.apply_lattice_symmetry(clean_lattice_rep, batch)
            source_lattice_rep = self.apply_lattice_symmetry(source_lattice_rep, batch)
        else:
            clean_lattice_rep = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            source_lattice_rep = torch.randn([batch_size, 3, 3], device=self.device)
        clean_lattice_mat = lattice_polar_build_torch(clean_lattice_rep) if self.lattice_polar else clean_lattice_rep

        lattice_teacher_forcing = self.current_epoch < self.lattice_teacher_forcing
        if lattice_teacher_forcing:
            source_lattice_rep = clean_lattice_rep

        clean_frac_coords = batch.frac_coords
        source_frac_coords = self.apply_coord_symmetry(torch.rand_like(clean_frac_coords), batch)

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
            _, source_lattice_rep = self.permute_l(clean_lattice_rep, source_lattice_rep)
            _, source_frac_coords = self.permute_f(clean_frac_coords, source_frac_coords)

        if self.transport_sigma_lattice > 0:
            transport_eps_l = torch.randn_like(clean_lattice_rep)
            target_lattice_rep = clean_lattice_rep + self.transport_sigma_lattice * transport_eps_l
            target_lattice_rep = self.apply_lattice_symmetry(target_lattice_rep, batch)
        else:
            target_lattice_rep = clean_lattice_rep

        if self.transport_sigma_coord > 0:
            transport_eps_f = torch.randn_like(clean_frac_coords)
            target_frac_coords = clean_frac_coords + self.transport_sigma_coord * transport_eps_f
            target_frac_coords = self.apply_coord_symmetry(target_frac_coords, batch)
        else:
            target_frac_coords = clean_frac_coords

        nll_target_lattice_rep = target_lattice_rep
        if self.refine_sigma_lattice > 0:
            nll_target_lattice_rep = nll_target_lattice_rep + self.refine_sigma_lattice * torch.randn_like(nll_target_lattice_rep)
            nll_target_lattice_rep = self.apply_lattice_symmetry(nll_target_lattice_rep, batch)

        nll_target_frac_coords = target_frac_coords
        if self.refine_sigma_coord > 0:
            nll_target_frac_coords = nll_target_frac_coords + self.refine_sigma_coord * torch.randn_like(nll_target_frac_coords)
            nll_target_frac_coords = self.apply_coord_symmetry(nll_target_frac_coords, batch)

        tar_l = target_lattice_rep - source_lattice_rep
        tar_f = (target_frac_coords - source_frac_coords - 0.5) % 1.0 - 0.5
        tar_t = gt_atom_types - rd_atom_types if self.pred_type else None

        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        input_lattice_rep = source_lattice_rep + t[l_expand_dim] * tar_l
        input_frac_coords = source_frac_coords + t.repeat_interleave(batch.num_atoms)[:, None] * tar_f
        input_lattice_mat = lattice_polar_build_torch(input_lattice_rep) if self.lattice_polar else input_lattice_rep
        input_atom_types = (
            rd_atom_types + t.repeat_interleave(batch.num_atoms)[:, None] * tar_t if self.pred_type else batch.atom_types
        )

        if self.keep_coords:
            input_frac_coords = clean_frac_coords
        if self.keep_lattice:
            input_lattice_rep = clean_lattice_rep
            input_lattice_mat = clean_lattice_mat

        return {
            'lattice_teacher_forcing': lattice_teacher_forcing,
            'source_lattice_rep': source_lattice_rep,
            'source_frac_coords': source_frac_coords,
            'source_atom_types': rd_atom_types,
            'tar_l': tar_l,
            'tar_f': tar_f,
            'tar_t': tar_t,
            'nll_target_l': nll_target_lattice_rep,
            'nll_target_f': nll_target_frac_coords,
            'nll_target_t': gt_atom_types,
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

        pred = self.decoder(
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
            pred_l_raw, pred_f_raw, pred_t_raw = pred
        else:
            pred_l_raw, pred_f_raw = pred
            pred_t_raw = None

        pred_l, pred_f, tar_f_loss, loss_sym_l, loss_sym_f = self.symmetrize_velocity(pred_l_raw, pred_f_raw, batch, tar_f=tar_f)
        pred_t = pred_t_raw
        tar_f_for_loss = tar_f_loss if tar_f_loss is not None else tar_f

        if self.pred_type:
            def u_fn(lattice_rep, frac_coords, atom_type_state, t_cur, r_cur):
                lattices_mat = lattice_polar_build_torch(lattice_rep) if self.lattice_polar else lattice_rep
                return self.decoder(
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
            tangents = (tar_l, tar_f, tar_t, torch.ones_like(t), torch.zeros_like(r))
            _, du_dt = jvp(u_fn, primals, tangents)
            du_dt_l, du_dt_f, du_dt_t = du_dt
        else:
            def u_fn(lattice_rep, frac_coords, t_cur, r_cur):
                lattices_mat = lattice_polar_build_torch(lattice_rep) if self.lattice_polar else lattice_rep
                return self.decoder(
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
            tangents = (tar_l, tar_f, torch.ones_like(t), torch.zeros_like(r))
            _, du_dt = jvp(u_fn, primals, tangents)
            du_dt_l, du_dt_f = du_dt
            du_dt_t = None

        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        dt_graph = (t - r)[l_expand_dim]
        dt_node = (t - r).repeat_interleave(batch.num_atoms)[:, None]
        target_l = tar_l - dt_graph * du_dt_l.detach()
        target_f = tar_f_for_loss - dt_node * du_dt_f.detach()
        target_t = tar_t - dt_node * du_dt_t.detach() if self.pred_type else None

        loss_cmfm_l_graph = self.graph_mse(pred_l, target_l)
        loss_cmfm_f_graph = self.node_mse(pred_f, target_f, batch.batch, batch_size)
        if self.pred_type:
            loss_cmfm_t_graph = self.node_mse(pred_t, target_t, batch.batch, batch_size)
        else:
            loss_cmfm_t_graph = torch.zeros(batch_size, device=self.device)

        cost_coord = self.hparams.cost_coord
        cost_lattice = 0.0 if state['lattice_teacher_forcing'] else self.hparams.cost_lattice
        cost_type = 0.0 if not self.pred_type else self.hparams.cost_type
        loss_cmfm_graph = (
            cost_lattice * loss_cmfm_l_graph
            + cost_coord * loss_cmfm_f_graph
            + cost_type * loss_cmfm_t_graph
        )
        loss_cmfm_raw = loss_cmfm_graph.mean()
        loss_cmfm_obj = self.adaptive_weight(loss_cmfm_graph).mean()

        r0 = torch.zeros(batch_size, device=self.device)
        t1 = torch.ones(batch_size, device=self.device)
        source_lattice_rep = state['source_lattice_rep']
        source_frac_coords = state['source_frac_coords']
        source_atom_types = state['source_atom_types'] if self.pred_type else batch.atom_types
        source_lattice_mat = lattice_polar_build_torch(source_lattice_rep) if self.lattice_polar else source_lattice_rep
        pred_01 = self.decoder(
            t=self.embed_time(r0, t1),
            atom_types=source_atom_types,
            frac_coords=source_frac_coords,
            lattices_rep=source_lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=source_lattice_mat,
            cemb=cemb,
            guide_indicator=guide_indicator,
        )
        if self.pred_type:
            pred_01_l, pred_01_f, pred_01_t = pred_01
        else:
            pred_01_l, pred_01_f = pred_01
            pred_01_t = None

        pred_01_l, pred_01_f, _, loss_sym_l_01, loss_sym_f_01 = self.symmetrize_velocity(pred_01_l, pred_01_f, batch)
        mu_l = source_lattice_rep + pred_01_l
        mu_f = source_frac_coords + pred_01_f
        loss_nll_l_graph = self.graph_mse(mu_l, state['nll_target_l'])
        loss_nll_f_graph = self.node_wrapped_mse(mu_f, state['nll_target_f'], batch.batch, batch_size)
        if self.pred_type:
            mu_t = source_atom_types + pred_01_t
            loss_nll_t_graph = self.node_mse(mu_t, state['nll_target_t'], batch.batch, batch_size)
        else:
            loss_nll_t_graph = torch.zeros(batch_size, device=self.device)
        loss_nll_graph = (
            cost_lattice * loss_nll_l_graph
            + cost_coord * loss_nll_f_graph
            + cost_type * loss_nll_t_graph
        )
        loss_nll = loss_nll_graph.mean()

        loss_cond_reg = cemb.pow(2).mean() if (cemb is not None and self.lambda_cond_reg > 0) else pred_l.new_tensor(0.0)
        loss_sym_l_total = loss_sym_l + loss_sym_l_01
        loss_sym_f_total = loss_sym_f + loss_sym_f_01
        loss = (
            loss_cmfm_obj
            + self.lambda_nll * loss_nll
            + self.lambda_cond_reg * loss_cond_reg
            + self.cost_sym_lattice * loss_sym_l_total
            + self.cost_sym_coord * loss_sym_f_total
        )

        return {
            'loss': loss,
            'loss_cmfm': loss_cmfm_raw,
            'loss_cmfm_obj': loss_cmfm_obj,
            'loss_nll': loss_nll,
            'loss_cond_reg': loss_cond_reg,
            'loss_cmfm_lattice': loss_cmfm_l_graph.mean(),
            'loss_cmfm_coord': loss_cmfm_f_graph.mean(),
            'loss_cmfm_type': loss_cmfm_t_graph.mean(),
            'loss_nll_lattice': loss_nll_l_graph.mean(),
            'loss_nll_coord': loss_nll_f_graph.mean(),
            'loss_nll_type': loss_nll_t_graph.mean(),
            'loss_sym_lattice': loss_sym_l_total,
            'loss_sym_coord': loss_sym_f_total,
            'fm_mask_ratio': fm_mask.float().mean(),
        }

    def _decoder_sample(self, batch, lattice_rep, frac_coords, atom_types, r, t, cemb=None, guide_indicator=None):
        lattices_mat = lattice_polar_build_torch(lattice_rep) if self.lattice_polar else lattice_rep
        return self.decoder(
            t=self.embed_time(r, t),
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices_rep=lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=lattices_mat,
            cemb=cemb,
            guide_indicator=guide_indicator,
        )

    @torch.no_grad()
    def sample(self, batch, step_lr=None, N=None, guide_factor=None, **kwargs):
        if N is None:
            N = round(1 / step_lr) if step_lr is not None else 1

        batch_size = batch.num_graphs
        if self.lattice_polar:
            l_t = self.sample_lattice_polar(batch_size)
            l_t = self.apply_lattice_symmetry(l_t, batch)
            lattices_mat_t = lattice_polar_build_torch(l_t)
        else:
            l_t = torch.randn([batch_size, 3, 3], device=self.device)
            lattices_mat_t = l_t

        f_t = self.apply_coord_symmetry(torch.rand([batch.num_nodes, 3], device=self.device), batch)

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

        if self.guide_threshold is None and guide_factor is not None:
            raise ValueError("Model is not trained with guidance but trying to sample with guidance.")
        if self.guide_threshold is not None and guide_factor is not None:
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = torch.ones(batch_size, device=self.device)
        else:
            cemb = None
            guide_indicator = None

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types,
                'frac_coords': f_t % 1.0,
                'lattices': lattices_mat_t,
            }
        }
        t_steps = torch.linspace(0.0, 1.0, N + 1, device=self.device)

        for step in tqdm(range(N)):
            r_i = t_steps[step].repeat(batch_size)
            t_i = t_steps[step + 1].repeat(batch_size)

            pred = self._decoder_sample(
                batch,
                l_t,
                f_t,
                atom_type_state if self.pred_type else atom_types,
                r_i,
                t_i,
                cemb=None,
                guide_indicator=None,
            )
            if self.pred_type:
                pred_l, pred_f, pred_t = pred
            else:
                pred_l, pred_f = pred
                pred_t = None

            pred_l, pred_f, _, _, _ = self.symmetrize_velocity(pred_l, pred_f, batch)

            if guide_factor is not None:
                pred_guide = self._decoder_sample(
                    batch,
                    l_t,
                    f_t,
                    atom_type_state if self.pred_type else atom_types,
                    r_i,
                    t_i,
                    cemb=cemb,
                    guide_indicator=guide_indicator,
                )
                if self.pred_type:
                    pred_l_guide, pred_f_guide, pred_t_guide = pred_guide
                else:
                    pred_l_guide, pred_f_guide = pred_guide
                    pred_t_guide = None
                pred_l_guide, pred_f_guide, _, _, _ = self.symmetrize_velocity(pred_l_guide, pred_f_guide, batch)
                pred_l = guide_factor * pred_l_guide + (1 - guide_factor) * pred_l
                pred_f = guide_factor * pred_f_guide + (1 - guide_factor) * pred_f
                if self.pred_type:
                    pred_t = guide_factor * pred_t_guide + (1 - guide_factor) * pred_t

            dt = (t_i - r_i)[0]
            l_t = l_t + dt * pred_l
            f_t = f_t + dt * pred_f
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
                'frac_coords': f_t % 1.0,
                'lattices': lattices_mat_t,
            }

        if self.sample_refine:
            if self.refine_sigma_lattice > 0:
                l_t = l_t + self.refine_sigma_lattice * torch.randn_like(l_t)
                l_t = self.apply_lattice_symmetry(l_t, batch)
            if self.refine_sigma_coord > 0:
                f_t = f_t + self.refine_sigma_coord * torch.randn_like(f_t)
                f_t = self.apply_coord_symmetry(f_t, batch)
            lattices_mat_t = lattice_polar_build_torch(l_t) if self.lattice_polar else l_t
            traj[N] = {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types,
                'frac_coords': f_t % 1.0,
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
                'train_loss': loss,
                'train_cmfm_loss': output_dict['loss_cmfm'],
                'train_cmfm_loss_obj': output_dict['loss_cmfm_obj'],
                'train_nll_loss': output_dict['loss_nll'],
                'train_cond_reg_loss': output_dict['loss_cond_reg'],
                'train_cmfm_lattice_loss': output_dict['loss_cmfm_lattice'],
                'train_cmfm_coord_loss': output_dict['loss_cmfm_coord'],
                'train_nll_lattice_loss': output_dict['loss_nll_lattice'],
                'train_nll_coord_loss': output_dict['loss_nll_coord'],
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
            f'{prefix}_loss': loss,
            f'{prefix}_cmfm_loss': output_dict['loss_cmfm'],
            f'{prefix}_cmfm_loss_obj': output_dict['loss_cmfm_obj'],
            f'{prefix}_nll_loss': output_dict['loss_nll'],
            f'{prefix}_cond_reg_loss': output_dict['loss_cond_reg'],
            f'{prefix}_cmfm_lattice_loss': output_dict['loss_cmfm_lattice'],
            f'{prefix}_cmfm_coord_loss': output_dict['loss_cmfm_coord'],
            f'{prefix}_cmfm_type_loss': output_dict['loss_cmfm_type'],
            f'{prefix}_nll_lattice_loss': output_dict['loss_nll_lattice'],
            f'{prefix}_nll_coord_loss': output_dict['loss_nll_coord'],
            f'{prefix}_nll_type_loss': output_dict['loss_nll_type'],
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