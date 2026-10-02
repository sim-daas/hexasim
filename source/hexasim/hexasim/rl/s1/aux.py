# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""S1 auxiliary learner: E_p, V, universal probe (O/B0 heads), VICReg, trained from the sequence buffer.

Owned by the env (the V_0 pattern), so latents exist from the first observation; ``S1PPO`` calls ``update()`` once
per PPO iteration and saves/loads ``state_dict()``. Nothing here receives a PPO or value gradient: the env writes
z_p and ê into the observation under ``no_grad`` (λ_rl_enc = 0, paradigm spec §8.2).

Per-step buffer rows (``SequenceBuffer``): the 111-D o_p frame; ``cur`` = current probe targets [contact (6), material
under the toe (6 × 4), local heights (6 × 9)] ⊕ estimator targets [v_true (3), F_n/F_t (6 × 2)] ⊕ slip speed (1);
touchdown events per leg: material (4) and local heights (9) at the ground-truth touchdown point, plus its world
position (→ p^TD in the row's body frame) and step (→ Δt^TD).

Losses (spec §3.3–3.4, v7 §4.3/§4.11): L_est = Huber(Δv̂, v_true − v_kin)·slip weight + Huber(f̂, F); L_probe = β-NLL
(β 0.5, BCE for contact), touchdown-material terms ×2, constant weight; L_vic = VICReg variance + covariance on z_p.
Target standardisation and the frame normaliser are frozen from the first full buffer, i.e. from the rollout of the
still-frozen π_S0 (actor frozen during critic burn-in).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.utils.configclass import configclass

from .kinematics import v_kin
from .losses import GaussianHead, beta_nll, vicreg_var_cov
from .seq_buffer import SequenceBuffer
from .tcn import ProprioTCN

STATS_DEFAULT = Path(__file__).with_name("probe_stats_pi_s0.pt")
STAT_KEYS = ("frame_mean", "frame_std", "cur_mean", "cur_std", "td_mean", "td_std")
FRAME = 111
# o_p frame slices (mdp.layout ACTOR blocks)
Q, QD, ERR, GYRO, IMU9 = slice(0, 18), slice(18, 36), slice(36, 54), slice(75, 78), slice(72, 81)
N_CONTACT, N_MAT, N_HGT = 6, 24, 54
CUR_PROBE = N_CONTACT + N_MAT + N_HGT  # 84
EST = 15
CUR_DIM = CUR_PROBE + EST + 1  # + slip speed
TD_EVENT = 4 + 9  # material, local heights
TD_PER_LEG = 3 + 1 + TD_EVENT  # p^TD, Δt^TD, material, heights → 17; × 6 = 102
LATENT = 64 + EST


@configclass
class AuxCfg:
    # Fixed target standardisation + frame normaliser from a π_S0 rollout over training terrain levels 0–14
    # (spec §3.3; scripts/tests/s1_probe_stats.py). One file for every S1 seed and arm.
    stats_path: str = str(STATS_DEFAULT)
    window: int = 50
    buffer_len: int = 128
    batch: int = 4096
    batches_per_iter: int = 8
    lr_heads: float = 1e-3
    lr_encoder: float = 3e-4  # ≈ 0.3 × heads (spec §8.4)
    lambda_est: float = 1.0
    lambda_probe: float = 1.0
    lambda_vic: float = 0.1
    vic_cov: float = 0.04
    td_material_weight: float = 2.0
    slip_weight: float = 20.0  # sample weight 1 + slip_weight · slip speed (m/s)
    force_scale: float = 8.0  # N, ≈ weight / 3
    # v_kin stance constants, sim-calibrated on π_S0 (scripts/tests/vkin_calibrate.py, D26): RMSE 0.063 m/s vs 0.087
    # at the first guess (0.02, 0.01, 0.005); stance/contact agreement 90 %. The grid optimum (0, 0.001, −0.02; RMSE
    # 0.058) weights only the lowest feet (agreement 58 %) and is rejected: Σw → 0 would pull v_kin to zero.
    vkin_tau_e: float = 0.005
    vkin_s: float = 0.003
    vkin_delta: float = -0.005


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class S1Aux(nn.Module):
    def __init__(self, cfg: AuxCfg, num_envs: int, q_default: torch.Tensor, dt: float, device):
        super().__init__()
        self.cfg, self.dt = cfg, dt
        self.register_buffer("q_default", q_default.clone())  # (18,) absolute rest angles
        self.encoder = ProprioTCN(FRAME, cfg.window)
        self.dv = mlp(64, 3)
        self.fhat = mlp(64, 12)
        self.cur_head = GaussianHead(64, CUR_PROBE)
        self.td_head = GaussianHead(64, 6 * TD_PER_LEG)
        # Fixed statistics (loaded from cfg.stats_path): frame normaliser, probe target mean/std.
        self.register_buffer("frame_mean", torch.zeros(FRAME))
        self.register_buffer("frame_std", torch.ones(FRAME))
        self.register_buffer("cur_mean", torch.zeros(CUR_PROBE))
        self.register_buffer("cur_std", torch.ones(CUR_PROBE))
        self.register_buffer("td_mean", torch.zeros(6 * TD_PER_LEG))
        self.register_buffer("td_std", torch.ones(6 * TD_PER_LEG))
        self.register_buffer("stats_frozen", torch.zeros((), dtype=torch.bool))
        if Path(cfg.stats_path).is_file():
            st = torch.load(cfg.stats_path, map_location="cpu", weights_only=False)
            for k in STAT_KEYS:
                getattr(self, k).copy_(st[k])
            self.stats_frozen.fill_(True)
        self.to(device)
        enc = list(self.encoder.parameters())
        heads = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        self.opt = torch.optim.Adam([{"params": enc, "lr": cfg.lr_encoder}, {"params": heads, "lr": cfg.lr_heads}])
        self.scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
        self.buffer = SequenceBuffer(num_envs, cfg.buffer_len, FRAME, CUR_DIM, TD_EVENT, cfg.window, device)
        self.updates = 0

    # ------------------------------------------------------------------ inference
    def vkin(self, frame: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        q = frame[..., Q] + self.q_default
        v, _ = v_kin(q, frame[..., QD], frame[..., ERR], frame[..., GYRO], c.vkin_tau_e, c.vkin_s, c.vkin_delta)
        return v

    def forward(self, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hist (B, W, 111) → z_p (B, 64), ê (B, 15) = [v̂_b, f̂], v_kin (B, 3)."""
        z = self.encoder((hist - self.frame_mean) / self.frame_std)
        vk = self.vkin(hist[:, -1])
        return z, torch.cat((vk + self.dv(z), self.fhat(z)), dim=-1), vk

    @torch.no_grad()
    def latents(self, hist: torch.Tensor) -> torch.Tensor:
        """Rollout inference (the actor's z_p, ê). fp16 autocast: 1.6× faster at 8,192 × 50 frames, max |Δz| 3.5e-3
        on unit-scale latents (bf16: 3.3e-2, rejected); training and the V tests stay fp32."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=hist.is_cuda):
            z, e, _ = self(hist)
        return torch.cat((z, e), dim=-1).float()

    # ------------------------------------------------------------------ training
    def _targets(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        td = torch.cat((b["td_pos_b"], b["td_dt"][..., None], b["td_event"]), dim=-1)  # (B, 6, 17)
        valid = b["td_valid"][..., None].expand_as(td).flatten(1)
        return b["cur"][:, :CUR_PROBE], td.flatten(1), valid

    @torch.no_grad()
    def compute_stats(self, n: int = 65536):
        """Set the statistics from the buffer's current contents (used by scripts/tests/s1_probe_stats.py)."""
        b = self.buffer.sample(n, self.dt)
        f = b["hist"][:, -1]
        self.frame_mean.copy_(f.mean(0))
        self.frame_std.copy_(f.std(0).clamp(min=1e-3))
        cur, td, valid = self._targets(b)
        self.cur_mean.copy_(cur.mean(0))
        # Target std floor 5e-3 (5 mm for positions/heights): touchdown local heights relative to the toe vary by only
        # 1–4 mm, and standardising at that scale would turn sub-mm noise into large targets.
        self.cur_std.copy_(cur.std(0).clamp(min=5e-3))
        self.cur_mean[:N_CONTACT], self.cur_std[:N_CONTACT] = 0.0, 1.0  # contact: BCE on raw 0/1
        n = valid.float().sum(0).clamp(min=1)
        m = (td * valid).sum(0) / n
        self.td_mean.copy_(m)
        self.td_std.copy_(torch.sqrt(((td - m) ** 2 * valid).sum(0) / n).clamp(min=5e-3))
        self.stats_frozen.fill_(True)

    def losses(self, b: dict) -> dict[str, torch.Tensor]:
        c = self.cfg
        z, e, vk = self(b["hist"])
        cur_all = b["cur"]
        v_true, force = cur_all[:, CUR_PROBE : CUR_PROBE + 3], cur_all[:, CUR_PROBE + 3 : CUR_PROBE + EST]
        slip = cur_all[:, -1]
        w = 1.0 + c.slip_weight * slip
        l_v = (F.huber_loss(e[:, :3] - vk, v_true - vk, reduction="none", delta=0.1).mean(-1) * w).sum() / w.sum()
        l_f = F.huber_loss(e[:, 3:] / c.force_scale, force / c.force_scale, delta=1.0)
        cur, td, valid = self._targets(b)
        mu, lv = self.cur_head(z)
        l_contact = F.binary_cross_entropy_with_logits(mu[:, :N_CONTACT], cur[:, :N_CONTACT])
        y = (cur[:, N_CONTACT:] - self.cur_mean[N_CONTACT:]) / self.cur_std[N_CONTACT:]
        l_cur = beta_nll(mu[:, N_CONTACT:], lv[:, N_CONTACT:], y).mean()
        mu_t, lv_t = self.td_head(z)
        yt = (td - self.td_mean) / self.td_std
        wt = torch.ones(TD_PER_LEG, device=z.device)
        wt[4:8] = c.td_material_weight  # material at touchdown ×2
        wt = wt.repeat(6) * valid
        l_td = (beta_nll(mu_t, lv_t, yt) * wt).sum() / wt.sum().clamp(min=1)
        var, cov = vicreg_var_cov(z)
        l_vic = var + c.vic_cov * cov
        total = c.lambda_est * (l_v + l_f) + c.lambda_probe * (l_contact + l_cur + l_td) + c.lambda_vic * l_vic
        return {
            "total": total,
            "est_v": l_v,
            "est_f": l_f,
            "probe_contact": l_contact,
            "probe_cur": l_cur,
            "probe_td": l_td,
            "vic_var": var,
            "vic_cov": cov,
        }

    def update(self) -> dict[str, float]:
        """One aux step block per PPO iteration; returns logs."""
        if not self.buffer.valid_mask().any() or self.buffer.step < self.buffer.L:
            return {}
        assert self.stats_frozen, f"no probe statistics at {self.cfg.stats_path}: run scripts/tests/s1_probe_stats.py"
        logs: dict[str, float] = {}
        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(self.cfg.batches_per_iter):
                b = self.buffer.sample(self.cfg.batch, self.dt)
                # fp16 mixed precision (loss scaling; β-NLL/BCE/Huber reductions run in fp32 under autocast).
                with torch.autocast("cuda", dtype=torch.float16, enabled=b["hist"].is_cuda):
                    out = self.losses(b)
                self.opt.zero_grad()
                self.scaler.scale(out["total"]).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
                for k, v in out.items():
                    logs[k] = logs.get(k, 0.0) + v.item() / self.cfg.batches_per_iter
        self.updates += 1
        return logs

    @torch.no_grad()
    def probe_skill(self, n: int = 16384) -> dict[str, float]:
        """Probe skill against the best constant predictor (the batch mean): R² = 1 − SSE/SST per group, averaged over
        the group's dims; contact: Brier skill 1 − Brier/Brier(base rate). ≤ 0 means the probe learned nothing a mean
        does not know. Material = (μ_s, μ_d, log10 k, log10(1 + c)); front = legs RF, LF, rear = the other four."""
        b = self.buffer.sample(n, self.dt)
        z, _, _ = self(b["hist"])
        cur, td, valid = self._targets(b)
        mu, _ = self.cur_head(z)
        p = torch.sigmoid(mu[:, :N_CONTACT])
        y = cur[:, :N_CONTACT]
        out = {"contact": (1 - (p - y).pow(2).mean() / (y.mean(0) - y).pow(2).mean().clamp(min=1e-9)).item()}

        def r2(pred, tgt, w=None):
            w = torch.ones_like(tgt) if w is None else w.float()
            nw = w.sum(0).clamp(min=1)
            m = (tgt * w).sum(0) / nw
            sst = ((tgt - m) ** 2 * w).sum(0)
            ok = sst > 1e-9 * nw  # skip constant dims (e.g. per-run damping)
            return (1 - ((pred - tgt) ** 2 * w).sum(0)[ok] / sst[ok]).mean().item() if ok.any() else float("nan")

        pc = mu[:, N_CONTACT:] * self.cur_std[N_CONTACT:] + self.cur_mean[N_CONTACT:]
        mat_p, mat_t = pc[:, :N_MAT].view(-1, 6, 4), cur[:, N_CONTACT : N_CONTACT + N_MAT].view(-1, 6, 4)
        front, rear = [0, 3], [1, 2, 4, 5]
        for j, k in enumerate(("mu_s", "mu_d", "logk", "damp")):
            out[f"cur_{k}_front"] = r2(mat_p[:, front, j], mat_t[:, front, j])
            out[f"cur_{k}_rear"] = r2(mat_p[:, rear, j], mat_t[:, rear, j])
        out["cur_height"] = r2(pc[:, N_MAT:], cur[:, N_CONTACT + N_MAT :])
        mu_t, _ = self.td_head(z)
        pt = (mu_t * self.td_std + self.td_mean).view(-1, 6, TD_PER_LEG)
        tt, vt = td.view(-1, 6, TD_PER_LEG), valid.view(-1, 6, TD_PER_LEG)
        for k, s in (("pos", slice(0, 3)), ("dt", slice(3, 4)), ("mu_s", slice(4, 5)), ("mu_d", slice(5, 6)),
                     ("logk", slice(6, 7)), ("height", slice(8, 17))):  # fmt: skip
            out[f"td_{k}"] = r2(pt[..., s].flatten(1), tt[..., s].flatten(1), vt[..., s].flatten(1))
        return out

    # ------------------------------------------------------------------ V tests (v7 §4.3)
    @torch.no_grad()
    def v_tests(self, n: int = 16384) -> dict[str, float]:
        """Slip-stratified v̂_b error and the IMU-zero ablation on buffer samples (current rollout data)."""
        b = self.buffer.sample(n, self.dt)
        v_true = b["cur"][:, CUR_PROBE : CUR_PROBE + 3]
        slip = b["cur"][:, -1]
        _, e, vk = self(b["hist"])
        err = (e[:, :3] - v_true).norm(dim=-1)
        rigid, slipping = slip < 0.01, slip > 0.05
        h0 = b["hist"].clone()
        h0[..., IMU9] = 0.0
        _, e0, _ = self(h0)
        rmse = lambda x: x.pow(2).mean().sqrt().item()  # noqa: E731
        base = rmse(err)
        return {
            "v_rmse": base,
            "v_rmse_rigid": rmse(err[rigid]) if rigid.any() else float("nan"),
            "v_rmse_slip": rmse(err[slipping]) if slipping.any() else float("nan"),
            "v_slip_share": slipping.float().mean().item(),
            "v_rmse_imu_zero": rmse((e0[:, :3] - v_true).norm(dim=-1)),
            "vkin_rmse": rmse((vk - v_true).norm(dim=-1)),
        }
