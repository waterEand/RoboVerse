"""
CBF: CVAE-Bootstrapped Flow Matching

Two-phase training:
  Phase 1 (steps 0 .. phase1_steps):
    Train CVAE only — learn q(z|a,obs), p(z|obs), p(a|z,obs).
    Loss = recon + beta-annealed KL with free bits.

  Phase 2 (steps > phase1_steps):
    Train flow net from prior to posterior in CVAE latent space.
    Joint CVAE regularisation keeps the latent geometry stable.
    Loss = flow + joint_cvae_weight*(recon + KL) [+ optional flow_recon].

Inference:
  z_0 ~ p(z|obs)  →  ODE(FlowNet)  →  z_1  →  Decoder  →  actions
  No train-test gap: both prior and decoder are obs-conditioned.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.utils.normalizer import LinearNormalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.policies.base_image_policy import BaseImagePolicy
from roboverse_learn.il.utils.models.flow_net import SimpleFlowNet
from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from roboverse_learn.il.utils.flow.flow_matchers import TorchFlowMatcher

from roboverse_learn.il.policies.cbf.cvae import (
    CVAEPosteriorEncoder,
    CVAEPriorNet,
    CVAEDecoder,
    reparameterize,
    kl_divergence,
)


class CBFImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        flow_net: dict,
        flow_matcher: TorchFlowMatcher,
        # Latent size (shared by CVAE and flow net)
        latent_dim: int = 512,
        # Phase control (step-based so no runner modification needed)
        phase1_steps: int = 10000,
        # CVAE architecture
        cvae_hidden_dim: int = 512,
        cvae_num_layers: int = 4,
        prior_hidden_dim: int = 256,
        prior_num_layers: int = 3,
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 4,
        # Loss weights
        kl_beta: float = 1e-3,
        beta_warmup_steps: int = 5000,   # linear warm-up for KL weight
        free_bits: float = 0.1,          # nats/dim — prevents posterior collapse
        recon_weight: float = 1.0,
        joint_cvae_weight: float = 0.1,  # CVAE loss weight in Phase 2
        flow_recon_weight: float = 0.0,  # expensive; 0 disables ODE during training
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps

        # Phase control
        self.phase1_steps = phase1_steps
        self._global_step = 0

        # Loss weights
        self.kl_beta = kl_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.free_bits = free_bits
        self.recon_weight = recon_weight
        self.joint_cvae_weight = joint_cvae_weight
        self.flow_recon_weight = flow_recon_weight

        self.flow_matcher = flow_matcher

        # ---- Obs encoder ----
        self.obs_encoder = obs_encoder
        self.obs_projector = nn.Linear(obs_feature_dim * n_obs_steps, latent_dim)

        # ---- CVAE ----
        self.posterior_encoder = CVAEPosteriorEncoder(
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            obs_dim=latent_dim,
            latent_dim=latent_dim,
            hidden_dim=cvae_hidden_dim,
            num_layers=cvae_num_layers,
        )
        self.prior_net = CVAEPriorNet(
            obs_dim=latent_dim,
            latent_dim=latent_dim,
            hidden_dim=prior_hidden_dim,
            num_layers=prior_num_layers,
        )
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            obs_dim=latent_dim,
            n_action_steps=n_action_steps,
            action_dim=action_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
        )

        # ---- Flow net (obs-conditioned MLP) ----
        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout,
            condition_dim=latent_dim,
        )

        self.normalizer = LinearNormalizer()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def training_phase(self) -> int:
        return 1 if self._global_step <= self.phase1_steps else 2

    def _annealed_beta(self) -> float:
        return self.kl_beta * min(1.0, self._global_step / max(1, self.beta_warmup_steps))

    def _encode_obs(self, nobs: dict, batch_size: int) -> torch.Tensor:
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        feats = self.obs_encoder(this_nobs).reshape(batch_size, -1)
        return self.obs_projector(feats)  # (B, latent_dim)

    def _future_actions(self, nactions: torch.Tensor) -> torch.Tensor:
        start = self.n_obs_steps - 1
        return nactions[:, start : start + self.n_action_steps, :]  # (B, T, Da)

    def _cvae_loss(
        self,
        future_actions: torch.Tensor,
        obs_latents: torch.Tensor,
        beta: float,
    ):
        post_mu, post_logvar = self.posterior_encoder(future_actions, obs_latents)
        z_post = reparameterize(post_mu, post_logvar)

        prior_mu, prior_logvar = self.prior_net(obs_latents)

        kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar, self.free_bits)
        a_recon = self.decoder(z_post, obs_latents)
        recon = F.l1_loss(a_recon, future_actions)

        loss = self.recon_weight * recon + beta * kl
        metrics = {
            'kl_loss': kl.item(),
            'recon_loss': recon.item(),
        }
        return loss, metrics, z_post, prior_mu, prior_logvar

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def compute_loss(self, batch) -> torch.Tensor:
        assert "valid_mask" not in batch
        self._global_step += 1

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        B = nactions.shape[0]

        obs_latents = self._encode_obs(nobs, B)
        future_actions = self._future_actions(nactions)
        beta = self._annealed_beta()

        cvae_loss, metrics, z_post, prior_mu, prior_logvar = self._cvae_loss(
            future_actions, obs_latents, beta
        )
        metrics.update({'beta': beta, 'training_phase': float(self.training_phase)})

        # ---- Phase 1: CVAE only ----
        if self.training_phase == 1:
            metrics['total_loss'] = cvae_loss.item()
            return cvae_loss

        # ---- Phase 2: flow + joint CVAE ----
        z_prior = reparameterize(prior_mu, prior_logvar)  # x_0 (source)
        # z_post is x_1 (target) — reuse from _cvae_loss, no extra forward pass

        flow_loss, _ = self.flow_matcher.compute_loss(
            self.flow_net,
            target=z_post,
            start=z_prior,
            global_cond=obs_latents,
        )
        metrics['flow_loss'] = flow_loss.item()

        loss = flow_loss + self.joint_cvae_weight * cvae_loss

        # Optional: supervise decoded output at flow endpoint (expensive)
        if self.flow_recon_weight > 0:
            z_end = self.flow_matcher.sample(
                self.flow_net,
                shape=(B, self.latent_dim),
                device=obs_latents.device,
                start=z_prior,
                num_steps=self.num_sampling_steps,
                global_cond=obs_latents,
            )
            flow_recon = F.l1_loss(self.decoder(z_end, obs_latents), future_actions)
            loss = loss + self.flow_recon_weight * flow_recon
            metrics['flow_recon_loss'] = flow_recon.item()

        metrics['total_loss'] = loss.item()
        return loss

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        obs_latents = self._encode_obs(nobs, B)

        # Sample from obs-conditioned prior (no train-test gap)
        z_0, _, _ = self.prior_net.sample(obs_latents)

        # Flow ODE: z_0 -> z_1
        z_1 = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=z_0,
            global_cond=obs_latents,
            return_traces=False,
        )

        # Decode latent to actions
        action_pred = self.decoder(z_1, obs_latents)
        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, :self.n_action_steps]

        return {"action": action, "action_pred": action_pred}

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
