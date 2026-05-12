"""
CVAE components for CBF (CVAE-Bootstrapped Flow Matching).

Three modules:
  CVAEPosteriorEncoder  q(z | a_future, obs)  — training only
  CVAEPriorNet          p(z | obs)             — training + inference
  CVAEDecoder           p(a | z, obs)          — training + inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from roboverse_learn.il.utils.models.layers import Mlp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)


def kl_divergence(
    post_mu: torch.Tensor,
    post_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """
    KL(q(z|a,obs) || p(z|obs)) for two diagonal Gaussians.

    KL = 0.5 * (lv2 - lv1 - 1 + exp(lv1-lv2) + (mu1-mu2)^2 * exp(-lv2))

    Returns a scalar (mean over batch, sum over latent dims).
    free_bits: minimum KL per dim (prevents posterior collapse).
    """
    kl_per_dim = 0.5 * (
        prior_logvar - post_logvar - 1.0
        + (post_logvar - prior_logvar).exp()
        + (post_mu - prior_mu).pow(2) * (-prior_logvar).exp()
    )  # (B, latent_dim)

    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)

    return kl_per_dim.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# shared MLP building block
# ---------------------------------------------------------------------------

def _make_mlp_stack(in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
    """Stack of (in→hidden) + (num_layers-1 residual Mlp blocks)."""
    def approx_gelu():
        return nn.GELU(approximate="tanh")

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        approx_gelu(),
    ]
    for _ in range(num_layers - 1):
        layers.append(
            Mlp(
                in_features=hidden_dim,
                hidden_features=hidden_dim * 2,
                out_features=hidden_dim,
                act_layer=approx_gelu,
                drop=0.0,
            )
        )
    return nn.Sequential(*layers)


def _xavier_init(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

class CVAEPosteriorEncoder(nn.Module):
    """
    q(z | a_future, obs_latents)  — used only during training.

    Encodes the future action sequence + obs features into
    the parameters of the posterior Gaussian.
    """

    def __init__(
        self,
        action_dim: int,
        n_action_steps: int,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim

        # Compress flat action sequence
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim * n_action_steps, hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        # Fuse with obs and run through MLP stack
        self.fusion = _make_mlp_stack(hidden_dim + obs_dim, hidden_dim, num_layers)

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        _xavier_init(self)
        # Zero-init output projections: start from N(0,1) posterior at epoch 0
        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

    def forward(
        self,
        a_future: torch.Tensor,   # (B, T, action_dim)
        obs_latents: torch.Tensor, # (B, obs_dim)
    ):
        B = a_future.shape[0]
        a_h = self.action_enc(a_future.reshape(B, -1))          # (B, hidden_dim)
        h = self.fusion(torch.cat([a_h, obs_latents], dim=-1))  # (B, hidden_dim)
        return self.mu_proj(h), self.logvar_proj(h)              # (B, latent_dim) each


class CVAEPriorNet(nn.Module):
    """
    p(z | obs_latents)  — learned obs-conditioned prior.

    Used in both training (as KL target) and inference (as flow source).
    Soft lower bound on logvar prevents prior from collapsing to a point mass.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        logvar_min: float = -4.0,  # std >= exp(-2) ≈ 0.14
    ):
        super().__init__()
        self.logvar_min = logvar_min

        self.net = _make_mlp_stack(obs_dim, hidden_dim, num_layers)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        _xavier_init(self)
        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

    def forward(self, obs_latents: torch.Tensor):
        """Returns mu, logvar — both (B, latent_dim)."""
        h = self.net(obs_latents)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(min=self.logvar_min)
        return mu, logvar

    def sample(self, obs_latents: torch.Tensor):
        """Sample z ~ p(z|obs). Returns (z, mu, logvar)."""
        mu, logvar = self(obs_latents)
        return reparameterize(mu, logvar), mu, logvar


class CVAEDecoder(nn.Module):
    """
    p(a | z, obs_latents)  — action decoder.

    Reconstructs the future action sequence from a latent z
    conditioned on obs features.
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        n_action_steps: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim

        self.net = _make_mlp_stack(latent_dim + obs_dim, hidden_dim, num_layers)
        self.out_proj = nn.Linear(hidden_dim, n_action_steps * action_dim)

        _xavier_init(self)
        nn.init.uniform_(self.out_proj.weight, -0.01, 0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, obs_latents: torch.Tensor):
        """
        z: (B, latent_dim)
        obs_latents: (B, obs_dim)
        Returns: (B, n_action_steps, action_dim)
        """
        h = self.net(torch.cat([z, obs_latents], dim=-1))
        return self.out_proj(h).view(-1, self.n_action_steps, self.action_dim)
