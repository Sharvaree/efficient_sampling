from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from functorch import jacrev, vjp

import torch.nn as nn
import torch.optim as optim
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torchvision
from torchdiffeq import odeint
import copy


def differences_vector(input_vector,T):
        # Compute differences between neighboring elements
        differences = input_vector[:-1] - input_vector[1:]  
        return differences*T

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5)


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
            - 0.5
        )
        * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class EnVariationalCNF(torch.nn.Module):
    """
    The E(n) CNF Module.

    Retains the diffusion noise scheduling parameters in order to have q(z | x, h)
    be defined similarly to diffusion models.

    The ELBO is E_q[ log p(x, h | z) + log p(z) - log q(z | x, h) ]
    where log p(z) is a CNF.
    """

    def __init__(
        self,
        dynamics: models.EGNN_dynamics_QM9,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 1000,
        noise_precision=1e-4,
        loss_type="l2",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
    ):
        super().__init__()

        assert loss_type in {"l2"}
        self.loss_type = loss_type

        self.include_charges = include_charges

        self.gamma = PredefinedNoiseSchedule(
            "polynomial_2", timesteps=timesteps, precision=noise_precision
        )

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def v_t(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(
            self.norm_values[0]
        )

        # Casting to float in case h still has long or int type.
        h_cat = (
            (h["categorical"].float() - self.norm_biases[1])
            / self.norm_values[1]
            * node_mask
        )
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {"categorical": h_cat, "integer": h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = (
            z[:, :, 0 : self.n_dims],
            z[:, :, self.n_dims : self.n_dims + self.num_classes],
        )
        h_int = z[
            :, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1
        ]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == "l2":
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        # Compute mu for p(zs | zt).
        mu_x = z0
        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise
        )

        x = xh[:, :, : self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(
            x, z0[:, :, self.n_dims : -1], h_int, node_mask
        )

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {"integer": h_int, "categorical": h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
        self, x, h, z_0, gamma_0, eps, net_out, node_mask, epsilon=1e-10
    ):
        # Discrete properties are predicted directly from z_0.
        z_h_cat = (
            z_0[:, :, self.n_dims : -1]
            if self.include_charges
            else z_0[:, :, self.n_dims :]
        )
        z_h_int = (
            z_0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_0.device)
        )
        z_0_x = z_0[:, :, : self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        alpha_0 = self.alpha(gamma_0, target_tensor=z_0_x)
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_x)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        x_mean = 1 / alpha_0 * z_0_x
        x_var = sigma_0 / alpha_0
        log_p_x_given_z_without_constants = sum_except_batch(
            -0.5 / x_var * (x - x_mean) ** 2
        )

        # Compute delta indicator masks.
        h_integer = torch.round(
            h["integer"] * self.norm_values[2] + self.norm_biases[2]
        ).long()
        onehot = h["categorical"] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon
        )
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        """Computes Conditional Flow Matching loss with CondOT path."""

        t = torch.rand(size=(x.size(0), 1), device=x.device).float()
        t = self.inflate_batch_array(t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        z1 = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        # Sample from q(z0 | x, h)
        zeros = torch.zeros(size=(xh.size(0), 1), device=xh.device)
        gamma_0 = self.gamma(zeros)
        alpha_0 = self.alpha(gamma_0, x)
        sigma_0 = self.sigma(gamma_0, x)

        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )
        z0 = alpha_0 * xh + sigma_0 * eps

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = z0 + t * (z1 - z0)

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # Neural net prediction.
        v_pred = self.v_t(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        denom = (self.n_dims + self.in_node_nf) * z_t.shape[1]
        error = sum_except_batch((v_pred - (z1 - z0)) ** 2) / denom
        loss_t = 0.5 * error
        loss = loss_t

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss

    @torch.no_grad()
    def compute_nll(self, x, h, node_mask, edge_mask, context):
        """Uses 1000 Euler steps to estimate NLL."""
        # Concatenate x, h[integer] and h[categorical].
        z = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        shape = z.shape
        flat_dim = shape[1] * shape[2]
        z = z.reshape(-1, flat_dim)

        nfe = [0]

        v = torch.randint(low=0, high=2, size=z.shape).to(z) * 2 - 1

        def v_t(t, z):
            t = t.reshape(-1).expand(z.shape[0])
            t = self.inflate_batch_array(t, z)
            z = z.reshape(*shape)
            v = self.v_t(z, t, node_mask, edge_mask, context)
            return v.reshape(-1, flat_dim)

        def odefunc(t, tensor):
            nfe[0] += 1
            x = tensor[..., : flat_dim]
            vecfield = lambda x: v_t(t, x)
            dx, div = output_and_div(vecfield, x, v=v)
            div = div.reshape(-1, 1)
            return torch.cat([dx, div], dim=-1)

        # Solve ODE on the product manifold of data manifold x euclidean.
        state0 = torch.cat([z, torch.zeros_like(z[..., :1])], dim=-1)

        state1 = odeint(
            odefunc,
            state0,
            t=torch.linspace(0, 1, 2).to(z),
            atol=1e-5,
            rtol=1e-5,
        )[-1]

        x1, logdetjac = state1[..., : flat_dim], state1[..., -1]

        x1 = x1.reshape(*shape)
        
        logpz1 = normal_logprob(x1, 0., 0.)
        if node_mask is not None:
            logpz1 = logpz1 * node_mask
        logpz1 = logpz1.reshape(x1.shape[0], -1).sum(1)

        logpz0 = logpz1 + logdetjac
        return -logpz0

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        if self.training:
            # Compute Conditional Flow Matching loss.
            loss = self.compute_loss(x, h, node_mask, edge_mask, context)
            return loss
        else:
            # Compute log p(z).
            return self.compute_nll(x, h, node_mask, edge_mask, context)

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()
    def sample(
        self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, ode_opts={},
    ):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        def v_t(t, x):
            t = t.reshape(-1).expand(x.shape[0])
            t = self.inflate_batch_array(t, x)
            return self.v_t(x, t, node_mask, edge_mask, context)
        ode_opts = {"method": "midpoint", "options": {"step_size":1/50}}
        z = odeint(
            v_t,
            z,
            t=torch.linspace(1, 0, 2).to(z),
            **ode_opts,
            atol=1e-5,
            rtol=1e-5,
        )[-1]

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context, fix_noise=fix_noise
        )

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        return x, h
    
    
    def xh_given_z0(self, z0, node_mask, smooth_h=False,tmp=1):
        x = z0[:, :, :self.n_dims]
        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)
        # create smooth version of h_cat
        if smooth_h:
            h_cat = F.softmax(h_cat*tmp, dim=2) + (F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) - F.softmax(h_cat*tmp, dim=2)).detach()
            h_cat = h_cat * node_mask
        else:
            h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h

    def set_mu_sig(self,old_T,midpoint=False):
        time_arr = torch.round(torch.linspace(0,old_T,self.T+1)).long()
        sig = torch.sqrt(torch.sigmoid(self.gamma.gamma))
        mu = torch.sqrt(torch.sigmoid(-self.gamma.gamma))
        # mu = torch.sqrt(1-sig.pow(2))
        mu_dt = differences_vector(mu, old_T)
        sig_dt = differences_vector(sig, old_T)
        return time_arr, mu, sig, mu_dt, sig_dt
    
    def set_data_for_classifier(self, x, h, node_mask, n_nodes, n_samples):
        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        # set data
        one_hot = h['categorical']
        one_hot = node_mask * one_hot.float()
        assert_correctly_masked(one_hot, node_mask)
        edges = get_adj_matrix(n_nodes, n_samples, x.device)
        atom_positions = x.view(n_samples * n_nodes, -1)
        atom_mask = node_mask.view(n_samples * n_nodes, -1)
        one_hot = one_hot.view(n_samples * n_nodes, -1)
        return one_hot, atom_positions, atom_mask, edges
    

    # @torch.no_grad()
    def optimized_sample(self, n_samples, n_nodes, node_mask,edge_mask, 
                         context,
                         target,
                         classifier,
                         optimization_steps=5,
                         iterations=5,
                         lr=1,
                         old_T = 1000,
                         new_T = 50,
                         fix_noise=False,
                         model_type='noise'):
        """
        Draw samples from the generative model.
        """

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            init_z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            init_z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        diffusion_utils.assert_mean_zero_with_mask(init_z[:, :, :self.n_dims], node_mask)

        init_z = nn.Parameter(init_z,requires_grad=True)
        node_mask = node_mask.detach()
        cur_number_nodes = node_mask.sum()

        optimizer = optim.LBFGS([init_z],lr=lr,max_iter=iterations, line_search_fn='strong_wolfe')

        def v_t(t, x):
            t = t.reshape(-1).expand(x.shape[0])
            t = self.inflate_batch_array(t, x)
            return self.v_t(x, t, node_mask, edge_mask, context)
        
        self.T = new_T
        step_size = 1/self.T
        ode_opts = {"method": "midpoint", "options": {"step_size":step_size}}
     

        for i in range(optimization_steps):
            x = 0
            h = 0
            loss = 0
            reg_loss = 0
            norm_loss = 0
            def closure():
                nonlocal loss
                nonlocal reg_loss
                nonlocal norm_loss
                nonlocal x
                nonlocal h
                optimizer.zero_grad()

                z = torch.cat([diffusion_utils.remove_mean_with_mask(node_mask*init_z[:, :, :self.n_dims],
                                                        node_mask), node_mask*init_z[:, :, self.n_dims:]],
                                                        dim=2)

                z_std = z[:,:cur_number_nodes.int(),:].std(1,keepdim=True) 
                z = z/z_std
                z = odeint(
                    v_t,
                    z,
                    t=torch.linspace(1, 0, 2).to(z),
                    **ode_opts,
                )[-1]
                
                # z = odeint(ode_func, z, t=torch.FloatTensor([0,1]).to(init_z.device), **ode_opts)[-1,...]
                
                x, h = self.xh_given_z0(z, node_mask, smooth_h=True, tmp=1)
                x = diffusion_utils.assert_and_center(x*node_mask,node_mask)
                diffusion_utils.assert_mean_zero_with_mask(x, node_mask)
                max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
                if max_cog > 5e-2:
                    print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                        f'the positions down.')
                    x = diffusion_utils.remove_mean_with_mask(x, node_mask)
                
                
                one_hot, atom_positions, atom_mask, edges = self.set_data_for_classifier(x, h, node_mask, n_nodes, n_samples)
                classifier.eval()
                pred = classifier(h0=one_hot, x=atom_positions, edges=edges, edge_attr=None,
                                node_mask=atom_mask, edge_mask=edge_mask,
                                n_nodes=n_nodes)
                reg_loss = F.mse_loss(pred, target.unsqueeze(-1)) 
                loss = reg_loss
                loss.backward()
                return loss
            optimizer.step(closure)
            print('optimization step {}, regression loss {}, norm loss {}'.format(i, reg_loss, norm_loss))  
        return x, h
    

    @torch.no_grad()
    def sample_chain(
        self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        def v_t(t, x):
            t = t.reshape(-1).expand(x.shape[0])
            t = self.inflate_batch_array(t, x)
            return self.v_t(x, t, node_mask, edge_mask, context)

        chain = odeint(
            v_t,
            z,
            t=torch.linspace(1, 0, keep_frames).to(z),
            atol=1e-5,
            rtol=1e-5,
        )
        z = chain[-1]

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {"log_SNR_max": log_SNR_max.item(), "log_SNR_min": log_SNR_min.item()}
        print(info)

        return info


def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)
    return lambda x: torch.trace(J(x))


def output_and_div(vecfield, x, v=None):
    dx, vjpfunc = vjp(vecfield, x)
    vJ = vjpfunc(v)[0]
    div = torch.sum(vJ * v.clone(), dim=-1)
    return dx, div


def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)
