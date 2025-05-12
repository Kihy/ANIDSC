from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..cdd_framework.drift_sense import DriftSense
from .model import BaseOnlineODModel
import torch
import numpy as np

from ..component.pipeline_component import PipelineComponent
from torch_geometric.data import Data
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.models import GAT, GAE, MLP, GCN
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_isolated_nodes
import pickle

from scipy import integrate

from ..save_mixin.torch import TorchSaveMixin
from ..scaler.t_digest import LivePercentile
from ..utils.helper import uniqueXT

torch.set_printoptions(precision=4)


class SWLoss(torch.nn.Module):
    def __init__(self, slices: int, shape: str):
        """sliced wasserstein loss for latent space embedding

        Args:
            slices (int): number of slices
            shape (str): name of distribution
        """
        super().__init__()
        self.slices = slices
        self.shape = shape

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """calculates sw loss based on input

        Args:
            encoded (torch.Tensor): encoded loss

        Returns:
            torch.Tensor: loss value
        """
        # Define a PyTorch Variable for theta_ls
        theta = generate_theta(self.slices, encoded.size(1)).to("cuda")

        # Define a PyTorch Variable for samples of z
        z = generate_z(encoded.size(0), encoded.size(1), self.shape, center=0).to(
            "cuda"
        )

        # Let projae be the projection of the encoded samples
        projae = torch.tensordot(encoded, theta.T, dims=1)

        # Let projz be the projection of the q_Z samples
        projz = torch.tensordot(z, theta.T, dims=1)

        # Calculate the Sliced Wasserstein distance by sorting
        # the projections and calculating the L2 distance between
        sw_loss = ((torch.sort(projae.T)[0] - torch.sort(projz.T)[0]) ** 2).mean()
        return sw_loss


def generate_theta(n, dim):
    """generates n slices to be used to slice latent space, with dim dimensions. the slices are unit vectors"""
    rand = np.random.normal(size=[n, dim])
    theta = np.linalg.norm(rand, axis=1, keepdims=True)
    return torch.tensor(rand / theta).float()


def generate_z(n, dim, shape, center=0):
    """generates n samples with shape in dim dimensions, represents the prior distribution"""
    if shape == "uniform":
        z = 2 * np.random.uniform(size=[n, dim]) - 1
    elif shape == "circular":
        u = np.random.normal(size=[n, dim])
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        normalised_u = u / u_norm
        r = np.random.uniform(size=[n]) ** (1.0 / dim)
        z = np.expand_dims(r, axis=1) * normalised_u
        z += center
    elif shape == "gaussian":
        z = np.random.normal(size=[n, dim])
    elif shape == "log-normal":
        z = np.exp(np.random.normal(size=[n, dim]))
    return torch.tensor(z).float()


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, edges1, edges2, num_nodes):
        # Convert edges to adjacency matrices
        adj1 = self.edges_to_adjacency(edges1, num_nodes)
        adj2 = self.edges_to_adjacency(edges2, num_nodes)

        # Compute intersection and union
        intersection = torch.minimum(adj1, adj2)
        union = torch.maximum(adj1, adj2)

        return intersection.sum() / union.sum()

    def edges_to_adjacency(self, edges, num_nodes):
        adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
        adjacency[edges[0], edges[1]] = 1
        return adjacency


def get_protocol_map(fe_name, dataset_name):
    with open(f"../datasets/{dataset_name}/{fe_name}/state.pkl", "rb") as pf:
        state = pickle.load(pf)
    return state["protocol_map"]



class Classifier(torch.nn.Module):
    def __init__(self, channel_list):
        super().__init__()
        if channel_list is None:
            self.nn = torch.nn.Identity()
        else:
            self.nn = MLP(channel_list=channel_list, norm=None)  #
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        # Convert node embeddings to edge-level representations:
        x = self.nn(x)
        adj_matrix = x @ x.T

        # adj_matrix-=adj_matrix.min(dim=0, keepdims=True).values
        # Apply dot-product to get a prediction per supervision edge:
        return self.act(adj_matrix)


class VariationalGATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super().__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.gcn_shared(x, edge_index, edge_attr))
        mu = self.gcn_mu(x, edge_index, edge_attr)
        logvar = self.gcn_logvar(x, edge_index, edge_attr)
        return mu, logvar




class Diffusion:
    def __init__(
        self,
        noise_steps=20,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=256,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        return x


class ReluBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.dense = torch.nn.Linear(time_dim, out_channels)

    def forward(self, x, t=None):
        x = self.linear(x)
        x = self.relu(x)

        if t is not None:
            x = x + self.dense(t)

        return x


class NoisePredictor(torch.nn.Module):
    def __init__(self, in_channels, time_dim=16):
        super().__init__()
        self.enc1 = ReluBlock(in_channels, 16)
        self.enc2 = ReluBlock(16, 8)
        self.enc3 = ReluBlock(8, 4)

        self.dec2 = ReluBlock(4, 8)
        self.dec3 = ReluBlock(8, 16)
        self.dec4 = ReluBlock(16, in_channels)

        self.device = "cuda"

        self.out = torch.nn.Linear(in_channels, in_channels)

        self.pos_encoding = PositionalEncoding(20, time_dim)
        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, in_channels),
        )

    def forward(self, x, t):
        t = self.pos_encoding(t)
        x = x + self.emb_layer(t)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x4 = self.dec2(x3) + x2
        x5 = self.dec3(x4) + x1
        x6 = self.dec4(x5)

        return self.out(x6)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_time_steps, embedding_size, n=10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(
            max_time_steps, embedding_size, requires_grad=False
        ).to("cuda")
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]


class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale, requires_grad=False
        ).to("cuda")

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNet(torch.nn.Module):
    def __init__(self, marginal_prob_std, in_channels, time_dim=16):
        super().__init__()
        self.enc1 = ReluBlock(in_channels, 16, time_dim)
        self.enc2 = ReluBlock(16, 8, time_dim)
        self.enc3 = ReluBlock(8, 4, time_dim)

        self.dec2 = ReluBlock(4, 8, time_dim)
        self.dec3 = ReluBlock(8, 16, time_dim)
        self.dec4 = ReluBlock(16, in_channels, time_dim)

        self.device = "cuda"
        # The swish activation function
        self.act = torch.nn.SiLU()
        self.marginal_prob_std = marginal_prob_std
        self.out = torch.nn.Linear(in_channels, in_channels)

        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim),
            torch.nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        x1 = self.enc1(x, embed)
        x2 = self.enc2(x1, embed)
        x3 = self.enc3(x2, embed)

        x4 = self.dec2(x3, embed) + x2
        x5 = self.dec3(x4, embed) + x1
        x6 = self.dec4(x5, embed)

        return self.out(x6) / self.marginal_prob_std(t)[:, None]


def ode_likelihood(
    x,
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    device="cuda",
    eps=1e-5,
):
    """Compute the likelihood with probability flow ODE.

    Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=(1))

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(
            time_steps, device=device, dtype=torch.float32
        ).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(
                shape
            )
            time_steps = torch.tensor(
                time_steps, device=device, dtype=torch.float32
            ).reshape((sample.shape[0],))
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
            return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones((shape[0],)) * t
        sample = x[: -shape[0]]
        logp = x[-shape[0] :]
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate(
        [x.cpu().detach().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0
    )
    # Black-box ODE solver
    res = integrate.solve_ivp(
        ode_func, (eps, 1.0), init, rtol=1e-5, atol=1e-5, method="RK45"
    )
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[: -shape[0]].reshape(shape)
    sigma_max = marginal_prob_std(1.0)
    prior_logp = prior_likelihood(z, sigma_max)
    return z, prior_logp


def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
    standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2.0 * torch.log(2 * np.pi * sigma**2) - torch.sum(z**2, dim=(1)) / (
        2 * sigma**2
    )


def marginal_prob_std(t):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    sigma = torch.tensor(25)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / torch.log(sigma))


def diffusion_coeff(
    t,
):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The vector of diffusion coefficients.
    """
    sigma = torch.tensor(25)
    return sigma**t


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a
        time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, random_t)
    loss = torch.sum((score * std[:, None] + z) ** 2, dim=(1))
    return loss



