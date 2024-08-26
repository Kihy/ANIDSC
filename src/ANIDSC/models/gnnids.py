from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..cdd_frameworks.drift_sense import DriftSense
from ..base_files.model import BaseOnlineODModel
import torch
import numpy as np

from ..base_files.pipeline import PipelineComponent
from torch_geometric.data import Data
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn.models import GAT, GAE, MLP, GCN
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_isolated_nodes
import pickle

from scipy import integrate

from ..base_files.save_mixin import TorchSaveMixin
from ..normalizer.t_digest import LivePercentile
from ..utils import uniqueXT

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


def find_edge_indices(edges, srcID, dstID):
    search_edges = torch.vstack([srcID, dstID])
    column_indices = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        index = torch.nonzero(mask, as_tuple=False)
        if index.size(0) > 0:
            # If the column exists, append its index

            column_indices.append(index.item())
        else:
            # If the column does not exist, append None
            column_indices.append(-1)
    return torch.tensor(column_indices)


def edge_exists(edges, search_edges):
    label = []
    for col in search_edges.t():
        # Check if column col of B exists in A
        mask = torch.all(edges == col.unsqueeze(1).to(edges.device), dim=0)
        # Find the index where column col of B exists in A
        label.append(mask.any())
    return torch.tensor(label)


def complete_directed_graph_edges(n):
    # Create an array of node indices
    nodes = torch.arange(n)

    # Generate all possible pairs of nodes
    pairs = torch.cartesian_prod(nodes, nodes)

    # Filter out self-loops
    edges = pairs[pairs[:, 0] != pairs[:, 1]]

    return edges.T


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


class BaseNodeEncoder(ABC, torch.nn.Module):
    def __init__(self, node_latent_dim: int, embedding_dist: str, device: str = "cuda"):
        torch.nn.Module.__init__(self)
        self.node_latent_dim = node_latent_dim
        self.embedding_dist = embedding_dist
        self.device = device
        self.linear = torch.nn.Linear(self.node_latent_dim, self.node_latent_dim).to(
            self.device
        )
        self.sw_loss = SWLoss(50, self.embedding_dist).to("cuda")

    def __str__(self):
        return f'{self.__class__.__name__}({self.embedding_dist})'

    @abstractmethod
    def setup(self):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings = self.node_embed(
            x=x, edge_index=edge_index, edge_attr=edge_attr
        )
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)


class GATNodeEncoder(BaseNodeEncoder):
    def setup(self, n_features):
        self.node_embed = GAT(
            in_channels=n_features,
            hidden_channels=n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).cuda()


class LinearNodeEncoder(BaseNodeEncoder):
    def setup(self, n_features):
        self.node_embed = MLP(
            in_channels=n_features,
            hidden_channels=n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).cuda()

    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings = self.node_embed(x=x)
        return self.linear(node_embeddings), self.sw_loss(node_embeddings)


class GCNNodeEncoder(BaseNodeEncoder):
    def setup(self, n_features):
        self.node_embed = GCN(
            in_channels=n_features,
            hidden_channels=n_features,
            out_channels=self.node_latent_dim,
            num_layers=2,
            norm=None,
        ).to(self.device)


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


class NodeEncoderWrapper(PipelineComponent, torch.nn.Module):
    def __init__(self, node_encoder, model, **kwargs):
        PipelineComponent.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        self.node_encoder = node_encoder
        self.node_encoder.parent = self
        self.model = model
        self.model.parent = self
        self.context = {}
        # self.loss_queue=self.model.loss_queue
        
    def __str__(self):
        return f"NodeEncoderWrapper({self.node_encoder}-{self.model})"

    def setup(self):
        context=self.get_context()
        self.node_encoder.setup(context["fe_features"])
        self.context["output_features"] = self.node_encoder.node_latent_dim
        self.model.setup()
        self.optimizer = torch.optim.Adam(params=self.parameters())

    # def predict_step(self, X, preprocess=False):
    #     if preprocess:
    #         X = self.preprocess(X)

    #     latent_nodes, dist_loss = self.node_encoder(*X)
    #     loss, threshold = self.model.predict_step(latent_nodes, preprocess)

    #     self.num_evaluated += 1
    #     return loss, threshold

    # def train_step(self, X, preprocess=False):
    #     if preprocess:
    #         X = self.preprocess(X)

    #     self.optimizer.zero_grad()
    #     latent_nodes, dist_loss = self.node_encoder(*X)
            
    #     if isinstance(self.model, torch.nn.Module):    
    #         loss = self.model.get_loss(latent_nodes, preprocess)
    #         if loss is not None:
    #             loss.mean().backward()
    #             self.optimizer.step()
                
    #             if self.profile:
    #                 self.prof.step()
                    
    #             loss=loss.detach().cpu().numpy()
    #         else:
    #             loss=0.
    #     else:
    #         loss=self.model.train_step(latent_nodes, preprocess)
            
    #         dist_loss.backward()
            
    #         self.optimizer.step()
            
    #     self.num_trained += 1
        
    #     # update scaler
    #     context=self.get_context()
    #     if "scaler" in context.keys():
    #         context["scaler"].update_current()

    #     return loss

    def process(self, X) -> Dict[str, Any]:
        latent_nodes, dist_loss = self.node_encoder(*X)
        
        if isinstance(self.model, DriftSense):
            
            result_dict = self.model.process(latent_nodes.detach())
        else:
            result_dict = self.model.process(latent_nodes)
        
        if result_dict.get("train", False):
            self.optimizer.zero_grad()
            dist_loss.mean().backward()
            self.optimizer.step()
        
        return result_dict


class HomoGraphRepresentation(PipelineComponent, torch.nn.Module, TorchSaveMixin):
    def __init__(
        self,
        device: str = "cuda",
        preprocessors: List[str] = [],
        load_name: str = "",
        **kwargs,
    ):
        """Homogeneous graph representation of network

        Args:
            device (str, optional): name of device. Defaults to "cuda".
            preprocessors (List[str], optional): list of preprocessors. Defaults to [].
            load_existing (bool, optional): whether to load existing graph representation. Defaults to False.
        """
        torch.nn.Module.__init__(self)
        PipelineComponent.__init__(self, **kwargs)
        self.device = device
        self.preprocessors = preprocessors
        self.load_name = load_name
        self.custom_params = ["G"]

    def preprocess(self, X):
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = getattr(self, p)(X)
        return X

    def to_device(self, X):
        return X.to(self.device)

    def to_float_tensor(self, X):
        return torch.from_numpy(X).float()

    def teardown(self):
        context = self.get_context()
        self.save(suffix=context.get("protocol", ""))

    def setup(self):
        super().setup()

        n_features = 15
        l_features = 15

        self.parent.context["fe_features"] = n_features
        self.parent.context["l_features"] = l_features
        self.parent.context["output_features"] = n_features

        if not self.loaded_from_file:
            # initialize graph
            self.G = Data()

            # initialize node features
            self.G.x = torch.empty((0, n_features)).to(self.device)
            self.G.idx = torch.empty(0).long().to(self.device)
            self.G.updated = torch.empty(0).to(self.device)

            # initialize edge features
            self.G.edge_index = torch.empty((2, 0)).to(self.device)
            self.G.edge_attr = torch.empty((0, l_features)).to(self.device)

        # self.mac_device_map={"d2:19:d4:e9:94:86":"Router",
        #                      "22:c9:ca:f6:da:60":"Smartphone 1",
        #                     "a2:bd:fa:b5:89:92":"Smart Clock 1",
        #                     "5e:ea:b2:63:fc:aa":"Google Nest Mini 1",
        #                     "7e:d1:9d:c4:d1:73":"Smart TV",
        #                     "52:a8:55:3e:34:46":"Smart Bulb 1",
        #                     "52:8e:44:e1:da:9a":"IP Camera 1",
        #                     "0a:08:59:8a:2f:1b":"Smart Plug 1",
        #                     "ea:d8:43:b8:6e:9a":"Raspberry Pi",
        #                     "ff:ff:ff:ff:ff:ff":"Broadcast",
        #                     "00:00:00:00:00:00":"Localhost",
        #                     "be:7b:f6:f2:1b:5f":"Attacker"}

    def update_nodes(self, srcID, src_feature, dstID, dst_feature):
        """updates source and destination nodes

        Args:
            srcID (int): id of source
            src_feature (array): src feature
            dstID (int): destionation id
            dst_feature (array): destination features
        """
        self.G.updated = torch.zeros_like(self.G.updated).to(self.device)

        unique_nodes = torch.unique(torch.cat([srcID, dstID]))

        # find set difference unique_nodes-self.G.node_idx
        uniques, counts = torch.cat((unique_nodes, self.G.idx, self.G.idx)).unique(
            return_counts=True
        )
        difference = uniques[counts == 1]

        expand_size = len(difference)

        if expand_size > 0:
            self.G.x = torch.vstack(
                [
                    self.G.x,
                    torch.zeros((expand_size, src_feature.shape[1])).to(self.device),
                ]
            )
            self.G.idx = torch.hstack([self.G.idx, difference.to(self.device)]).long()
            self.G.updated = torch.hstack(
                [self.G.updated, torch.ones(expand_size).to(self.device)]
            )

        # update
        for i in range(len(srcID)):
            self.G.x[self.G.idx == srcID[i]] = src_feature[i]
            self.G.x[self.G.idx == dstID[i]] = dst_feature[i]

        self.G.updated = torch.isin(self.G.idx, unique_nodes)

    def update_edges(self, srcID, dstID, edge_feature):
        """updates edges between source and destination

        Args:
            srcID (int): source ID
            dstID (int): destination ID
            edge_feature (array): edge feature
        """
        # convert ID to index
        src_idx = torch.nonzero(self.G.idx == srcID[:, None], as_tuple=False)[:, 1]
        dst_idx = torch.nonzero(self.G.idx == dstID[:, None], as_tuple=False)[:, 1]

        edge_indices = find_edge_indices(self.G.edge_index, src_idx, dst_idx)
        existing_edge_idx = edge_indices != -1

        # update found edges if there are any
        if existing_edge_idx.any():
            self.G.edge_attr[edge_indices[existing_edge_idx]] = edge_feature[
                existing_edge_idx
            ]

        num_new_edges = torch.count_nonzero(~existing_edge_idx)
        if num_new_edges > 0:
            new_edges = torch.vstack(
                [src_idx[~existing_edge_idx], dst_idx[~existing_edge_idx]]
            )
            self.G.edge_index = torch.hstack([self.G.edge_index, new_edges]).long()
            self.G.edge_attr = torch.vstack(
                [self.G.edge_attr, edge_feature[~existing_edge_idx]]
            )

    def get_graph_representation(self):
        G = to_networkx(
            self.G, node_attrs=["idx", "updated","x"], edge_attrs=['edge_attr'], to_undirected=False, to_multi=False
        )
        return G

    def split_data(self, x):
        """splits input data into different parts

        Args:
            x (_type_): input data

        Returns:
            _type_: graph IDs and features
        """
        srcIDs = x[:, 1].long()
        dstIDs = x[:, 2].long()

        context = self.get_context()
        src_features = x[:, 4:19].float()
        dst_features = x[:, 19:34].float()
        edge_features = x[:, 34 : 34 + context["l_features"]].float()

        return srcIDs, dstIDs, src_features, dst_features, edge_features

    def sample_edges(self, p):
        n = self.G.x.size(0)
        probabilities = torch.rand(n, n)

        # Get indices where probabilities are less than p and apply the mask
        indices = torch.argwhere(probabilities < p).T

        return indices, edge_exists(self.edge_idx, indices)

    def process(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """updates data with x and output graph representation after update

        Args:
            x (_type_): input data features

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: tuple of node features, edge indices, edge features
        """
        x = self.preprocess(x)

        diff = x[1:, 0] - x[:-1, 0]

        split_indices = torch.nonzero(diff, as_tuple=True)[0] + 1
        split_indices = split_indices.tolist()
        split_indices = [0] + split_indices + [x.size(0)]
        split_indices = np.array(split_indices[1:]) - np.array(split_indices[:-1])
        updated = False

        for data in torch.split(x, split_indices.tolist(), dim=0):
            # update chunk
            if data[0, 0] == 1:
                srcIDs, dstIDs, src_features, dst_features, edge_features = (
                    self.split_data(data)
                )

                # find index of last update
                _, indices = uniqueXT(
                    data[:, 1:3], return_index=True, occur_last=True, dim=0
                )
                self.subset = self.update_nodes(
                    srcIDs[indices],
                    src_features[indices],
                    dstIDs[indices],
                    dst_features[indices],
                )
                self.update_edges(
                    srcIDs[indices], dstIDs[indices], edge_features[indices]
                )
                updated = True

            # delete links
            else:
                srcIDs, dstIDs, _, _, _ = self.split_data(data)

                # convert to idx
                src_idx = torch.nonzero(self.G.idx == srcIDs[:, None], as_tuple=False)[
                    :, 1
                ]
                dst_idx = torch.nonzero(self.G.idx == dstIDs[:, None], as_tuple=False)[
                    :, 1
                ]

                edge_indices = find_edge_indices(self.G.edge_index, src_idx, dst_idx)

                edge_mask = torch.ones(self.G.edge_index.size(1), dtype=torch.bool)
                edge_mask[edge_indices] = False

                self.G.edge_index = self.G.edge_index[:, edge_mask]

                self.G.edge_attr = self.G.edge_attr[edge_mask]

                # remove isolated nodes
                edge_index, _, mask = remove_isolated_nodes(
                    self.G.edge_index, num_nodes=self.G.x.size(0)
                )
                # update edge index
                self.G.edge_index = edge_index
                self.G.x = self.G.x[mask]
                self.G.idx = self.G.idx[mask]

        self.parent.context["G"] = self.get_graph_representation()
        # if only deletion, no need to return anything
        if not updated:
            return None

        return self.G.x, self.G.edge_index, self.G.edge_attr
