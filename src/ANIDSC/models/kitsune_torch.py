from doctest import UnexpectedException
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

from ANIDSC.base_files.model import BaseOnlineODModel


class FeatureMapper:
    """Computes a hierarchical cluster on the given dataset columns.

    Parameters
    ----------
    dim : int
        Number of dataset columns (or features).
    max_features_per_cluster : int
        Maximum numbers of features in a cluster.
    """

    def __init__(
        self, dim: int, max_features_per_cluster: int, grace_period: int, device='cuda') -> None:
        # Algorithm hyperparameters
        self.dim = dim
        self.max_features_per_cluster = max_features_per_cluster
        self.grace_period = grace_period
        # Variable to lazily evaluate the clusters_ property
        self._clusters = []
        self.device=device

        # Check if fit method has already been called
        self._fitted = False

        self._init_stats()

    def _init_stats(self) -> None:
        # linear num of features
        self.running_sum = np.zeros(self.dim)

        # linear sum of feature residuals
        self.running_residuals = np.zeros(self.dim)

        # squared sum of feature residuals
        self.running_squared_residuals = np.zeros(self.dim)

        # partial correlation matrix
        self.running_corr_mat = np.zeros(shape=(self.dim, self.dim))
        
        self.seen_samples = 0
        self.seen_batches=0
        self._fitted = False
        self._clusters = []

    def parital_fit(self, x) -> None:
        """Updates the internal statistics based on data batch.

        Parameters
        ----------
        x 
            Dataset batch.

        Raises
        ------
        ValueError
            In case the number of columns is not the same as the one specified
            in the init. Otherwise this is triggered if the given x has an
            invalid number of dimensions.
        """
        if x.ndim not in {1, 2}:
            raise ValueError(f"Invalid number of input axes: {x.ndim}")

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Invalid number of input dims: got {x.shape[-1]} "
                f"expected {self.dim}"
            )

        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        self._update_stats(x)
        self._update_clusters()
        
        self._fitted = self.seen_batches > self.grace_period

    def transform(self, x):
        """Splits the given data into subgroups.

        Each group is a subset of columns of the given data based on the
        clustering strategy learned on the `fit` method.

        Parameters
        ----------
        x 
            Data to transform

        Returns
        -------
        List
            Groups of columns.

        Raises
        ------
        ValueError
            If the x dimensions is invalid.
        """
        if x.ndim not in {1, 2}:
            raise ValueError(f"Invalid number of input axes: {x.ndim}")

        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        return [x[:, indices] for indices in self.clusters_]

    @property
    def clusters_(self) -> List[List[int]]:
        """Property to get the learned clusters.

        Returns
        -------
        List[List[int]]
            Each position of the list contains a cluster with the corresponding
            column indices.

        Raises
        ------
        UnexpectedException
            If `fit` or `partial_fit` has not been called yet.
        """
        if not self._fitted:
            raise UnexpectedException("fit or partial_fit has not been called yet")
        return self._clusters

    @property
    def n_clusters_(self) -> int:
        return len(self.clusters_)

    def _update_stats(self, x) -> None:
        self.seen_samples += x.shape[0]
        self.seen_batches+=1
        self.running_sum += np.sum(x, axis=0)
        running_mean = self.running_sum / self.seen_samples
        residuals = x - running_mean
        self.running_residuals += np.sum(residuals, axis=0)
        self.running_squared_residuals += np.sum(residuals **2, axis=0)  
        self.running_corr_mat += np.sum(np.outer(r, r) for r in residuals)

    def _update_clusters(self) -> None:
        dist = self._compute_correlation_distance()
        indices = np.triu_indices(self.dim, k=1)
        link_mat = linkage(dist[indices[0], indices[1]])
        self._clusters = self._split_clusters(to_tree(link_mat))
        self._clusters = self._join_clusters(self._clusters)

    def _compute_correlation_distance(self):
        rs_sqrt = np.sqrt(self.running_squared_residuals)
        rs_sqrt_mat = np.outer(rs_sqrt, rs_sqrt)
        rs_sqrt_mat[rs_sqrt_mat==0] = 1e-16
        dist = 1.0 - self.running_corr_mat / rs_sqrt_mat
        dist[dist<0] = 0.0
        return dist

    def _join_clusters(self, clusters: List[List[int]]) -> List[List[int]]:
        # Join clusters until they at least have n elements
        # This not follows any heuristic, it just merges clusters to meet
        # the required cardinality (TODO improve this)
        while True:
            ci = [i for i, o in enumerate(clusters) if len(o) < 4]
            if not ci:
                return clusters
            elif len(ci) == 1:
                c = clusters.pop(ci[0])
                min(clusters, key=len).extend(c)
            else:
                merged_cluster = clusters.pop(ci[0]) + clusters.pop(ci[1] - 1)
                clusters.append(merged_cluster)

    def _split_clusters(self, cn: ClusterNode) -> List[List[int]]:
        # If clusters are to large split them recursively
        if cn.get_count() <= self.max_features_per_cluster:
            return [cn.pre_order()]

        left_clusters = self._split_clusters(cn.get_left())
        right_clusters = self._split_clusters(cn.get_right())
        return left_clusters + right_clusters


class TinyAutoEncoder(nn.Module):
    """AutoEncoder with one compresion layer.

    The inputs must be normalized between 0 and 1. For instance you can use
    a MinMaxScaler from scikit-learn.

    Parameters
    ----------
    in_features : int
        Number of input features. The reconstructed output will have the same
        output.
    compression_rate : float
        Percentage to compress the hidden auto encoder representation space.
        For example if `in_features=100` and `compression_rate=0.6` then the
        hidden dimension will have 40 dimensions. Defaults 0.6.
    dropout_rate : float
        Dropout rate applied to input features. Defaults 0.2.
    """

    def __init__(
        self,
        *,
        in_features: int,
        compression_rate: float = 0.6,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.compression_rate = compression_rate
        self.dropout_rate = dropout_rate

        hidden_units = int(in_features * (1.0 - self.compression_rate))

        # Initialize weight
        w_data = torch.empty(in_features, hidden_units)
        w_data.uniform_(-1.0 / in_features, to=1.0 / in_features)
        self.w = nn.Parameter(w_data)

        # Initialize biases
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_units))
        self.reconstruct_bias = nn.Parameter(torch.zeros(in_features))

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        hidden = (x @ self.w + self.hidden_bias).tanh()
        return (hidden @ self.w.t() + self.reconstruct_bias).sigmoid()


class RMSELoss(torch.nn.Module):
    """Convinen Root Mean Squared Error implementation.

    Just applies the `sqrt` method to the vanilla PyTorch MSELoss.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.criterion = nn.MSELoss(**kwargs)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.criterion(inputs, targets) + 1e-7)


class Kitsune(BaseOnlineODModel, nn.Module):

    def __init__(self, **kwargs) -> None:
        BaseOnlineODModel.__init__(self, **kwargs)
        nn.Module.__init__(self)

        self.compression_rate = 0.75
        self.dropout_rate = 0
        self.max_features_per_cluster = 10
        self.grace_period=40
        self.mse = RMSELoss(reduction="none")
        self.optimizer=None
        

    def init_model(self, context):
        self.fm = FeatureMapper(
            context["output_features"], self.max_features_per_cluster
        , self.grace_period)
        self.head = None

    def _build_tails(
        self, compression_rate: float, dropout_rate: float
    ) -> List[nn.Module]:
        return nn.ModuleList(
            [
                TinyAutoEncoder(
                    in_features=len(c),
                    compression_rate=compression_rate,
                    dropout_rate=dropout_rate,
                )
                for c in self.fm.clusters_
            ]
        ).to(self.device)

    def forward(self, x: torch.Tensor, inference: bool = True):
        # transform to numpy array for feature mapper
        if isinstance(x, torch.Tensor):
            x=self.to_numpy(x)
            
        if not self.fm._fitted:
            if not inference:
                self.fm.parital_fit(x)
                
                
            return None, None
        else:
            if self.head is None:
                self.tails = self._build_tails(self.compression_rate, self.dropout_rate)
                self.head = TinyAutoEncoder(
                    in_features=self.fm.n_clusters_,
                    compression_rate=self.compression_rate,
                    dropout_rate=self.dropout_rate,
                ).to(self.device)
                self.optimizer=torch.optim.Adam(params=self.parameters())

        # transform and convert to float tensor
        split_feat = self.fm.transform(x)
        
        tails_losses = torch.empty(x.shape[0], len(self.tails), device=self.device)
        for i, (feat, tail) in enumerate(zip(split_feat, self.tails)):
            feat = self.to_device(self.to_float_tensor(feat))
        
            reconstructed_feat = tail(feat)
            tails_losses[:, i] = self.mse(reconstructed_feat, feat).mean(-1)

        reconstructed_loss_dist = self.head(tails_losses)
        head_loss = self.mse(reconstructed_loss_dist, tails_losses)
        return reconstructed_loss_dist, head_loss.mean(dim=1)
