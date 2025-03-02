from typing import Tuple
import torch
from .base_torch_model import BaseTorchModel


class AE(BaseTorchModel):
    def __init__(self, *args, **kwargs):
        """a base autoencoder

        Args:
            device (str, optional): device for this model. Defaults to "cuda".
            node_encoder (Dict[str,Any], optional): the node encoder to encode features. Defaults to None.
        """
        super().__init__(*args, **kwargs)

    def init_model(self):

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.context["output_features"], 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ).to(self.device)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, self.context["output_features"]),
        ).to(self.device)

        self.criterion = torch.nn.MSELoss(reduction="none").to(self.device)
        self.optimizer = torch.optim.Adam(params=self.parameters())

    def forward(self, X, inference=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """the forward pass of the model

        Args:
            X (_type_): input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of output and loss
        """

        encoded = self.encoder(X)
        decoded = self.decoder(encoded)

        return decoded, self.criterion(X, decoded).mean(dim=1)


class VAE(BaseTorchModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.context["output_features"], 8),
            torch.nn.ReLU(),
        ).to(self.device)

        self.fc_mu = torch.nn.Linear(8, 2).to(self.device)
        self.fc_logvar = torch.nn.Linear(8, 2).to(self.device)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, self.context["output_features"]),
        ).to(self.device)

        self.mse = torch.nn.MSELoss(reduction="none").to(self.device)
        self.kl_div = torch.nn.KLDivLoss(reduction="none").to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, x, inference=False):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        logvar = torch.clip(logvar, max=10.0)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        loss = self.mse(x, decoded).mean(dim=1) + kl_loss * 0.3

        return decoded, loss
