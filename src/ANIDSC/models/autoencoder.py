from typing import Any, Dict, Tuple
import torch
from numpy.typing import NDArray
from ..base_files.save_mixin import TorchSaveMixin 
from ..base_files import BaseOnlineODModel

from ..models import gnnids

class VAE(BaseOnlineODModel, torch.nn.Module):
    def __init__(self, n_features=16, **kwargs):
        BaseOnlineODModel.__init__(self,
            n_features=n_features, **kwargs
        )
        torch.nn.Module.__init__(self)
        
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 8),
            torch.nn.ReLU(),
        ).to(self.device)
        
        self.fc_mu=torch.nn.Linear(8, 2).to(self.device)
        self.fc_logvar=torch.nn.Linear(8, 2).to(self.device)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, n_features),
        ).to(self.device)
        
        self.mse=torch.nn.MSELoss(reduction='none').to(self.device)
        self.kl_div=torch.nn.KLDivLoss(reduction='none').to(self.device)
        self.optimizer= torch.optim.Adam(self.parameters())
    def to_device(self,X):
        return X.to(self.device)
    
    def reparameterize(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        
        eps=torch.rand_like(std)
        return eps*std+mu 
    
    def forward(self, x):
        x = self.preprocess(x)
        encoded = self.encoder(x)
        mu=self.fc_mu(encoded)
        logvar=self.fc_logvar(encoded)
        logvar=torch.clip(logvar, max=10.)
        z=self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return x, decoded, mu, logvar
    
    def get_loss(self, x):
        x, recon, mu, logvar=self.forward(x)
        kl_loss=torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = self.mse(x, recon) + kl_loss*0.3
        loss=torch.mean(loss)
        return loss
    
    def train_step(self, X):
        self.optimizer.zero_grad()
        loss=self.get_loss(X)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def process(self,x):        
        score, threshold=self.predict_scores(x)
        self.loss_queue.extend(score)
        self.update_scaler(x)
        self.train_step(x)
        
        return {
            "threshold": threshold,
            "score": score,
            "batch_num":self.num_batch
        }
    
    def predict_scores(self, x):
        x, recon, mu, logvar=self.forward(x)
        loss = self.mse(x, recon).mean(dim=1)

        return loss.detach().cpu().numpy(), self.get_threshold()


class AE(BaseOnlineODModel, TorchSaveMixin, torch.nn.Module):
    def __init__(self, device="cuda", **kwargs):
        """a base autoencoder

        Args:
            device (str, optional): device for this model. Defaults to "cuda".
            node_encoder (Dict[str,Any], optional): the node encoder to encode features. Defaults to None.
        """        
        BaseOnlineODModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        self.device=device
            
    def forward(self, X)->Tuple[torch.Tensor, torch.Tensor]:
        """ the forward pass of the model

        Args:
            X (_type_): input data
            include_dist (bool, optional): whether to include distance loss. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple of output and loss
        """        
        
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        
        return decoded, self.criterion(X, decoded)

    def process(self,X)->Dict[str, Any]:
        """process the input. First preprocess, then predict, then extend loss queue and finally train

        Args:
            X (_type_): _description_

        Returns:
            Dict[str, Any]: output dictionary
        """        
        X_scaled=self.preprocess(X)
        score, threshold=self.predict_step(X_scaled)
        self.loss_queue.extend(score)
        
        self.train_step(X_scaled)
        
        return {
            "threshold": threshold,
            "score": score,
            "batch_num":self.num_batch
        }
    
    def setup(self):
        context=self.get_context()
        
            
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(context["output_features"], 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2)
        ).to(self.device)
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, context["output_features"]),
        ).to(self.device)
        
        self.criterion=torch.nn.MSELoss(reduction='none').to(self.device)
        self.optimizer=torch.optim.Adam(params=self.parameters())
        
        super().setup()


    
    
    def predict_step(self, X, preprocess:bool=False)->Tuple[NDArray, float]:
        """predict the input

        Args:
            X (_type_): input data
            preprocess (bool, optional): whether to preprocess the data. Defaults to False.

        Returns:
            Tuple[NDArray, float]: _description_
        """        
        if preprocess:
            X=self.preprocess(X)
        _, loss = self.forward(X, include_dist=False)
        return loss.detach().cpu().numpy(), self.get_threshold()