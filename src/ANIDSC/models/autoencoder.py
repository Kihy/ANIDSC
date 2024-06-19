import torch 
from ..base_files import BaseOnlineODModel
from ..models.base_model import TorchSaveMixin
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
    def __init__(self, device="cuda", node_encoder=None, **kwargs):
        BaseOnlineODModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        self.device=device
        self.node_encoder=node_encoder 
            
    def forward(self, X, include_dist=False):
        if self.node_encoder is not None:
            X, dist_loss=self.node_encoder(*X)
        else:
            dist_loss=0
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        if include_dist:
            loss = self.criterion(X, decoded).mean() + dist_loss
        else:
            loss = self.criterion(X, decoded).mean(dim=1)
        return decoded, loss

    def process(self,X):
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
        if self.node_encoder is not None:
            self.node_encoder=getattr(gnnids, self.node_encoder["encoder_name"])(self.node_encoder["node_latent_dim"],
                 context["n_features"],
                 self.node_encoder["embedding_dist"],
                 self.device)
            
            self.parent.context['n_features']=self.node_encoder.node_latent_dim
            context['n_features']=self.node_encoder.node_latent_dim
            
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(context["n_features"], 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2)
        ).to(self.device)
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, context["n_features"]),
        ).to(self.device)
        
        self.criterion=torch.nn.MSELoss(reduction='none').to(self.device)
        self.optimizer=torch.optim.Adam(params=self.parameters())
        
        super().setup()


    
    
    def predict_step(self, X):
        _, loss = self.forward(X, include_dist=False)
        return loss.detach().cpu().numpy(), self.get_threshold()