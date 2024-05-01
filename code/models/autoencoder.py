import torch 
from .base_model import *

class VAE(BaseOnlineODModel, torch.nn.Module, TorchSaveMixin):
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
        self.net.zero_grad()
        loss=self.get_loss(X)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def process(self,x):        
        score, threshold=self.predict_scores(x)
        self.score_hist.extend(score)
        self.train_step(x)
        
        return score, threshold
    
    def predict_scores(self, x):
        x, recon, mu, logvar=self.forward(x)
        loss = self.mse(x, recon).mean(dim=1)

        return loss.detach().cpu().numpy(), self.get_threshold()

class AE(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, n_features=16, **kwargs):
        BaseOnlineODModel.__init__(self,
            n_features=n_features, **kwargs
        )
        torch.nn.Module.__init__(self)
        
         
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2)
        ).to(self.device)
         
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, n_features),
        ).to(self.device)
        
        self.criterion=torch.nn.MSELoss(reduction='none').to(self.device)
    
    def to_device(self,X):
        return X.to(self.device)
    
    def forward(self, x):
        x = self.preprocess(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x, decoded
    
    def get_loss(self, x):
        x, recon=self.forward(x)
        loss = self.criterion(x, recon)
        loss=torch.mean(loss)
        return loss
    
    def train_step(self, X):
        self.net.zero_grad()
        loss=self.get_loss(X)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def process(self,x):        
        score, threshold=self.predict_scores(x)
        self.score_hist.extend(score)
        self.train_step(x)
        
        return score, threshold
    
    def predict_scores(self, x):
        x, recon=self.forward(x)
        loss = self.criterion(x, recon).mean(dim=1)

        return loss.detach().cpu().numpy(), self.get_threshold()