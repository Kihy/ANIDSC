import torch.nn.functional as F
import torch
import numpy as np
from .base_model import BaseOnlineODModel, TorchSaveMixin
from deepod.core.networks.base_networks import ConvNet
from pathlib import Path


class GOAD(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    """
    Classification-Based Anomaly Detection for General Data (ICLR'20)
    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 n_trans=64, trans_dim=16,
                 alpha=0.1, margin=1., eps=0,
                 kernel_size=1, hidden_dim=8, n_layers=5,
                 act='LeakyReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 n_features=100, preprocessors=[],
                 verbose=2, random_state=42, **kwargs):
        self.device=device
        
        BaseOnlineODModel.__init__(self,
            epochs=epochs, batch_size=batch_size, lr=lr,n_features=n_features,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,preprocessors=preprocessors,
            verbose=verbose, random_state=random_state,**kwargs
        )
        torch.nn.Module.__init__(self)
        
        
        self.n_trans = n_trans
        self.trans_dim = trans_dim

        self.alpha = alpha
        self.margin = margin
        self.eps = eps

        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.act = act
        self.bias = bias
        self.device=device

        self.affine_weights = torch.from_numpy(np.random.randn(self.n_trans, self.n_features, self.trans_dim)).float().to(self.device)
        self.net = GoadNet(
            self.trans_dim,
            kernel_size=self.kernel_size,
            n_hidden=self.hidden_dim,
            n_layers=self.n_layers,
            n_output=self.n_trans,
            activation=self.act, bias=False
        ).to(self.device)
        weights_init(self.net)
        
        self.criterion = GoadLoss(alpha=self.alpha, margin=self.margin, device=self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=self.lr,
                                            weight_decay=1e-5)      
        
        
        self.preprocessors=list(preprocessors)
        
        self.preprocessors.append("goad_transforms")
        
        self.rep_means=None
        self.nb=0
        self.train()
        
    def to_device(self, X):
        return X.to(self.device)
    
    def forward(self, X):
        X, labels=self.preprocess(X)
        batch_rep, batch_pred = self.net(X)
        
        if self.training:
            if self.rep_means is None:
                self.rep_means = batch_rep.mean(0).t().unsqueeze(0)
            else:
                self.rep_means = self.rep_means+(batch_rep.mean(0).t().unsqueeze(0)-self.rep_means)/self.nb
            self.nb += 1
        batch_rep = batch_rep.permute(0, 2, 1)
        
        return batch_rep, batch_pred, labels
    
    
    def get_loss(self, X):
        batch_rep, batch_pred, labels=self.forward(X)
        loss = self.criterion(batch_rep, batch_pred, labels).mean()
        return loss
    
    def train_step(self, X):
        self.net.zero_grad()
        batch_rep, batch_pred, labels=self.forward(X)
        loss = self.criterion(batch_rep, batch_pred, labels).mean()

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_scores(self, X):
        batch_rep, batch_pred, labels=self.forward(X)
        diffs = ((batch_rep.unsqueeze(2) - self.rep_means.unsqueeze(0)) ** 2).sum(-1)
        
        diffs_eps = self.eps * torch.ones_like(diffs)
        diffs = torch.max(diffs, diffs_eps)
        logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
        s = -torch.diagonal(logp_sz, 0, 1, 2)
        s = s.sum(1)
        
        return s.detach().cpu().numpy(), self.get_threshold()


    def state_dict(self):
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]

    def process(self, X):
        scores, threshold=self.predict_scores(X)
        
        self.update_scaler(X)
        
        # update
        self.train_step(X)
        
        self.loss_queue.extend(scores)

        return {
            "threshold": threshold,
            "score": scores,
            "batch_num":self.num_batch
        }

    def goad_transforms(self, X):
        X=X.float().to(self.device)
        x_trans = torch.einsum('bn,ant->bta', X, self.affine_weights)
        labels = torch.arange(self.n_trans, device=self.device).unsqueeze(0).expand((X.shape[0], self.n_trans))
        labels = labels.long()
        
        return x_trans, labels    
    @property
    def threshold(self):
        return self._threshold.detach().cpu().numpy()
    
    @threshold.setter
    def threshold(self, t):
        self._threshold=torch.tensor(t).float()
    
    def on_train_begin(self):
        self.train()
    
    def on_train_end(self):
        self.eval()



class GoadNet(torch.nn.Module):
    def __init__(self, n_input,
                 kernel_size=1, n_hidden=8, n_layers=5, n_output=256,
                 activation='LeakyReLU', bias=False):
        super(GoadNet, self).__init__()

        self.enc = ConvNet(
            n_features=n_input,
            kernel_size=kernel_size,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            bias=bias
        )

        self.head = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv1d(n_hidden, n_output,
                            kernel_size=kernel_size, bias=True)
        )
        return

    def forward(self, x):
        rep = self.enc(x)
        pred = self.head(rep)
        return rep, pred


class GoadLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, margin=1., device='cuda'):
        super(GoadLoss, self).__init__()
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha
        self.margin = margin
        self.device = device
        return 

    def forward(self, rep, pred, labels):
        loss_ce = self.ce_criterion(pred, labels)

        # means = rep.mean(0).unsqueeze(0)
        # res = ((rep.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
        # pos = torch.diagonal(res, dim1=1, dim2=2)
        # offset = torch.diagflat(torch.ones(rep.size(1))).unsqueeze(0).to(self.device) * 1e6
        # neg = (res + offset).min(-1)[0]

        means = rep.mean(dim=0, keepdim=True)
        res = ((rep.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)

        # Compute pos
        pos = res.diagonal(dim1=1, dim2=2)

        # Compute neg
        offset = torch.eye(rep.size(1), device=rep.device).unsqueeze(0) * 1e6  # Broadcast offset along the batch dimension
        neg = (res + offset).min(dim=-1)[0]

        # Compute loss
        loss_tc = torch.clamp(pos + self.margin - neg, min=0).mean()

        loss = self.alpha * loss_tc + loss_ce
        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        torch.nn.init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        torch.nn.init.normal(m.weight, mean=0, std=0.01)