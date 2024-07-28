import torch.nn.functional as F
import torch
import numpy as np
from deepod.core.networks.base_networks import ConvNet
from ..base_files import BaseOnlineODModel

class GOAD(BaseOnlineODModel,torch.nn.Module):
    """
    Classification-Based Anomaly Detection for General Data (ICLR'20)
    """
    def __init__(self,  **kwargs):
        BaseOnlineODModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        
        self.n_trans = 32
        self.trans_dim = 16

        self.alpha = 0.1
        self.margin = 1.
        self.eps = 0

        self.kernel_size = 1
        self.hidden_dim = 10
        self.n_layers = 4

        
    def init_model(self, context):
        self.affine_weights = torch.from_numpy(np.random.randn(self.n_trans, context['output_features'], self.trans_dim)).float().to(self.device)
        self.net = GoadNet(
            self.trans_dim,
            kernel_size=self.kernel_size,
            n_hidden=self.hidden_dim,
            n_layers=self.n_layers,
            n_output=self.n_trans,
            activation='LeakyReLU', bias=False
        ).to(self.device)
        weights_init(self.net)
        
        self.criterion = GoadLoss(alpha=self.alpha, margin=self.margin, device=self.device)
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=1e-3,
                                            weight_decay=1e-5)      
        
        self.rep_means = torch.zeros((1, self.n_trans, self.hidden_dim)).to(self.device)
        
    def forward(self, X, inference=False):       
        # goad transform 
        X, batch_target=self.goad_transforms(X)
        
        batch_rep, batch_pred = self.net(X)
        batch_rep = batch_rep.permute(0, 2, 1)
        
        if inference:
            diffs = ((batch_rep.unsqueeze(2) - self.rep_means) ** 2).sum(-1)
            diffs_eps = self.eps * torch.ones_like(diffs)
            diffs = torch.max(diffs, diffs_eps)

            logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
            s = -torch.diagonal(logp_sz, 0, 1, 2)
            loss = s.sum(1)
        else:
            #online calculation of rep means
            self.rep_means=self.rep_means+(batch_rep.mean(0).unsqueeze(0)-self.rep_means)/(self.num_trained+1) 
            
            loss = self.criterion(batch_rep, batch_pred, batch_target)
            loss=loss.mean(1)
        return batch_rep, loss 

    def goad_transforms(self, X):
        x_trans = torch.einsum('bn,ant->bta', X, self.affine_weights)
        labels = torch.arange(self.n_trans, device=self.device).unsqueeze(0).expand((X.shape[0], self.n_trans))
        labels = labels.long()
        return x_trans, labels    

    


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