import torch.nn.functional as F
import torch
import numpy as np
from .base_model import *
from deepod.core.networks.base_networks import MLPnet
from pathlib import Path


class ICL(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    """
    Anomaly Detection for Tabular Data with Internal Contrastive Learning
     (ICLR'22)
    """
    
    def __init__(self, epochs=100, batch_size=64, lr=1e-3, n_ensemble='auto',
                 rep_dim=128, hidden_dims='100,50', act='LeakyReLU', bias=False,
                 kernel_size='auto', temperature=0.01, max_negatives=1000,n_features=100,
                 epoch_steps=-1, prt_steps=10, device='cuda', preprocessors=[],
                 verbose=2, random_state=42, **kwargs):
        self.device=device
        BaseOnlineODModel.__init__(self,
            epochs=epochs, batch_size=batch_size,n_features=n_features,
            lr=lr, n_ensemble=n_ensemble,preprocessors=preprocessors,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, **kwargs
        )
        torch.nn.Module.__init__(self)

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.kernel_size = kernel_size
        self.tau = temperature
        self.max_negatives = max_negatives

        

        if self.kernel_size == 'auto':
            if self.n_features <= 40:
                self.kernel_size = 2
            elif 40 < self.n_features <= 160:
                self.kernel_size = 10

            elif 160 < self.n_features <= 240:
                self.kernel_size = self.n_features - 150
            elif 240 < self.n_features <= 480:
                self.kernel_size = self.n_features - 200
            else:
                self.kernel_size = self.n_features - 400

        if self.n_features < 3:
            raise ValueError('ICL model cannot handle the data that have less than three features.')

        self.net = ICLNet(
            n_features=self.n_features,
            kernel_size=self.kernel_size,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation=self.act,
            bias=self.bias
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=self.lr,
                                            weight_decay=1e-5)
    
    def to_device(self, X):
        return X.to(self.device)
    
    def forward(self, X):
        X = self.preprocess(X)
        positives, query = self.net(X)
        logit = self.cal_logit(query, positives)
        logit = logit.permute(0, 2, 1)
        
        correct_class = torch.zeros((logit.shape[0], logit.shape[2]),
                                    dtype=torch.long).to(self.device)
        return logit, correct_class

    def get_loss(self, X):
        logit, correct_class=self.forward(X)
        loss = self.criterion(logit, correct_class)
        loss=torch.mean(loss)
        return loss

    def train_step(self, X):
        loss=self.get_loss(X)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()

    def process(self,X):
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


    def state_dict(self):
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):
        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
        

    def predict_scores(self, X):
        logit, correct_class=self.forward(X)
        loss = self.criterion(logit, correct_class)

        return loss.mean(dim=1).detach().cpu().numpy(), self.get_threshold()

    def cal_logit(self, query, pos):
        n_pos = query.shape[1]
        batch_size = query.shape[0]
        
       
        # get negatives
        negative_index = torch.randperm(n_pos)[:min(self.max_negatives, n_pos)]

        negative = pos.permute(0, 2, 1)[:, :, negative_index]

        pos_multiplication = (query * pos).sum(dim=2).unsqueeze(2)

        neg_multiplication = torch.matmul(query, negative)  # [batch_size, n_neg, n_neg]

        # Removal of the diagonals
        identity_matrix = torch.eye(n_pos).unsqueeze(0).to(self.device)
        identity_matrix = identity_matrix.repeat(batch_size, 1, 1)
        identity_matrix = identity_matrix[:, :, negative_index]

        neg_multiplication.masked_fill_(identity_matrix==1, -float('inf'))

        logit = torch.cat((pos_multiplication, neg_multiplication), dim=2)
        logit = torch.div(logit, self.tau)
        return logit


class ICLNet(torch.nn.Module):
    def __init__(self, n_features, kernel_size,
                 hidden_dims='100,50', rep_dim=64,
                 activation='ReLU', bias=False):
        super(ICLNet, self).__init__()
        self.n_features = n_features
        self.kernel_size = kernel_size

        # get consecutive subspace indices and the corresponding complement indices
        start_idx = np.arange(n_features)[: -kernel_size + 1]  # [0,1,2,...,dim-kernel_size+1]
        self.all_idx = start_idx[:, None] + np.arange(kernel_size)
        self.all_idx_complement = np.array([np.setdiff1d(np.arange(n_features), row)
                                            for row in self.all_idx])

        if type(hidden_dims)==str:
            hidden_dims = hidden_dims.split(',')
            hidden_dims = [int(a) for a in hidden_dims]
        n_layers = len(hidden_dims) # hidden layers
        f_act = ['Tanh']
        for _ in range(n_layers):
            f_act.append(activation)

        self.enc_f_net = MLPnet(
            n_features=n_features-kernel_size,
            n_hidden=hidden_dims,
            n_output=rep_dim,
            mid_channels=len(self.all_idx),
            batch_norm=True,
            activation=f_act,
            bias=bias,
        )

        hidden_dims2 = [int(0.5*h) for h in hidden_dims]
        g_act = []
        for _ in range(n_layers+1):
            g_act.append(activation)

        self.enc_g_net = MLPnet(
            n_features=kernel_size,
            n_hidden=hidden_dims2,
            n_output=rep_dim,
            mid_channels=len(self.all_idx),
            batch_norm=True,
            activation=g_act,
            bias=bias,
        )

        return

    def forward(self, x):
        x1, x2 = self.positive_matrix_builder(data=x)
        x1 = self.enc_g_net(x1)
        x2 = self.enc_f_net(x2)
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        return x1, x2

    def positive_matrix_builder(self, data):
        """
        Generate matrix of sub-vectors and matrix of complement vectors (positive pairs)

        Parameters
        ----------
        data: torch.Tensor shape (n_samples, n_features), required
            The input data.

        Returns
        -------
        matrix: torch.Tensor of shape [n_samples, number of sub-vectors, kernel_size]
            Derived sub-vectors.

        complement_matrix: torch.Tensor of shape [n_samples, number of sub-vectors, n_features-kernel_size]
            Complement vector of derived sub-vectors.

        """
        dim = self.n_features

        data = torch.unsqueeze(data, 1)  # [size, 1, dim]
        data = data.repeat(1, dim, 1)  # [size, dim, dim]

        matrix = data[:, np.arange(self.all_idx.shape[0])[:, None], self.all_idx]
        complement_matrix = data[:, np.arange(self.all_idx.shape[0])[:, None], self.all_idx_complement]

        return matrix, complement_matrix