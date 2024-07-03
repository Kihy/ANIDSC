import torch.nn.functional as F
import torch
import numpy as np
from deepod.core.networks.base_networks import MLPnet

from ANIDSC.base_files.model import BaseOnlineODModel


class ICL(BaseOnlineODModel,torch.nn.Module):
    """
    Anomaly Detection for Tabular Data with Internal Contrastive Learning
     (ICLR'22)
    """
    
    def __init__(self, **kwargs):
        
        BaseOnlineODModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)

        self.hidden_dims = '16,4'
        self.rep_dim = 32

        
        self.tau = 0.01
        self.max_negatives = 1000

        

    def init_model(self, context):
        if context['output_features'] <= 40:
            self.kernel_size = 2
        elif 40 < context['output_features'] <= 160:
            self.kernel_size = 10

        elif 160 < context['output_features'] <= 240:
            self.kernel_size = context['output_features'] - 150
        elif 240 < context['output_features'] <= 480:
            self.kernel_size = context['output_features'] - 200
        else:
            self.kernel_size = context['output_features'] - 400

        if context['output_features'] < 3:
            raise ValueError('ICL model cannot handle the data that have less than three features.')

        self.net = ICLNet(
            n_features=context['output_features'],
            kernel_size=self.kernel_size,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            activation='LeakyReLU',
            bias=False
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=1e-3,
                                            weight_decay=1e-5)
    
    def forward(self, X, inference=False):
        
        positives, query = self.net(X)
        logit = self.cal_logit(query, positives)
        logit = logit.permute(0, 2, 1)
        
        correct_class = torch.zeros((logit.shape[0], logit.shape[2]),
                                    dtype=torch.long).to(self.device)
        
        loss=self.criterion(logit, correct_class).mean(dim=1)
        return X, loss

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