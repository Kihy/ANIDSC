from deepod.core.networks.base_networks import MLPnet, LinearBlock
import numpy as np
import torch.nn.functional as F
import torch

from ANIDSC.base_files.model import BaseOnlineODModel



class SLAD(BaseOnlineODModel,torch.nn.Module):
    """
    Fascinating Supervisory Signals and Where to Find Them:
    Deep Anomaly Detection with Scale Learning (ICML'23)
    """
    def __init__(self, **kwargs):
        
        BaseOnlineODModel.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        
        self.hidden_dims = 8
        self.distribution_size = 10
        self.n_slad_ensemble = 4

        self.max_subspace_len = None # maximum length of subspace
        self.sampling_size = None # number of objects per ensemble member
        self.subspace_pool_size = 32

        self.n_unified_features = 16
        self.magnify_factor = 200

        self.affine_network_lst = {}
        self.subspace_indices_lst = []

    def init_model(self, context):
        self.f_weight = torch.ones(context["output_features"])
        
        self.max_subspace_len = context["output_features"]

        if self.subspace_pool_size is None:
            self.subspace_pool_size = min(self.max_subspace_len, 256)
        else:
            self.subspace_pool_size = min(self.subspace_pool_size, self.max_subspace_len)

        self.sampling_size = 100
        
        # randomly determines the pool of subspace sizes
        self.len_pool = np.sort(np.random.choice(np.arange(1, self.max_subspace_len+1),
                                                 self.subspace_pool_size,
                                                 replace=False))
        
        for s in self.len_pool:
            self.affine_network_lst[s] = LinearBlock(
                in_channels=s, out_channels=self.n_unified_features,
                bias=False, activation=None
            ).to(self.device)

        self.net = MLPnet(
            n_features=self.n_unified_features,
            n_hidden=self.hidden_dims,
            n_output=1,
            activation='LeakyReLU'
        ).to(self.device)
        
        
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=1e-3,
                                            weight_decay=1e-5)
        
        self.criterion= SLADLoss(reduction='none')        
        
        # random transformations to transform data
        rng = np.random.RandomState()
        for i in range(self.n_slad_ensemble):
            replace = True if context["output_features"] <= 10 else False
            subspace_indices = [
                rng.choice(np.arange(context["output_features"]), rng.choice(self.len_pool, 1), replace=replace)
                for _ in range(self.distribution_size)
            ]
            self.subspace_indices_lst.append(subspace_indices)
        
    def forward(self, X, inference=False):
        
        x_new_lst = []
        y_new_lst = []
        
        for i in range(self.n_slad_ensemble):
            subspace_indices = self.subspace_indices_lst[i]
            # the size of newly generated data: [sampling_size, distribution_size(c), n_unified_features(h)]
            batch_x, batch_y = self._transform_data(X, subspace_indices, torch.arange(len(X)))
                        
            x_new_lst.append(batch_x)
            y_new_lst.append(batch_y)
            
        x_new = torch.vstack(x_new_lst)
        y_new = torch.vstack(y_new_lst)
        
        predict_y = self.net(x_new)
        predict_y = predict_y.squeeze(dim=2)
        
        loss=self.criterion(predict_y, y_new).view(-1, self.n_slad_ensemble).mean(dim=1)
        return X, loss


    def _transform_data(self, X, subspace_indices, subset_idx):
        """generate new data sets with supervision according to subspace indices"""

        # x_new = [np.zeros([len(subset_idx), self.distribution_size, self.n_unified_features])]
        # y_new = np.zeros([len(subset_idx), self.distribution_size])

        x_new=[]
        y_new=[]

        for ii, subspace_idx in enumerate(subspace_indices):
            n_f = len(subspace_idx)

            # # get the subspace
            # # use two steps here, otherwise the output shape is 1-dim when subspace is with 1 feature
            x_sub = X[subset_idx, :]
            x_sub = x_sub[:, subspace_idx]

            # # transform function: get transformed vectors
            x_sub_projected = self._transformation_function(x_sub, n_f)

            # # use feature weight / number of features to set supervision signals
            target = (torch.sum(self.f_weight[subspace_idx]) / self.n_unified_features) * self.magnify_factor
            
            y_true = torch.full((x_sub.shape[0],), target)

            # x_new[:, ii, :], y_new[:, ii] = x_sub_projected, y_true

            x_new.append(x_sub_projected)
            y_new.append(y_true)
            
        
        return torch.stack(x_new, 1).float().to(self.device), torch.stack(y_new,1).float().to(self.device)

    def _transformation_function(self, x, n_f):
        """transform phase to obtain a unified dimensionality"""

        # # affine transform
        transform_net = self.affine_network_lst[n_f]

        transform_net.eval()
        with torch.no_grad():
            x_projected = transform_net(x.float()) #.data.cpu().numpy()
        return x_projected




class SLADLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SLADLoss, self).__init__()
        assert reduction in ['mean', 'none', 'sum'], 'unsupported reduction operation'

        self.reduction = reduction
        self.kl = torch.nn.KLDivLoss(reduction='none')

        return

    def forward(self, y_pred, y_true):
        """
        forward function

        Parameters:
        y_pred: torch.Tensor, shape = [batch_size, distribution_size]
            output of the network

        y_true: torch.Tensor, shape = [batch_size, distribution_size]
            ground truth labels

        return_raw_results: bool, default=False
            return the raw results with shape [batch_size, distribution_size]
            accompanied by reduced loss value

        Return:

        loss value: torch.Tensor
            reduced loss value
        """

        
        reduction = self.reduction
        
        #add small eps value
        preds_smax = F.softmax(y_pred, dim=1)+1e-6
        true_smax = F.softmax(y_true, dim=1)+1e-6

        total_m = 0.5 * (preds_smax + true_smax)

        js1 = F.kl_div(F.log_softmax(preds_smax, dim=1), total_m, reduction='none')
        
        js2 = F.kl_div(F.log_softmax(true_smax, dim=1), total_m, reduction='none')
        js = torch.sum(js1 + js2, dim=1)

        if reduction == 'mean':
            loss = torch.mean(js)
        elif reduction == 'sum':
            loss = torch.sum(js)
        elif reduction == 'none':
            loss = js
        else:
            return

        return loss