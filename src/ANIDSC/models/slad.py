from .base_model import *
from deepod.core.networks.base_networks import MLPnet, LinearBlock
import numpy as np
import torch.nn.functional as F
import torch



class SLAD(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    """
    Fascinating Supervisory Signals and Where to Find Them:
    Deep Anomaly Detection with Scale Learning (ICML'23)
    """
    def __init__(self, epochs=100, batch_size=128, lr=1e-3,
                 hidden_dims=100, act='LeakyReLU',
                 distribution_size=10, # the member size in a group, c in the paper
                 n_slad_ensemble=5,
                 subspace_pool_size=25,
                 magnify_factor=100,
                 n_unified_features=64, # dimensionality after transformation, h in the paper
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 n_features=100, preprocessors=[],
                 verbose=2, random_state=42, **kwargs):
        self.device=device

        BaseOnlineODModel.__init__(self,
            epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state, n_features=n_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
        self.device=device
        self.hidden_dims = hidden_dims
        self.act = act

        self.distribution_size = distribution_size
        self.n_slad_ensemble = n_slad_ensemble

        self.max_subspace_len = None # maximum length of subspace
        self.sampling_size = None # number of objects per ensemble member
        self.subspace_pool_size = subspace_pool_size

        self.n_unified_features = n_unified_features
        self.magnify_factor = magnify_factor

        self.affine_network_lst = {}
        self.subspace_indices_lst = []

        self.f_weight = torch.ones(self.n_features)
        
        self.adaptively_setting()
        
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
            activation=self.act
        ).to(self.device)
        
        
        self.optimizer=torch.optim.Adam(self.net.parameters(),
                                            lr=self.lr,
                                            weight_decay=1e-5)
        
        self.criterion= SLADLoss(reduction='none')
        self.custom_params+=["subspace_indices_lst","affine_network_lst"]
        
        
        # random transformations to transform data
        rng = np.random.RandomState(seed=self.random_state)
        for i in range(self.n_slad_ensemble):
            replace = True if self.n_features <= 10 else False
            subspace_indices = [
                rng.choice(np.arange(self.n_features), rng.choice(self.len_pool, 1), replace=replace)
                for _ in range(self.distribution_size)
            ]
            self.subspace_indices_lst.append(subspace_indices)
        

    def to_device(self, X):
        return X.to(self.device)

    def forward(self, X):
        X=self.preprocess(X)     
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
        

        
        return predict_y, y_new
    
    
    def state_dict(self):
        state = super().state_dict()
        for i in self.custom_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.custom_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
    
    def get_loss(self, X):
        predict_y, batch_y=self.forward(X)
        loss = self.criterion(predict_y, batch_y)
        loss=torch.mean(loss)
        return loss
    
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
        
    
    def train_step(self, X):
        predict_y, batch_y=self.forward(X)
        
        loss = self.criterion(predict_y, batch_y)
        #average over ensemble
        loss=loss.view(-1, self.n_slad_ensemble).mean(dim=1)
        
        loss=torch.mean(loss)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()
    
    def predict_scores(self, X):
        predict_y, batch_y=self.forward(X)
        loss = self.criterion(predict_y, batch_y)
        
        #average over ensemble
        loss=loss.view(-1, self.n_slad_ensemble).mean(dim=1).detach().cpu().numpy()
        
        return loss, self.get_threshold()

    

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

    def adaptively_setting(self):
        n_features = self.n_features

        self.max_subspace_len = n_features

        if self.subspace_pool_size is None:
            self.subspace_pool_size = min(self.max_subspace_len, 256)
        else:
            self.subspace_pool_size = min(self.subspace_pool_size, self.max_subspace_len)

        self.sampling_size = 50

        return


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