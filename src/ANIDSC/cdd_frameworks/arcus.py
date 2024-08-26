from collections import deque
import copy
import numpy as np

import torch
import math
import torch.nn as nn
import torch.optim as optim
from ..base_files.model import BaseOnlineODModel


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(
        np.dot(H, K), H
    )  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def SAP(H):
    hidden_difference = []
    for hidden_activation_origin_x, hidden_activation_recons_x in H:
        l2_norm = torch.square(
            torch.norm(hidden_activation_origin_x - hidden_activation_recons_x, dim=1)
        )
        hidden_difference.append(l2_norm)

    SAP = 0
    for difference in hidden_difference:
        SAP += difference

    return SAP


def NAP(H):
    hidden_difference = []
    for hidden_activation_origin_x, hidden_activation_recons_x in H:
        hidden_difference.append(
            hidden_activation_origin_x - hidden_activation_recons_x
        )

    # matrix D
    D = torch.cat(hidden_difference, dim=1)  # N X (d1 + d2 + ...)
    U = torch.mean(D, dim=0)  # (d1 + d2 + ...)

    D_bar = D - U  # N X (d1 + d2 + ...)

    u, s, v = torch.linalg.svd(D_bar, full_matrices=False, driver="gesvd")
    s = torch.where(s == 0, torch.ones_like(s), s)

    # When N < (d1+d2+..) tensorflow gives v of shape [(d1+d2+..), N] while torch gives [N, (d1+d2+..)]
    if D.size(0) < D.size(1):
        v = v.T
    
    diag_s=torch.diag(s)
    cond_number = torch.linalg.cond(diag_s)

    if cond_number > 1e12:
        diag_s_inv=torch.linalg.pinv(torch.diag(s))
    else:
        diag_s_inv=torch.inverse(torch.diag(s))
    NAP = torch.norm(
        torch.matmul(torch.matmul(D_bar, v), diag_s_inv), dim=1
    )

    return NAP


class RAPP(nn.Module):
    def __init__(
        self, hidden_layer_sizes, learning_rate=1e-4, activation="relu", bn=True
    ):
        super(RAPP, self).__init__()
        self.name = "RAPP"
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leakyrelu":
            activation = nn.LeakyReLU()
        else:
            raise ValueError(f"This {activation} function is not allowed")

        self.encoder_layers = nn.ModuleList()
        for idx, layer in enumerate(hidden_layer_sizes[1:]):
            self.encoder_layers.extend(
                [nn.Linear(hidden_layer_sizes[idx], layer), activation]
            )
            if bn:
                self.encoder_layers.append(nn.BatchNorm1d(layer))

        self.decoder_layers = nn.ModuleList()
        hidden_layer_sizes = hidden_layer_sizes[::-1]
        for idx, layer in enumerate(hidden_layer_sizes[1:-1]):
            self.decoder_layers.extend(
                [nn.Linear(hidden_layer_sizes[idx], layer), activation]
            )
            if bn:
                self.decoder_layers.append(nn.BatchNorm1d(layer))
        self.decoder_layers.append(
            nn.Linear(hidden_layer_sizes[-2], hidden_layer_sizes[-1])
        )

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.bn = bn

    def forward(self, x):
        latent_x = self.encoder(x)
        recons_x = self.decoder(latent_x)
        return recons_x
    
    def get_loss(self, X):
        recons_x = self.forward(X)
        loss = self.loss(X, recons_x)
        return loss

    def train_step(self, x):
        self.optimizer.zero_grad()
        recons_x = self.forward(x)
        loss = self.loss(x, recons_x)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        return loss

    def get_latent(self, x):
        latent_x = self.encoder(x)
        return latent_x

    def get_hidden_set(self, x):
        origin_x = x
        recons_x = self.forward(x)

        dense_layer = []
        activ_layer = []

        if not self.bn:
            for idx, layer in enumerate(self.encoder):
                if idx % 2 == 0:
                    dense_layer.append(layer)
                else:
                    activ_layer.append(layer)
        else:
            for idx, layer in enumerate(self.encoder):
                if idx % 3 == 0:
                    dense_layer.append(layer)
                elif idx % 3 == 1:
                    activ_layer.append(layer)
                else:
                    continue

        H = []
        temp_origin = origin_x
        temp_recons = recons_x

        for dense, activation in zip(dense_layer, activ_layer):
            hidden_activation_origin_x = activation(dense(temp_origin))
            hidden_activation_recons_x = activation(dense(temp_recons))
            H.append((hidden_activation_origin_x, hidden_activation_recons_x))
            temp_origin = hidden_activation_origin_x
            temp_recons = hidden_activation_recons_x

        self.H = H
        return H

    def inference_step(self, x):
        self.get_hidden_set(x)
        NAP_value = NAP(self.H)
        return NAP_value.detach().cpu().numpy()


class ModelGenerator:
    def __init__(self, input_dim, hidden_dim=8, layer_num=2, _model_type="RAPP"):

        self.layer_size = []
        gap = (input_dim - hidden_dim) / (layer_num - 1)
        for idx in range(layer_num):
            self.layer_size.append(int(input_dim - (gap * idx)))

        self.model_type = _model_type
        self.learning_rate = 1e-4

    def init_model(self):
        # initialize ARCUS framework
        if self.model_type == "RAPP":
            model = RAPP(
                hidden_layer_sizes=self.layer_size,
                activation="relu",
                learning_rate=self.learning_rate,
                bn=True,
            )

        """
        You can add your autoencoder model here by inheriting Base in model_base.py.
        """

        model.num_batch = 0
        return model




class ARCUS(BaseOnlineODModel):
    def __init__(
        self, model: BaseOnlineODModel,
        reliability_thred = 0.95,
        similarity_thred = 0.80,
        **kwargs
    ):
        
        super().__init__(**kwargs)

        self.model = model
        self.model_pool = []
        self.model_idx = 0
        self.num_trained = 0
        self.reliability_thred = reliability_thred
        self.similarity_thred = similarity_thred
        self.max_model_pool_size = 20
        
        self.context={}
        
        self.name = self.__str__()
    
    def forward(self, data, inference=True):
        pass
    
    def predict_step(self, X, preprocess=False):
        confidence_driven_score=0
        self.product_rm=1
        max_reliability=-1
        self.untrained_idx=None
        
        for i, m in enumerate(self.model_pool):
            score, threshold=m.predict_step(X, preprocess=True)
            
            # Kitsune might return None when constructing feature mapper
            if score is None:
                self.untrained_idx=i
                continue
            
            curr_mean_score = np.mean(score)
            curr_max_score = np.max(score)
            curr_min_score = np.min(score)
            
            if hasattr(m, "last_min_score"):
            
                min_score = (
                    curr_min_score
                    if curr_min_score < m.last_min_score
                    else m.last_min_score
                )
                max_score = (
                    curr_max_score
                    if curr_max_score > m.last_max_score
                    else m.last_max_score
                )
                gap = np.abs(curr_mean_score - m.last_mean_score)
                reliability = np.exp(
                        -2
                        * gap
                        * gap
                        / (
                            (2 / X.shape[0])
                            * (max_score - min_score)
                            * (max_score - min_score)
                        )
                    )
            else:
                m.last_mean_score=curr_mean_score
                m.last_max_score=curr_max_score
                m.last_min_score=curr_min_score
                reliability=1
            
            # keep track of untrained index
            if m.num_trained<m.warmup//10:
                self.untrained_idx=i 
                
            if max_reliability<reliability:
                self.model_idx=i
                max_reliability=reliability
                      
            self.product_rm*=(1-reliability)
            
            #concept_drift anomaly score 
            confidence_driven_score+=reliability*(score-np.mean(score))/np.std(score)
        
        self.num_evaluated += 1

        # all model return None
        if max_reliability==-1:
            return None, None
        
        return confidence_driven_score, self.get_threshold()
    
    def train_step(self, X, preprocess=False):
        self.num_trained+=1
        return self.model_pool[self.model_idx].train_step(X, preprocess=preprocess)
         
    
    def process(self, data):
        # anomaly detection 
        confidence_driven_score, threshold=self.predict_step(data)
        
        # reliability of model pool
        Rp=1-self.product_rm
        
        #train if model is reliable or still in warmup stage
        if Rp>=self.reliability_thred:
            self.train_step(data, preprocess=True)
        
        # if there are untrained model
        elif self.untrained_idx is not None:
            self.model_idx=self.untrained_idx
            self.train_step(data, preprocess=True)
        
        # create new mdoel
        else:    
            self.init_model()
            self.train_step(data, preprocess=True)
        
        # periodically clean
        if self.num_trained%1000==0:
            self.reduce_model_pool(data)
        
        if confidence_driven_score is not None:
            self.loss_queue.append(np.mean(confidence_driven_score))
            
        return {
            "drift_level": Rp,
            "threshold": threshold,
            "score": confidence_driven_score,
            "batch_num": self.num_trained,
            "model_batch_num": self.model_pool[self.model_idx].num_trained,
            "model_idx": self.model_idx,
            "num_model": len(self.model_pool),
            "trained":True
        }
    
    def reduce_model_pool(self, data):
        
        latents = []
        # keep track of valid models
        valid_models=[]
        for i, m in enumerate(self.model_pool):
            if m.num_trained<m.warmup:
                continue 
            processed_data=m.preprocess(data)
            z, _ = m.forward(processed_data)
            
            # arcus only works with 1d latents, flatten if greater than 2d
            if z.ndim >2:
                z=z.reshape(z.size(0),-1)
            latents.append(z.detach().cpu().numpy())
            valid_models.append(i)
        
        if len(valid_models)<2:
            return 
        
        max_CKA = 0 
        max_Idx1 = None
        for i, latent in zip(valid_models[:-1], latents[:-1]):
            CKA = linear_CKA(latent, latents[-1])
            if CKA > max_CKA:
                max_CKA = CKA
                max_Idx1 = i

        if max_Idx1 is not None and max_CKA >= self.similarity_thred:
            self.model_pool[valid_models[-1]] = self.merge_models(max_Idx1, valid_models[-1])
            self.model_pool.pop(max_Idx1)
            
            # recursively remove
            if len(self.model_pool)>1:
                self.reduce_model_pool(data)
        
        self.model_idx=-1
    
    def merge_models(self, idx1, idx2):
        model1=self.model_pool[idx1]
        model2=self.model_pool[idx2]
        
        num_batch_sum = model1.num_trained + model2.num_trained
        w1 = model1.num_trained / num_batch_sum
        w2 = model2.num_trained / num_batch_sum
        
        # merged_model = copy.deepcopy(model1)
        
        with torch.no_grad():
            for param1, param2 in zip(model1.parameters(), model2.parameters()):
                param2.data.copy_(param1.data *w1 + param2.data *w2 )
        
        return model2
        
    def init_model(self, context=None) -> BaseOnlineODModel:
        """creates a new model based on base model

        Returns:
            BaseOnlineODModel: the new model
        """
        new_model = copy.deepcopy(self.model)
        new_model.parent = self
        new_model.setup()
        self.model_idx=-1
        self.model_pool.append(new_model)
    
    def setup(self):
        if not self.loaded_from_file:
            self.init_model()
        self.parent.context["concept_drift_detection"] = True
        self.parent.context["model_name"] = f"{self.name}"
        
        

    def __str__(self):
        return f"ARCUS({self.model})"