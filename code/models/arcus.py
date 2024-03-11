from .base_model import *
import numpy as np
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
import torch.optim as optim

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
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
        l2_norm = torch.square(torch.norm(hidden_activation_origin_x - hidden_activation_recons_x, dim=1))
        hidden_difference.append(l2_norm)
    
    SAP = 0
    for difference in hidden_difference:
        SAP += difference
    
    return SAP

def NAP(H):
    hidden_difference = []
    for hidden_activation_origin_x, hidden_activation_recons_x in H:
        hidden_difference.append(hidden_activation_origin_x - hidden_activation_recons_x)

    # matrix D
    D = torch.cat(hidden_difference, dim=1)  # N X (d1 + d2 + ...)
    U = torch.mean(D, dim=0)  # (d1 + d2 + ...)
    
    D_bar = D - U  # N X (d1 + d2 + ...)
    u, s, v = torch.linalg.svd(D_bar,full_matrices=False)
    s = torch.where(s == 0, torch.ones_like(s), s)
    
    # When N < (d1+d2+..) tensorflow gives v of shape [(d1+d2+..), N] while torch gives [N, (d1+d2+..)] 
    if D.size(0)<D.size(1): 
        v=v.T
    NAP = torch.norm(torch.matmul(torch.matmul(D_bar, v), torch.inverse(torch.diag(s))), dim=1)
    
    return NAP

class RAPP(nn.Module):
    def __init__(self, hidden_layer_sizes, learning_rate=1e-4, activation='relu', bn=True):
        super(RAPP, self).__init__()
        self.name="RAPP"
        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU()
        else:
            raise ValueError(f'This {activation} function is not allowed')

        self.encoder_layers = nn.ModuleList()
        for idx, layer in enumerate(hidden_layer_sizes[1:]):
            self.encoder_layers.extend([
                nn.Linear(hidden_layer_sizes[idx], layer),
                activation
            ])
            if bn:
                self.encoder_layers.append(nn.BatchNorm1d(layer))

        self.decoder_layers = nn.ModuleList()
        hidden_layer_sizes=hidden_layer_sizes[::-1]
        for idx, layer in enumerate(hidden_layer_sizes[1:-1]):
            self.decoder_layers.extend([
                nn.Linear(hidden_layer_sizes[idx], layer),
                activation
            ])
            if bn:
                self.decoder_layers.append(nn.BatchNorm1d(layer))
        self.decoder_layers.append(nn.Linear(hidden_layer_sizes[-2], hidden_layer_sizes[-1]))
        
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.bn = bn

    def forward(self, x):
        latent_x = self.encoder(x)
        recons_x = self.decoder(latent_x)
        return recons_x

    def train_step(self, x):
        self.optimizer.zero_grad()
        recons_x = self.forward(x)
        loss = self.loss(x, recons_x)
        loss.backward()
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

        if self.bn == False:
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
    def __init__(self, **kwargs):
        
        self.layer_size = []   
        gap = (kwargs["input_dim"] - kwargs["hidden_dim"])/(kwargs["layer_num"]-1)
        for idx in range(kwargs["layer_num"]):
            self.layer_size.append(int(kwargs["input_dim"]-(gap*idx)))
        
        self.model_type = kwargs["_model_type"]
        self.learning_rate = kwargs["learning_rate"]
        
        # For RSRAE
        if kwargs["_model_type"] == 'RSRAE':
            self.input_dim = kwargs["input_dim"]
            self.intrinsic_size = kwargs["intrinsic_size"]
            self.RSRAE_hidden_layer_size = kwargs["RSRAE_hidden_layer_size"]
        
    def init_model(self):
        # initialize ARCUS framework


        if self.model_type == "RAPP":
            model = RAPP(hidden_layer_sizes = self.layer_size,
                         activation = 'relu',
                         learning_rate = self.learning_rate,
                         bn = True,)

        '''
        You can add your autoencoder model here by inheriting Base in model_base.py.
        '''

        model.num_batch = 0
        return model

class ARCUS(BaseOnlineODModel,torch.nn.Module, TorchSaveMixin):
    def __init__(self, device='cuda',
                 n_features=100, preprocessors=[],
                 **kwargs):
        self.device=device
        preprocessors.append(self.normalize)
        preprocessors.append(self.to_device)
        BaseOnlineODModel.__init__(self,
            model_name='ARCUS', n_features=n_features,
            preprocessors=preprocessors, **kwargs
        )
        torch.nn.Module.__init__(self)
        
        self._itr_num    = int(kwargs["_batch_size"]/kwargs["_min_batch_size"])
        self._model_generator = ModelGenerator(input_dim=n_features, **kwargs)
        self.model_pool=[]
        self.additional_params+=["model_pool", "steps"]
        self.steps=0
    
    def _standardize_scores(self, score):
        # get standardized anomaly scores
        mean_score = np.mean(score)
        std_score  = np.std(score)

        standardized_score = np.array([(k-mean_score) / std_score for k in score])
        return standardized_score
    
    def state_dict(self):
        state = super().state_dict()
        for i in self.additional_params:
            state[i] = getattr(self, i)
        return state
    
    def load_state_dict(self, state_dict):

        for i in self.additional_params:
            setattr(self, i, state_dict[i])
            del state_dict[i]
    
    def _merge_models(self, model1, model2):
        # Merge a previous model and a current model
        num_batch_sum = model1.num_batch + model2.num_batch
        w1 = model1.num_batch / num_batch_sum
        w2 = model2.num_batch / num_batch_sum

        # Merge encoder
        for layer_idx in range(len(model2.encoder_layers)):
            l_base = model1.encoder_layers[layer_idx]
            l_target = model2.encoder_layers[layer_idx]
            if isinstance(l_base, nn.Linear) or isinstance(l_base, nn.BatchNorm1d):
                new_weight = (l_base.weight.data * w1 + l_target.weight.data * w2)
                new_bias = (l_base.bias.data * w1 + l_target.bias.data * w2)
                if isinstance(l_base, nn.Linear):
                    l_target.weight.data = new_weight
                    l_target.bias.data = new_bias
                elif isinstance(l_base, nn.BatchNorm1d):
                    new_gamma = (l_base.weight.data * w1 + l_target.weight.data * w2)
                    new_beta = (l_base.bias.data * w1 + l_target.bias.data * w2)
                    new_mm = (l_base.running_mean * w1 + l_target.running_mean * w2)
                    new_mv = (l_base.running_var * w1 + l_target.running_var * w2)
                    l_target.weight.data = new_gamma
                    l_target.bias.data = new_beta
                    l_target.running_mean = new_mm
                    l_target.running_var = new_mv

        # Merge decoder
        for layer_idx in range(len(model2.decoder_layers)):
            l_base = model1.decoder_layers[layer_idx]
            l_target = model2.decoder_layers[layer_idx]
            if isinstance(l_base, nn.Linear) or isinstance(l_base, nn.BatchNorm1d):
                new_weight = (l_base.weight.data * w1 + l_target.weight.data * w2)
                new_bias = (l_base.bias.data * w1 + l_target.bias.data * w2)
                if isinstance(l_base, nn.Linear):
                    l_target.weight.data = new_weight
                    l_target.bias.data = new_bias
                elif isinstance(l_base, nn.BatchNorm1d):
                    new_gamma = (l_base.weight.data * w1 + l_target.weight.data * w2)
                    new_beta = (l_base.bias.data * w1 + l_target.bias.data * w2)
                    new_mm = (l_base.running_mean * w1 + l_target.running_mean * w2)
                    new_mv = (l_base.running_var * w1 + l_target.running_var * w2)
                    l_target.weight.data = new_gamma
                    l_target.bias.data = new_beta
                    l_target.running_mean = new_mm
                    l_target.running_var = new_mv

        if self._model_type == 'RSRAE':
            model2.A.data = (model1.A.data * w1 + model2.A.data * w2)

        model2.num_batch = num_batch_sum
        return model2
    
    def _reduce_models_last(self, x_inp, epochs):
        # delete similar models for reducing redundancy in a model pool
        latents = []
        for m in self.model_pool:
            z = m.get_latent(x_inp)
            latents.append(z.detach().cpu().numpy())

        max_CKA = 0 
        max_Idx1 = None
        max_Idx2 = len(latents)-1
        for idx1 in range(len(latents)-1):
            CKA = linear_CKA(latents[idx1], latents[max_Idx2])
            if CKA > max_CKA:
                max_CKA = CKA
                max_Idx1 = idx1

        if max_Idx1 != None and max_CKA >= self._similarity_thred:
            self.model_pool[max_Idx2] = self._merge_models(self.model_pool[max_Idx1], self.model_pool[max_Idx2])
            self._train_model(self.model_pool[max_Idx2], x_inp, epochs) # Train just one epoch to get the latest score info

            self.model_pool.remove(self.model_pool[max_Idx1])       
            if len(self.model_pool) > 1:
                self._reduce_models_last(x_inp, epochs)
    
    def _train_model(self, model, x_inp, epochs):
        # train a model in the model pool of ARCUS
        tmp_losses = []
        for _ in range(epochs):
            for _ in range(self._itr_num):
                indices = torch.randperm(x_inp.size(0))
                min_batch_x_inp = x_inp[indices[:self._min_batch_size]]
                
                loss = model.train_step(min_batch_x_inp)
                tmp_losses.append(loss.detach().cpu().numpy())
        temp_scores = model.inference_step(x_inp)
        model.last_mean_score = np.mean(temp_scores)
        model.last_max_score = np.max(temp_scores)
        model.last_min_score = np.min(temp_scores)
        model.num_batch = model.num_batch+1
        return tmp_losses
    
    def process(self, x):
        threshold=self.get_threshold()
        x=self.preprocess(x).float()
        
        if self.steps==0:
            initial_model = self._model_generator.init_model()
            initial_model.to(self.device)
            self.model_pool.append(initial_model)
            self._train_model(initial_model, x, self._init_epoch)
        
        #reliability calculation
        scores = []
        model_reliabilities = []
        for m in self.model_pool:
            scores.append(m.inference_step(x))
            curr_mean_score = np.mean(scores[-1])
            curr_max_score = np.max(scores[-1])
            curr_min_score = np.min(scores[-1])
            min_score = curr_min_score if curr_min_score < m.last_min_score else m.last_min_score
            max_score = curr_max_score if curr_max_score > m.last_max_score else m.last_max_score
            gap = np.abs(curr_mean_score - m.last_mean_score)
            reliability = np.round(np.exp(-2*gap*gap/((2/self._batch_size)*(max_score-min_score)*(max_score-min_score))),4)
            model_reliabilities.append(reliability)

        curr_model_index = model_reliabilities.index(max(model_reliabilities))
        curr_model = self.model_pool[curr_model_index]
        
        weighted_scores = []
        for idx in range(len(self.model_pool)):
            weight = model_reliabilities[idx]
            weighted_scores.append(self._standardize_scores(scores[idx]) * weight)
        final_scores = np.sum(weighted_scores, axis=0)
        
        
        self.score_hist.extend(final_scores)
        
        #drift detection
        pool_reliability = 1-np.prod([1-p for p in model_reliabilities])

        if pool_reliability < self._reliability_thred:
            drift = True
        else:
            drift = False
        
 
        if drift:
            #Create new model
            new_model = self._model_generator.init_model()
            new_model.to(self.device)
            self._train_model(new_model, x, self._init_epoch)
            self.model_pool.append(new_model)
            #Merge models
            self._reduce_models_last(x, 1)

        else:
            self._train_model(curr_model, x, self._intm_epoch)
            
        self.steps+=1
        return final_scores, threshold
    
    
    def to_device(self, X):
        return X.to(self.device)