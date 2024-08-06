from collections import deque
import copy
import numpy as np

import torch
import math
import torch.nn as nn
import torch.optim as optim

from ..base_files.model import BaseOnlineODModel
from ..base_files.pipeline import PipelineComponent


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


# class ARCUS(BaseOnlineODModel, torch.nn.Module):
#     def __init__(self, **kwargs):
#         BaseOnlineODModel.__init__(self,loss_dist="norm",**kwargs)
#         torch.nn.Module.__init__(self)
#         self.model_pool = []
#         self._init_epoch = 5
#         self._intm_epoch = 1
#         self._reliability_thred = 0.95
#         self._similarity_thred = 0.80
#         self._min_batch_size = 32
#         self._model_type="RAPP"

#     def init_model(self, context):
#         self._itr_num = context["batch_size"] // self._min_batch_size
#         self._model_generator = ModelGenerator(input_dim=context["output_features"])
#         self._batch_size = context["batch_size"]
#         initial_model = self._model_generator.init_model()
#         initial_model.to(self.device)
#         self.model_pool.append(initial_model)

#     def get_total_params(self):
#         return sum(
#             p.numel() for p in self.model_pool[0].parameters() if p.requires_grad
#         )

#     def _standardize_scores(self, score):
#         # get standardized anomaly scores
#         mean_score = np.mean(score)
#         std_score = np.std(score)

#         standardized_score = np.array([(k - mean_score) / std_score for k in score])
#         return standardized_score

#     def _merge_models(self, model1, model2):
#         # Merge a previous model and a current model
#         num_batch_sum = model1.num_batch + model2.num_batch
#         w1 = model1.num_batch / num_batch_sum
#         w2 = model2.num_batch / num_batch_sum

#         # Merge encoder
#         for layer_idx in range(len(model2.encoder_layers)):
#             l_base = model1.encoder_layers[layer_idx]
#             l_target = model2.encoder_layers[layer_idx]
#             if isinstance(l_base, nn.Linear) or isinstance(l_base, nn.BatchNorm1d):
#                 new_weight = l_base.weight.data * w1 + l_target.weight.data * w2
#                 new_bias = l_base.bias.data * w1 + l_target.bias.data * w2
#                 if isinstance(l_base, nn.Linear):
#                     l_target.weight.data = new_weight
#                     l_target.bias.data = new_bias
#                 elif isinstance(l_base, nn.BatchNorm1d):
#                     new_gamma = l_base.weight.data * w1 + l_target.weight.data * w2
#                     new_beta = l_base.bias.data * w1 + l_target.bias.data * w2
#                     new_mm = l_base.running_mean * w1 + l_target.running_mean * w2
#                     new_mv = l_base.running_var * w1 + l_target.running_var * w2
#                     l_target.weight.data = new_gamma
#                     l_target.bias.data = new_beta
#                     l_target.running_mean = new_mm
#                     l_target.running_var = new_mv

#         # Merge decoder
#         for layer_idx in range(len(model2.decoder_layers)):
#             l_base = model1.decoder_layers[layer_idx]
#             l_target = model2.decoder_layers[layer_idx]
#             if isinstance(l_base, nn.Linear) or isinstance(l_base, nn.BatchNorm1d):
#                 new_weight = l_base.weight.data * w1 + l_target.weight.data * w2
#                 new_bias = l_base.bias.data * w1 + l_target.bias.data * w2
#                 if isinstance(l_base, nn.Linear):
#                     l_target.weight.data = new_weight
#                     l_target.bias.data = new_bias
#                 elif isinstance(l_base, nn.BatchNorm1d):
#                     new_gamma = l_base.weight.data * w1 + l_target.weight.data * w2
#                     new_beta = l_base.bias.data * w1 + l_target.bias.data * w2
#                     new_mm = l_base.running_mean * w1 + l_target.running_mean * w2
#                     new_mv = l_base.running_var * w1 + l_target.running_var * w2
#                     l_target.weight.data = new_gamma
#                     l_target.bias.data = new_beta
#                     l_target.running_mean = new_mm
#                     l_target.running_var = new_mv

#         if self._model_type == "RSRAE":
#             model2.A.data = model1.A.data * w1 + model2.A.data * w2

#         model2.num_batch = num_batch_sum
#         return model2

#     def _reduce_models_last(self, x_inp, epochs):
#         # delete similar models for reducing redundancy in a model pool
#         latents = []
#         for m in self.model_pool:
#             z = m.get_latent(x_inp)
#             latents.append(z.detach().cpu().numpy())

#         max_CKA = 0
#         max_Idx1 = None
#         max_Idx2 = len(latents) - 1
#         for idx1 in range(len(latents) - 1):
#             CKA = linear_CKA(latents[idx1], latents[max_Idx2])
#             if CKA > max_CKA:
#                 max_CKA = CKA
#                 max_Idx1 = idx1

#         if max_Idx1 != None and max_CKA >= self._similarity_thred:
#             self.model_pool[max_Idx2] = self._merge_models(
#                 self.model_pool[max_Idx1], self.model_pool[max_Idx2]
#             )
#             self._train_model(
#                 self.model_pool[max_Idx2], x_inp, epochs
#             )  # Train just one epoch to get the latest score info

#             self.model_pool.remove(self.model_pool[max_Idx1])
#             if len(self.model_pool) > 1:
#                 self._reduce_models_last(x_inp, epochs)

#     def get_loss(self, X, preprocess=False):
#         if preprocess:
#             X = self.preprocess(X).float()
        
        
#         indices = torch.randperm(X.size(0))
#         min_batch_x_inp = X[indices[: self._min_batch_size]]

#         return self.curr_model.get_loss(min_batch_x_inp)    
    

#     def _train_model(self, model, x_inp, epochs):
#         # train a model in the model pool of ARCUS
#         tmp_losses = []
#         for _ in range(epochs):
#             for _ in range(self._itr_num):
#                 indices = torch.randperm(x_inp.size(0))
#                 min_batch_x_inp = x_inp[indices[: self._min_batch_size]]

#                 loss = model.train_step(min_batch_x_inp)
#                 tmp_losses.append(loss.detach().cpu().numpy())
#         temp_scores = model.inference_step(x_inp)
#         if np.isfinite(temp_scores).all():
#             model.last_mean_score = np.mean(temp_scores)
#             model.last_max_score = np.max(temp_scores)
#             model.last_min_score = np.min(temp_scores)
#         model.num_batch = model.num_batch + 1
        
#         # update scaler
#         context=self.get_context()
#         if "scaler" in context.keys():
#             context["scaler"].update_current()
#         return tmp_losses

#     def forward(self, X, inference=False):
#         """not used since process is overriden"""
#         pass

#     def predict_step(self, X, preprocess=False):
#         threshold = self.get_threshold()
#         if preprocess:
#             X = self.preprocess(X).float()
            
#         if self.num_trained == 0:
#             self._train_model(self.model_pool[0], X, self._init_epoch)

#         # reliability calculation
#         scores = []
#         self.model_reliabilities = []
#         for m in self.model_pool:
#             scores.append(m.inference_step(X))
#             curr_mean_score = np.mean(scores[-1])
#             curr_max_score = np.max(scores[-1])
#             curr_min_score = np.min(scores[-1])
#             min_score = (
#                 curr_min_score
#                 if curr_min_score < m.last_min_score
#                 else m.last_min_score
#             )
#             max_score = (
#                 curr_max_score
#                 if curr_max_score > m.last_max_score
#                 else m.last_max_score
#             )
#             gap = np.abs(curr_mean_score - m.last_mean_score)
#             reliability = np.round(
#                 np.exp(
#                     -2
#                     * gap
#                     * gap
#                     / (
#                         (2 / self._batch_size)
#                         * (max_score - min_score)
#                         * (max_score - min_score)
#                     )
#                 ),
#                 4,
#             )
#             self.model_reliabilities.append(reliability)

#         curr_model_index = self.model_reliabilities.index(max(self.model_reliabilities))
#         self.curr_model = self.model_pool[curr_model_index]

#         weighted_scores = []
#         for idx in range(len(self.model_pool)):
#             weight = self.model_reliabilities[idx]
#             weighted_scores.append(self._standardize_scores(scores[idx]) * weight)
#         final_scores = np.sum(weighted_scores, axis=0)
#         self.num_evaluated+=1
#         return final_scores, threshold

#     def train_step(self, X, preprocess=False):
#         if preprocess:
#             X = self.preprocess(X).float()
            
#         if self.drift:
#             # Create new model
#             new_model = self._model_generator.init_model()
#             new_model.to(self.device)
#             loss=self._train_model(new_model, X, self._init_epoch)
#             self.model_pool.append(new_model)
#             # Merge models
#             self._reduce_models_last(X, 1)

#         else:
#             loss=self._train_model(self.curr_model, X, self._intm_epoch)
            
#         self.num_trained += 1
        
#         return np.mean(loss)

#     def process(self, X):
#         X_scaled = self.preprocess(X)

#         final_scores, threshold = self.predict_step(X_scaled)
        
#         if final_scores is not None:
#             self.loss_queue.extend(final_scores)

#         # drift detection
#         pool_reliability = 1 - np.prod([1 - p for p in self.model_reliabilities])

#         self.drift=pool_reliability < self._reliability_thred

#         self.train_step(X_scaled, False)
        
#         return {
#             "drift_level": pool_reliability,
#             "threshold": threshold,
#             "score": final_scores,
#             "batch_num": self.num_evaluated,
#             "num_model": len(self.model_pool),
#         }


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
        for i, m in enumerate(self.model_pool):
            score, threshold=m.predict_step(X, preprocess=True)
            
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
            
            if max_reliability<reliability:
                self.model_idx=i 
                max_reliability=reliability
                      
            self.product_rm*=(1-reliability)
            
            #concept_drift anomaly score 
            confidence_driven_score+=reliability*(score-np.mean(score))/np.std(score)
        
        self.num_evaluated += 1
        return confidence_driven_score, self.get_threshold()
    
    def train_step(self, X, preprocess=False):
        return self.model_pool[self.model_idx].train_step(X, preprocess=preprocess)
         
    
    def process(self, data):
        # anomaly detection 
        confidence_driven_score, threshold=self.predict_step(data)
        
        # reliability of model pool
        Rp=1-self.product_rm
        
        if Rp>=self.reliability_thred:
            self.train_step(data, preprocess=True)
        else:
            self.init_model()
            self.train_step(data, preprocess=True)
            self.reduce_model_pool(data)
        
        self.loss_queue.append(confidence_driven_score)
        return {
            "drift_level": Rp,
            "threshold": threshold,
            "score": confidence_driven_score,
            "batch_num": self.num_trained,
            "model_batch_num": self.model_pool[self.model_idx].num_trained,
            "model_idx": self.model_idx,
            "num_model": len(self.model_pool),
        }
    
    def reduce_model_pool(self, data):
        
        latents = []
        for m in self.model_pool:
            processed_data=m.preprocess(data)
            z, _ = m.forward(processed_data)
            latents.append(z.numpy())

        max_CKA = 0 
        max_Idx1 = None
        for i, latent in enumerate(latents[:-1]):
            CKA = linear_CKA(latent, latents[-1])
            if CKA > max_CKA:
                max_CKA = CKA
                max_Idx1 = i

        if max_Idx1 != None and max_CKA >= self.similarity_thred:
            self.model_pool[-1] = self.merge_models(max_Idx1, -1)
            self.model_pool.pop(max_Idx1)
            
            # recursively remove
            if len(self.model_pool)>1:
                self.reduce_model_pool(data)
    
    def merge_models(self, idx1, idx2):
        model1=self.model_pool[idx1]
        model2=self.model_pool[idx2]
        
        num_batch_sum = model1.num_trained + model2.num_trained
        w1 = model1.num_trained / num_batch_sum
        w2 = model2.num_trained / num_batch_sum
        
        merged_model = copy.deepcopy(model1)
        
        with torch.no_grad():
            for param1, param2, merged_param in zip(model1.parameters(), model2.parameters(), merged_model.parameters()):
                merged_param.data.copy_(param1.data *w1 + param2.data *w2 )
        
        return merged_model
        
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