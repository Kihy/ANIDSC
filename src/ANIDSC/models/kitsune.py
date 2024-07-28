import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.stats import norm

from ANIDSC.base_files.model import BaseOnlineODModel


np.seterr(all="ignore")


# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document


class KitNET(BaseOnlineODModel):
    # n: the number of features in your input dataset (i.e., x \in R^n)
    # m: the maximum size of any autoencoder in the ensemble layer
    # AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    # FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    # learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    # hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    # feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self, **kwargs):
        BaseOnlineODModel.__init__(self, **kwargs)
        # Parameters:        
        self.FM_grace_period = 40
        self.input_precision = None
        self.m = 10
        self.lr = 0.1
        self.hr = 0.75

        # Variables
        self.v = None
        self.ensembleLayer = []
        self.outputLayer = None
        self.quantize = None
    
    def init_model(self, context):
        if self.v is None:
            pass
            # print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        # incremental feature cluatering for the feature mapping process
        self.FM = corClust(context["output_features"])
    
    def forward(self, X, inference=False):
        pass
    
    
    def train_step(self, X, preprocess=False):
        if preprocess:
            X=self.preprocess(X)
        loss=[self.train_single(i) for i in X]
        self.num_trained+=1
        
        # update scaler
        context=self.get_context()
        if "scaler" in context.keys():
            context["scaler"].update_current()
        
        return np.array(loss) 
    
    def predict_step(self, X,preprocess=False):
        if self.v is None:
            return None, None
        
        if preprocess:
            X=self.preprocess(X)
        
        loss=[self.execute_single(i) for i in X]
        self.num_evaluated+=1
        return np.array(loss), self.get_threshold()

    # force train KitNET on x
    # returns the anomaly score of x during training (do not use for alerting)
    def train_single(self, X):
        # If the FM is in train-mode, and the user has not supplied a feature mapping
        if self.num_trained <= self.FM_grace_period and self.v is None:
            # update the incremetnal correlation matrix
            self.FM.update(X)
            if (
                self.num_trained == self.FM_grace_period
            ):  # If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                # print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
                # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
            return 0.
        else:  # train
            # Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = X[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            # OutputLayer
            return self.outputLayer.train(S_l1)
            
    # force execute KitNET on x
    def execute_single(self, x):
        
        # Ensemble Layer
        S_l1 = np.zeros(len(self.ensembleLayer))
        for a in range(len(self.ensembleLayer)):
            # make sub inst
            xi = x[self.v[a]]
            S_l1[a] = self.ensembleLayer[a].execute(xi)
        # OutputLayer
        return self.outputLayer.execute(S_l1)

    def get_total_params(self):
        if self.v is None:
            return 0
        else:
            total_params=0
            for i in range(len(self.ensembleLayer)):
                params=self.ensembleLayer[i].get_params()
                total_params+=params["W"].size
                total_params+=params["hbias"].size
                total_params+=params["vbias"].size
            params = self.outputLayer.get_params()
            total_params+=params["W"].size
            total_params+=params["hbias"].size
            total_params+=params["vbias"].size
            return total_params

    def __createAD__(self):
        # construct ensemble layer
        for map in self.v:
            params = dA_params(
                n_visible=len(map),
                n_hidden=0,
                lr=self.lr,
                corruption_level=0,
                gracePeriod=0,
                hiddenRatio=self.hr,

                input_precision=self.input_precision,
                quantize=self.quantize,
            )
            self.ensembleLayer.append(dA(params))

        # construct output layer
        params = dA_params(
            len(self.v),
            n_hidden=0,
            lr=self.lr,
            corruption_level=0,
            gracePeriod=0,
            hiddenRatio=self.hr,
            quantize=self.quantize,
            input_precision=self.input_precision,
        )
        self.outputLayer = dA(params)

    def get_params(self):
        return_dict = {"ensemble": []}
        for i in range(len(self.ensembleLayer)):
            return_dict["ensemble"].append(self.ensembleLayer[i].get_params())
        return_dict["output"] = self.outputLayer.get_params()
        return return_dict

    def set_params(self, new_param):
        for i in range(len(new_param["ensemble"])):
            self.ensembleLayer[i].set_params(new_param["ensemble"][i])
        self.outputLayer.set_params(new_param["output"])


# A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
# n: the number of dimensions in the dataset
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
class corClust:
    def __init__(self, n):
        # parameter:
        self.n = n
        # varaibles
        self.c = np.zeros(n)  # linear num of features
        self.c_r = np.zeros(n)  # linear sum of feature residules
        self.c_rs = np.zeros(n)  # linear sum of feature residules
        self.C = np.zeros((n, n))  # partial correlation matrix
        self.N = 0  # number of updates performed

    # x: a numpy vector of length n
    def update(self, x):
        self.N += 1
        self.c += x
        c_rt = x - self.c / self.N
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt, c_rt)

    # creates the current correlation distance matrix between the features
    def corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        C_rs_sqrt[
            C_rs_sqrt == 0
        ] = 1e-100  # this protects against dive by zero erros (occurs when a feature is a constant)
        D = 1 - self.C / C_rs_sqrt  # the correlation distance matrix
        D[
            D < 0
        ] = 0  # small negatives may appear due to the incremental fashion in which we update the mean. Therefore, we 'fix' them
        return D

    # clusters the features together, having no more than maxClust features per cluster
    def cluster(self, maxClust):
        D = self.corrDist()
        Z = linkage(
            D[np.triu_indices(self.n, 1)]
        )  # create a linkage matrix based on the distance matrix
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        map = self.__breakClust__(to_tree(Z), maxClust)
        return map

    # a recursive helper function which breaks down the dendrogram branches until all clusters have no more than maxClust elements
    def __breakClust__(self, dendro, maxClust):
        if (
            dendro.count <= maxClust
        ):  # base case: we found a minimal cluster, so mark it
            return [
                dendro.pre_order()
            ]  # return the origional ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(), maxClust) + self.__breakClust__(
            dendro.get_right(), maxClust
        )


def squeeze_features(fv, precision):
    """rounds features to siginificant figures

    Args:
        fv (array): feature vector.
        precision (int): number of precisions to use.

    Returns:
        array: rounded array of floats.

    """

    return np.around(fv, decimals=precision)


def quantize(x, k):
    n = 2**k - 1
    return np.round(np.multiply(n, x)) / n


def quantize_weights(w, k):
    x = np.tanh(w)
    q = x / np.max(np.abs(x)) * 0.5 + 0.5
    return 2 * quantize(q, k) - 1


class dA_params:
    def __init__(
        self,
        n_visible=5,
        n_hidden=3,
        lr=0.001,
        corruption_level=0.0,
        gracePeriod=10000,
        hiddenRatio=None,
        input_precision=None,
        quantize=None,
    ):
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio
        self.quantize = quantize
        self.input_precision = input_precision
        if quantize:
            self.q_wbit, self.q_abit = quantize


class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(
                np.ceil(self.params.n_visible * self.params.hiddenRatio)
            )

        # for 0-1 normlaization
        self.norm_max = np.ones((self.params.n_visible,)) * -np.Inf
        self.norm_min = np.ones((self.params.n_visible,)) * np.Inf
        self.n = 0

        self.rng = np.random.RandomState(1234)

        a = 1.0 / self.params.n_visible
        self.W = np.array(
            self.rng.uniform(  # initialize W uniformly
                low=-a, high=a, size=(self.params.n_visible, self.params.n_hidden)
            )
        )

        # quantize weights
        if self.params.quantize:
            self.W = quantize_weights(self.W, self.params.q_wbit)

        self.hbias = np.zeros(self.params.n_hidden)  # initialize h bias 0
        self.vbias = np.zeros(self.params.n_visible)  # initialize v bias 0
        # self.W_prime = self.W.T

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(np.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(np.dot(hidden, self.W.T) + self.vbias)

    def train(self, x):
        self.n = self.n + 1

        if self.params.input_precision:
            x = squeeze_features(x, self.params.input_precision)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x

        y = self.get_hidden_values(tilde_x)
        if self.params.quantize:
            y = quantize(y, self.params.q_abit)

        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = np.outer(tilde_x.T, L_h1) + np.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.vbias += self.params.lr * L_vbias

        if self.params.quantize:
            self.W = quantize_weights(self.W, self.params.q_wbit)
            self.hbias = quantize_weights(self.hbias, self.params.q_wbit)
            self.vbias = quantize_weights(self.vbias, self.params.q_wbit)
        # the RMSE reconstruction error during training
        return np.sqrt(np.mean(L_h2**2))

    def reconstruct(self, x):
        y = self.get_hidden_values(x)

        try:
            if self.params.quantize:
                y = quantize(y, self.params.q_abit)
        except AttributeError as e:
            pass

        z = self.get_reconstructed_input(y)
        return z

    def get_params(self):
        params = {"W": self.W, "hbias": self.hbias, "vbias": self.vbias}
        return params

    def set_params(self, new_param):
        self.W = new_param["W"]
        self.hbias = new_param["hbias"]
        self.vbias = new_param["vbias"]

    def execute(self, x):  # returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            try:
                if self.params.input_precision:
                    x = squeeze_features(x, self.params.input_precision)

            except AttributeError as e:
                pass

            z = self.reconstruct(x)
            rmse = np.sqrt(((x - z) ** 2).mean())  # MSE
            return rmse

    def inGrace(self):
        return self.n < self.params.gracePeriod


def pdf(x, mu, sigma):  # normal distribution pdf
    x = (x - mu) / sigma
    return np.exp(-(x**2) / 2) / (np.sqrt(2 * np.pi) * sigma)


def invLogCDF(x, mu, sigma):  # normal distribution cdf
    x = (x - mu) / sigma
    return norm.logcdf(
        -x
    )  # note: we mutiple by -1 after normalization to better get the 1-cdf


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1.0 - x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1.0 - x * x


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1.0 * (x > 0)


class rollmean:
    def __init__(self, k):
        self.winsize = k
        self.window = np.zeros(self.winsize)
        self.pointer = 0

    def apply(self, newval):
        self.window[self.pointer] = newval
        self.pointer = (self.pointer + 1) % self.winsize
        return np.mean(self.window)


# probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * np.power(scale, 2)
#     e = np.exp( - np.power((x - mean), 2) / s )

#     return e / np.square(np.pi * s)


# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
