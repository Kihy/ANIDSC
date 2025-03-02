import numpy as np
from scipy.special import erfinv



def normal_quantile(x,p):
    mean=np.mean(x) 
    std=np.std(x)
    quantile=mean+np.sqrt(2)*std*erfinv(2*p-1)
    return quantile 

def log_normal_quantile(x,p):    
    eps=1e-6
    x=np.log(np.array(x)+eps)
    quantile=normal_quantile(x,p)
    return np.exp(quantile)

def quantile(x,p):
    return np.quantile(x,p)