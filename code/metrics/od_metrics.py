import numpy as np
from scipy import stats


def detection_rate(result_dict):
    return np.mean(result_dict["scores"]> result_dict["threshold"])

def average_metric(result_name):
    def metric(result_dict):
        if result_name not in result_dict.keys() or result_dict[result_name] is None: 
            return np.nan
        return np.nanmean(result_dict[result_name])
    metric.__name__ = "average_"+result_name
    return metric 