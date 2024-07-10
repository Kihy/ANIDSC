import numpy as np

def detection_rate(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.mean(result_dict["score"]> result_dict["threshold"])

def pos_count(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.sum(result_dict["score"]> result_dict["threshold"])

def batch_size(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return len(result_dict["score"])

def median_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanmedian(result_dict["score"])

def median_threshold(result_dict):
    if result_dict["threshold"] is None or np.isnan(result_dict["threshold"]).all():
        return 0
    return np.nanmedian(result_dict["threshold"])