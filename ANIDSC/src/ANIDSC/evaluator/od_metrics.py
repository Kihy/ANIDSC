import numpy as np

def detection_rate(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.mean(result_dict["score"]> result_dict["threshold"])

def pos_count(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.sum(result_dict["score"]> result_dict["threshold"])

def pos_idx(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return []
    return np.where(result_dict["score"]> result_dict["threshold"])[0]
        
def batch_size(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return len(result_dict["score"])

def median_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanmedian(result_dict["score"])

def lower_quartile_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanpercentile(result_dict["score"],25)

def upper_quartile_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanpercentile(result_dict["score"],75)

def soft_min_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanpercentile(result_dict["score"],1)

def soft_max_score(result_dict):
    if result_dict["score"] is None or np.isnan(result_dict["score"]).all():
        return 0
    return np.nanpercentile(result_dict["score"],99)

def median_threshold(result_dict):
    if result_dict["threshold"] is None or np.isnan(result_dict["threshold"]).all():
        return 0
    return np.nanmedian(result_dict["threshold"])

def pool_size(result_dict):
    return result_dict["num_model"]

def drift_level(result_dict):
    return result_dict["drift_level"]