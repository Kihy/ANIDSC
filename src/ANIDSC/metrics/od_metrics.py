import numpy as np

def detection_rate(result_dict):
    return np.mean(result_dict["score"]> result_dict["threshold"])

def average_score(result_dict):
    return np.nanmean(result_dict["score"])

def average_threshold(result_dict):
    return np.nanmean(result_dict["threshold"])