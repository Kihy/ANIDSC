import numpy as np
from scipy import stats


def detection_rate(scores, threshold):
    return np.mean(scores> threshold)

def average_score(scores, threshold):
    return np.nanmean(scores)

def average_threshold(scores, threshold):
    return np.nanmean(threshold)