import numpy as np
from scipy import stats


def detection_rate(scores, threshold):
    return np.mean(scores>threshold)

def count(scores, threshold):
    return scores.shape[0]