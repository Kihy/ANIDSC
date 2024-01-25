import numpy as np 
import matplotlib.pyplot as plt

def special_f1(model_output):
    total_benign=0
    positive_benign=0
    total_malicious=0
    positive_malicious=0
    t=model_output["threshold"]
    for dataset_name, score in model_output.items():
        if dataset_name=="threshold":
            continue 
        if dataset_name.endswith("test"):
            total_benign+=score.shape[0]
            positive_benign+=np.sum(score>t)
        else:
            total_malicious+=score.shape[0]
            positive_malicious+=np.sum(score>t)
    
    BDR=1.-positive_benign/total_benign
    MDR=positive_malicious/total_malicious
    return 2./(1./BDR+1./MDR)

def plot_scores(model_output):
    t=model_output["threshold"]
    num_plots=len(model_output)-1
    fig, ax = plt.subplots(num_plots, figsize=(6,num_plots*2))
    count=0
    for dataset_name, score in model_output.items():
        if dataset_name=="threshold":
            continue 
        ax[count].scatter(range(len(score)), score, s=5)
        ax[count].set_title(dataset_name)
        ax[count].axhline(t)
        count+=1
    fig.tight_layout()
    return fig, ax
    