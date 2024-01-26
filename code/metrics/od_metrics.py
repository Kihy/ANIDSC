import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats

def mean_dr(model_output):
    total=0
    positive=0
    t=model_output["threshold"]
    
    dr=[]
    for data_type in["benign","malicious","adversarial"]:
        for dataset_name, score in model_output[data_type].items():
            total+=score.shape[0]
            positive+=np.sum(score>t)
            
        detection_rate=positive/total
        if data_type=="benign":
            detection_rate-=detection_rate
        dr.append(detection_rate)
        
    return stats.hmean(dr)

def plot_scores(model_output):
    t=model_output["threshold"]
    num_plots=len(model_output["benign"])+len(model_output["malicious"])+len(model_output["adversarial"])
    fig, ax = plt.subplots(num_plots, figsize=(6,num_plots*2))
    count=0
    
    for dataset_type in ["benign","malicious","adversarial"]:
        for dataset_name, score in model_output[dataset_type].items():
            ax[count].scatter(range(len(score)), score, s=5)
            ax[count].set_title(dataset_name)
            ax[count].axhline(t)
            count+=1
    fig.tight_layout()
    save_path="test.png"
    fig.savefig(save_path)
    return save_path
    