import json 
from pathlib import Path
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict
from utils import *

plt.style.use(['science','ieee'])

def get_results(dataset_name):
    results_path=Path(f"../../datasets/{dataset_name}/results.json")
            
    with open(str(results_path)) as f:
        results=json.load(f)
    
    records=[]
    for model, contents in results.items():
        dr_dict=defaultdict(list)
        for file, metrics in contents.items():
            name=file.split("/")[-1]
            dr_dict[name].append(metrics["detection_rate"])
        
        records.append([model]+[np.mean(v) for k, v in sorted(dr_dict.items())])
    
    records=pd.DataFrame(records,columns=["Model"]+[k for k, v in sorted(dr_dict.items())])
    print(records.to_csv())


def compare_spikes(dataset_name, fe_name, file_name, model_name):
    results_file=Path(f"../../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}_raw_scores.csv")
    df=pd.read_csv(str(results_file),header=None)
    
    meta_file=Path(f"../../datasets/{dataset_name}/{fe_name}/{file_name}_meta.csv")
    meta_df=pd.read_csv(str(meta_file))
    
    scores=df.to_numpy()
    
    
    od_idx=np.where(scores[500000:]>calc_quantile(scores[500000:],0.999))[0]+500000
    
    all_df=meta_df.iloc[od_idx]
    all_df=all_df.drop(columns=["time_stamp","scr_ip","dst_ip","packet_size"])
    
    print(all_df.value_counts(sort=True)[:20].to_csv())
    
def plot_concept_drift(dataset_name, fe_name, file_name, model_name):
    results_file=Path(f"../../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}.csv")
    df=pd.read_csv(str(results_file))
    
    
    
    df["model_idx"]=df["model_idx"].astype('string')
    g = sns.relplot(
    data=df,
    x="index",
    y="score",
    s=3,
    hue="model_idx",
    col="protocol",
    # hue=f"drift_level",
    edgecolor=None,
    )
    for i, protocol in enumerate(["UDP","TCP","ARP","ICMP","Other"]):
        g.axes[0,i].scatter(x=df[df["protocol"]==protocol]["index"], y=df[df["protocol"]==protocol]["threshold"], s=1)
 
    # g.axes[0,0].scatter(x=df["index"], y=df["threshold"], s=3)
    
    g.tight_layout()
    # g.set(yscale="log")
    
    fig_path=f"../../datasets/{dataset_name}/{fe_name}/plots/{file_name}/{model_name}_drift.png"
    g.savefig(fig_path)
    print(f"concept drift plot saved at {fig_path}")
    plt.close()

if __name__=="__main__":
    # get_results("UQ_IoT_IDS21/AfterImage")
    # compare_spikes("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "OnlineCDModel-ICL-100-50")
    plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "benign/whole_week", "MultiLayerOCDModel-100-50-GCNNodeEncoder-gaussian-VAE")
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "malicious/Smart_TV/ACK_Flooding", "MultiLayerGNNIDS-gaussian")
    # plot_concept_drift("FakeGraphData", "SyntheticFeatureExtractor", "benign/mean_std_drift_4", "MultiLayerGNNIDS")
    