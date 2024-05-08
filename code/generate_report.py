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
    
    records=pd.DataFrame.from_dict({(i,j): results[i][j]
                           for i in results.keys() 
                           for j in results[i].keys()},
                       orient='index')
    
    records=records.reset_index()
    records=records.replace("benign/whole_week","benign/all_device/whole_week")
    records=records.rename(columns={"level_0":"model","level_1":"file"})

    records[["label","device","name"]]=records["file"].str.split("/", expand=True)
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
    
    df["relative_batch_num"]=df["batch_num"]-df.iloc[0]["batch_num"]
    
    x_str="relative_batch_num"
    
    if "model_idx" not in df:
        df["model_idx"]=1
        
    df["model_idx"]=df["model_idx"].astype(str)
    
    df=df.melt(id_vars=[x_str,"protocol","model_idx"], value_vars=["score","threshold"])

    g = sns.relplot(
    data=df,
    kind="line",
    x=x_str,
    y="value",
    hue="variable",
    col="protocol",
    row="model_idx",
    aspect=3,
    errorbar=None,
    height=3,
    # facet_kws=dict(legend_out=False)
    )

    
    # g.set(xlim=(0, 4000), ylim=(1,10000)) 
    g.set_titles("")
    g.set_ylabels("Anomaly Score")
    g.set_xlabels("Batch Number")
    g._legend.set_title("")
    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, .9))

    if "ARCUS" not in model_name:
        for ax in g.axes.flat:
            ax.set_yscale('log')
             
    g.tight_layout()
    fig_path=f"../../datasets/{dataset_name}/{fe_name}/plots/{file_name}/{model_name}_drift.pdf"
    g.savefig(fig_path)
    print(f"concept drift plot saved at {fig_path}")
    plt.close()

if __name__=="__main__":
    # get_results("UQ_IoT_IDS21/AfterImage")
    get_results("UQ_IoT_IDS21/AfterImageGraph_multi_layer")
    # compare_spikes("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "AE")
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "AE")
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImage", "malicious/Smart_TV/ACK_Flooding", "AE")
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "benign/whole_week", "MultiLayerOCDModel-100-50-GCNNodeEncoder-gaussian-AE")

    # plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "malicious/Smart_TV/ACK_Flooding", "MultiLayerGNNIDS-gaussian")
    # plot_concept_drift("FakeGraphData", "SyntheticFeatureExtractor", "benign/mean_std_drift_4", "MultiLayerGNNIDS")
    