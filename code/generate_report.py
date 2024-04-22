import json 
from pathlib import Path
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

def get_results(dataset_name):
    results_path=Path(f"../../datasets/{dataset_name}/results.json")
            
    with open(str(results_path)) as f:
        results=json.load(f)
    
    records=[]
    for model, contents in results.items():
        benign_dr=[]
        mal_dr=[]
        for file, metrics in contents.items():
            if file.startswith("benign"):
                benign_dr.append(metrics["detection_rate"])
            elif file.startswith("malicious"):
                mal_dr.append(metrics["detection_rate"])
        records.append([model, np.mean(benign_dr), np.mean(mal_dr)])
    
    records=pd.DataFrame(records,columns=["Model","Ben DR", "Mal DR"])
    print(records)

def plot_concept_drift(dataset_name, fe_name, file_name, model_name):
    results_file=Path(f"../../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}.csv")
    df=pd.read_csv(str(results_file))
    
    df["model_index"]=df["model_index"].astype('string')
    g = sns.relplot(
    data=df,
    x="index",
    y="loss",
    s=3,
    # hue="model_index",
    col="protocol",
    hue=f"drift_level",
    edgecolor=None,
    )
    g.axes[0,0].scatter(x=df[df["protocol"]=="UDP"]["index"], y=df[df["protocol"]=="UDP"]["threshold"], s=1)
    # g.axes[0,1].scatter(x=df[df["protocol"]=="TCP"]["index"], y=df[df["protocol"]=="TCP"]["threshold"], s=3)
    
    g.tight_layout()
    g.set(yscale="log")
    
    g.savefig(f"../../datasets/{dataset_name}/{fe_name}/plots/{file_name}/{model_name}_cd.png")
    print("cd plot saved at "+f"../../datasets/{dataset_name}/{fe_name}/plots/{file_name}/{model_name}_cd.png")

if __name__=="__main__":
    # get_results("UQ_IoT_IDS21")
    plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "benign/whole_week", "MultiLayerGNNIDS")
    plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "malicious/Smart_TV/ACK_Flooding", "MultiLayerGNNIDS")
    # plot_concept_drift("FakeGraphData", "SyntheticFeatureExtractor", "benign/mean_std_drift_4", "MultiLayerGNNIDS")
    