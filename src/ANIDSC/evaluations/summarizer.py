from collections import defaultdict
from itertools import product
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import hmean

def find_name(names, value, abbrev=False):
    for i in names:
        if i in value:
            if abbrev:
                return i[:3]
            else:
                return i
    return None
    
def name_map_func(value):
    models = ["VAE", "AE", "SLAD", "GOAD", "ICL", "KitNET", "BoxPlot", "ARCUS"]
    node_encoders=["LinearNodeEncoder","GCNNodeEncoder","GATNodeEncoder"]
    dists=["gaussian","lognormal","uniform"]
        
    name_str=[]
    
    
    name=find_name(models, value)
    if name is not None:
        name_str.append(name)
        
    name=find_name(node_encoders, value, True)
    if name is not None:
        name_str.append(name)    
        
    name=find_name(dists, value, True)
    if name is not None:
        name_str.append(name)
    
    if "ConceptDriftWrapper" in value:
        name_str.append("CDD")
        
    if len(name_str)==0:
        return "NA"
    else:
        return "-".join(name_str)

def f_beta(tp, fn, fp, beta=1):
    opbs=(1+beta**2)
    return opbs*tp/(opbs*tp+beta**2*fn+fp)

def weighted_hmean(pr, re, beta=1):
    opbs=(1+beta**2)
    return opbs*pr*re/(beta**2*pr+re)

class BasicSummarizer:
    def __init__(self, datasets, fe_name, files, calc_f1=False, col=None):
        self.datasets = datasets
        self.files = files
        self.fe_name = fe_name
        self.calc_f1 = calc_f1
        self.col=col
        self.batch_size = 256

        self.agg_funcs = {
            "time": "mean",
            "detection_rate": "mean",
            "median_score": "mean",
            "median_threshold": "mean",
            "pos_count": "sum",
            "batch_size": "sum",
        }

    def plot_scores(self, df, dataset, file, filename):
        if self.col is not None:
            id_vars=["index",self.col]
        else:
            id_vars=["index"]
            
        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=["median_score", "median_threshold"],
            var_name="variable",
            value_name="value",
        )
        
        g = sns.relplot(
            x="index",
            y="value",
            hue="variable",
            kind="scatter",
            data=df_melted,
            # height=1.5,
            col=self.col,
            s=2,
            linewidth=0,
        )
        g.set(yscale="log")
        path = Path(f"{dataset}/{self.fe_name}/plots/{file}/{filename[:-4]}_scores.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        g.savefig(path)
        print(f"score plot saved to {path}")
        plt.close()

    def plot_dr(self, df, dataset, file, filename):
        g = sns.relplot(
            x="index",
            y="detection_rate",
            kind="scatter",
            data=df,
            # height=1.5,
            col=self.col,
            s=2,
            linewidth=0,
        )

        path = Path(f"{dataset}/{self.fe_name}/plots/{file}/{filename[:-4]}_dr.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        g.savefig(path)
        plt.close()
        print(f"score plot saved to {path}")

    def plots(self):
        for dataset, file in product(self.datasets, self.files):
            dir = f"{dataset}/{self.fe_name}/results/{file}"
            for filename in os.listdir(dir):
                f = os.path.join(dir, filename)
                df = pd.read_csv(f)

                df.reset_index(inplace=True)

                self.plot_scores(df, dataset, file, filename)
                self.plot_dr(df, dataset, file, filename)

    def calc_stats(self, group):
        pos_benign = group[group["file"].str.startswith("benign")][
                        "pos_count"
                    ].sum()
        total_benign = group[group["file"].str.startswith("benign")][
            "batch_size"
        ].sum()

        pos_malicious = group[group["file"].str.startswith("malicious")][
            "pos_count"
        ].sum()
        total_malicious = group[group["file"].str.startswith("malicious")][
            "batch_size"
        ].sum()

        
        mean_mal=np.mean(group[group["file"].str.startswith("malicious")]['acc'])
        mean_ben=np.mean(group[group["file"].str.startswith("benign")]['acc'])
        return pos_benign, total_benign, pos_malicious, total_malicious, mean_mal, mean_ben
                    

    def gen_summary(self):
        for dataset in self.datasets:
            dataframes = []

            for file in self.files:
                dir = f"{dataset}/{self.fe_name}/results/{file}"

                for filename in os.listdir(dir):
                    f = os.path.join(dir, filename)
                    df = pd.read_csv(f)
                    df = df.replace([np.inf, -np.inf], np.nan)

                    if self.col is not None:
                        df_summary=df.groupby(self.col).agg(self.agg_funcs)
                        df_summary.reset_index(inplace=True)
                    else:
                        df_summary = df.agg(self.agg_funcs)
                        
                    df_summary["dataset"] = dataset
                    df_summary["file"] = file
                    df_summary["pipeline"] = filename

                    dataframes.append(df_summary)

            all_df = pd.concat(dataframes, ignore_index=True)
            all_df["pipeline"] = all_df["pipeline"].apply(name_map_func)
            all_df['dr']=all_df['pos_count']/all_df['batch_size']
            all_df['acc']=all_df['dr'].where(all_df['file'].str.startswith("malicious"), other=1-all_df['dr'])
                    
            
            path = Path(f"{dataset}/{self.fe_name}/plots")
            path.mkdir(exist_ok=True, parents=True)

            if self.calc_f1:
                data = defaultdict(list)
                
                for pipeline, group in all_df.groupby("pipeline"):
                    pos_benign, total_benign, pos_malicious, total_malicious, mean_mal, mean_ben=self.calc_stats(group)
                    f1 = f_beta(pos_malicious, total_malicious-pos_malicious, pos_benign, beta=1)
                    
                    data["protocol"].append("All")
                    data['pos_mal'].append(pos_malicious)
                    data['total_mal'].append(total_malicious)
                    data['pos_ben'].append(pos_benign)
                    data["total_ben"].append(total_benign)
                    
                    data['pipelines'].append(pipeline)
                    data['f1_scores'].append(f1)
                    
                    data['mean_mal'].append(mean_mal)
                    data['mean_ben'].append(mean_ben)
                    data['hmean_1_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=1))
                    data['hmean_0.5_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=0.5))
                    data['hmean_2_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=2))
                
                if self.col is not None:
                    for (pipeline, col), group in all_df.groupby(["pipeline",self.col]):
                        pos_benign, total_benign, pos_malicious, total_malicious, mean_mal, mean_ben=self.calc_stats(group)
                        f1 = f_beta(pos_malicious, total_malicious-pos_malicious, pos_benign, beta=1)
                        
                        data["protocol"].append(col)
                        data['pos_mal'].append(pos_malicious)
                        data['total_mal'].append(total_malicious)
                        data['pos_ben'].append(pos_benign)
                        data["total_ben"].append(total_benign)
                        
                        data['pipelines'].append(pipeline)
                        data['f1_scores'].append(f1)
                        
                        data['mean_mal'].append(mean_mal)
                        data['mean_ben'].append(mean_ben)
                        data['hmean_1_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=1))
                        data['hmean_0.5_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=0.5))
                        data['hmean_2_acc'].append(weighted_hmean(mean_ben, mean_mal, beta=2))
                
                
                summary_df= pd.DataFrame(data)

                g = sns.catplot(kind="bar", data=summary_df, x="pipelines", y="hmean_2_acc", col=self.col)
                # Rotate x-axis labels
                g.set_xticklabels(rotation=90)

                # Adjust layout to prevent overlap
                plt.tight_layout()
                g.savefig(f"{path}/f1_summary.png")
                plt.close()
                print(f"F1 plot saved to {path}/f1_summary.png")
                summary_df.to_csv(f"{dataset}/{self.fe_name}/results/summary.csv", index=False)
                print(
                f"resulting csv saved to {dataset}/{self.fe_name}/results/summary.csv"
                )

            # g = sns.catplot(
            #     data=all_df, kind="bar", x="pipeline", y="detection_rate", row="file", col=self.col
            # )
            # g.savefig(f"{path}/dr_summary.png")
            # plt.close()
            # print(f"DR plot saved to {path}/dr_summary.png")

            g = sns.catplot(data=all_df, kind="bar", x="pipeline", y="time", col=self.col)
            g.set_xticklabels(rotation=90)
            g.savefig(f"{path}/time_summary.png")
            plt.close()
            print(f"time plot saved to {path}/time_summary.png")

            all_df.to_csv(f"{dataset}/{self.fe_name}/results/overview.csv", index=False)
            print(
                f"resulting csv saved to {dataset}/{self.fe_name}/results/overview.csv"
            )

        return all_df
