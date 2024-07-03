from itertools import product
import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def name_map_func(value):
    models = ["VAE", "AE", "SLAD", "GOAD", "ICL", "KitNET"]
    for i in models:
        if i in value:
            return i
    return "NA"


class AfterImageSummarizer:
    def __init__(self, datasets, files, calc_f1=False):
        self.datasets = datasets
        self.files = files
        self.calc_f1 = calc_f1
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
        df_melted = df.melt(
            id_vars=["index"],
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
            s=2,
            linewidth=0,
        )
        g.set(yscale="log")
        path = Path(f"{dataset}/AfterImage/plots/{file}/{filename[:-4]}_scores.png")
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
            s=2,
            linewidth=0,
        )

        path = Path(f"{dataset}/AfterImage/plots/{file}/{filename[:-4]}_dr.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        g.savefig(path)
        plt.close()
        print(f"score plot saved to {path}")

    def plots(self):
        for dataset, file in product(self.datasets, self.files):
            dir = f"{dataset}/AfterImage/results/{file}"
            for filename in os.listdir(dir):
                f = os.path.join(dir, filename)
                df = pd.read_csv(f)

                df.reset_index(inplace=True)

                self.plot_scores(df, dataset, file, filename)
                self.plot_dr(df, dataset, file, filename)

    def gen_summary(self):
        for dataset in self.datasets:
            dataframes = []

            for file in self.files:
                dir = f"{dataset}/AfterImage/results/{file}"

                for filename in os.listdir(dir):
                    f = os.path.join(dir, filename)
                    df = pd.read_csv(f)
                    df = df.replace([np.inf, -np.inf], np.nan)

                    df_summary = df.agg(self.agg_funcs)
                    df_summary["dataset"] = dataset
                    df_summary["file"] = file
                    df_summary["pipeline"] = filename

                    dataframes.append(df_summary)

            all_df = pd.DataFrame(dataframes)
            all_df["pipeline"] = all_df["pipeline"].apply(name_map_func)

            path = Path(f"{dataset}/AfterImage/plots")
            path.mkdir(exist_ok=True, parents=True)

            if self.calc_f1:
                pipelines=[]
                f1_scores=[]
                for pipeline, group in all_df.groupby('pipeline'):
                    pos_benign=group[group['file'].str.startswith('benign')]['pos_count'].sum()
                    total_benign=group[group['file'].str.startswith('benign')]['batch_size'].sum()
                    
                    pos_malicious=group[group['file'].str.startswith('malicious')]['pos_count'].sum()
                    total_malicious=group[group['file'].str.startswith('malicious')]['batch_size'].sum()
                    
                    Rd_Xm=pos_malicious/total_malicious
                    Rd_Xb=pos_benign/total_benign
                    
                    f1=(2*Rd_Xm)/(1+Rd_Xm+Rd_Xb)
                    
                    pipelines.append(pipeline)
                    f1_scores.append(f1)
                g = sns.catplot(kind="bar", x=pipelines, y=f1_scores)
                g.savefig(f"{path}/f1_summary.png")
                print(f"F1 plot saved to {path}")

            
            g = sns.catplot(
                data=all_df, kind="bar", x="pipeline", y="detection_rate", col="file"
            )
            g.savefig(f"{path}/dr_summary.png")
            print(f"DR plot saved to {path}")

            g = sns.catplot(data=all_df, kind="bar", x="pipeline", y="time", col="file")
            g.savefig(f"{path}/time_summary.png")
            print(f"time plot saved to {path}")

        return all_df
