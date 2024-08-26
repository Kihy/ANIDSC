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
    models = ["VAE", "AE", "SLAD", "GOAD", "ICL", "KitNET", "BoxPlot", "Kitsune"]
    node_encoders = ["LinearNodeEncoder", "GCNNodeEncoder", "GATNodeEncoder"]
    dists = ["gaussian", "log-normal", "uniform"]
    cdd_frameworks=["ARCUS","DriftSense"]

    name_str = []

    for components in [models, node_encoders, dists, cdd_frameworks]:
        name = find_name(components, value)
        if name is not None:
            name_str.append(name)

    if len(name_str) == 0:
        return "NA"
    else:
        return "|".join(name_str)


def f_beta(tp, fn, fp, beta=1):
    opbs = 1 + beta**2
    return opbs * tp / (opbs * tp + beta**2 * fn + fp)


def weighted_hmean(pr, re, beta=1):
    opbs = 1 + beta**2
    return opbs * pr * re / (beta**2 * pr + re)


class BasicSummarizer:
    def __init__(self, datasets, fe_name, files, calc_f1=False, col=None):
        self.datasets = datasets
        self.files = files
        self.fe_name = fe_name
        self.calc_f1 = calc_f1
        self.col = col
        if self.col is None:
            self.add_dummy_col=True 
            self.col="protocol"
        else:
            self.add_dummy_col=False
        self.batch_size = 256
        
    def plot_scores(self, df, dataset, file, filename):
        id_vars = ["index", self.col] if self.col else ["index"]
        
        # Sample data if it exceeds 1000 rows
        if len(df) > 1000:
            df = df.sample(1000).sort_index()

        # Define the variables and custom color palette
        value_vars = ["median_score", "median_threshold"]
        custom_palette = {"median_score": "#7FB7BE", "median_threshold": "#F25F5C"}
        
        # Melt the DataFrame
        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                            var_name="variable", value_name="value")

        # Create the Seaborn plot
        g = sns.relplot(
            x="index",
            y="value",
            hue="variable",
            kind="line",
            data=df_melted,
            aspect=2,
            col=self.col,
            palette=custom_palette,
            facet_kws={"sharey": False},
        )
        
        # Function to fill between quartiles
        def fill_quartiles(ax, data):
            ax.fill_between(data["index"], data["lower_quartile_score"],
                            data["upper_quartile_score"], alpha=0.5, color="#7FB7BE")
            ax.fill_between(data["index"], data["soft_min_score"],
                            data["soft_max_score"], alpha=0.2, color="#7FB7BE")

        # Function to add vertical spans based on pool size changes
        def add_pool_size_spans(ax, data):
            if "pool_size" in data.columns:
                segments = data[data['pool_size'] != data['pool_size'].shift()]["index"]
                segments = segments.to_list() + [data["index"].iloc[-1]]
                cmap = plt.get_cmap("viridis")
                colors = [cmap(i / (len(segments) - 1)) for i in range(len(segments) - 1)]
                for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
                    ax.axvspan(start, end, facecolor=colors[i], alpha=0.2)

        # Apply fill and spans for each subplot
        if self.col is None:
            ax = g.axes[0, 0]
            fill_quartiles(ax, df)
            add_pool_size_spans(ax, df)
        else:
            for ax, col_name in zip(g.axes.flat, g.col_names):
                col_data = df[df[self.col] == col_name]
                fill_quartiles(ax, col_data)
                add_pool_size_spans(ax, col_data)

        # Set y-axis to symmetrical logarithmic scale
        g.set(yscale="symlog")
        
        # Save the plot to the specified path
        path = Path(f"{dataset}/{self.fe_name}/plots/{file}/{filename[:-4]}_scores.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        g.savefig(path)
        print(f"Score plot saved to {path}")
        plt.close()


    def plot_metrics(self, df, dataset, file, filename):
        if self.col is not None:
            id_vars = ["index", self.col]
        else:
            id_vars = ["index"]
            
        value_vars=["detection_rate"]
        if "pool_size" in df.columns:
            value_vars+=["pool_size", "drift_level"]
        
        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="variable",
            value_name="value",
        )

        g = sns.relplot(
            x="index",
            y="value",
            row="variable",
            kind="line",
            data=df_melted,
            # height=1.5,
            aspect=2,
            col=self.col,
            # s=2,
            # linewidth=0,
            facet_kws={"sharey": False},
        )

        path = Path(f"{dataset}/{self.fe_name}/plots/{file}/{filename[:-4]}_metrics.png")
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
                self.plot_metrics(df, dataset, file, filename)

    def calc_stats(self, group):
        pos_benign = group[group["file"].str.startswith("benign")]["pos_count_sum"].sum()
        total_benign = group[group["file"].str.startswith("benign")]["batch_size_sum"].sum()
        
        
        pos_malicious = group[group["file"].str.startswith("malicious")][
            "pos_count_sum"
        ].sum()
        total_malicious = group[group["file"].str.startswith("malicious")][
            "batch_size_sum"
        ].sum()

        mean_mal = np.mean(group[group["file"].str.startswith("malicious")]["acc"])
        mean_ben = np.mean(group[group["file"].str.startswith("benign")]["acc"])
        
        kl_divs=[]
        #calculate kl divergence
        for name, g in group.groupby(self.col):
            benign_mean=g[g["file"].str.startswith("benign")]["diff_magnitude_mean"].to_numpy()
            benign_std=g[g["file"].str.startswith("benign")]["diff_magnitude_std"].to_numpy()
            
            mal_means=g[g["file"].str.startswith("malicious")]["diff_magnitude_mean"].to_numpy()
            mal_stds=g[g["file"].str.startswith("malicious")]["diff_magnitude_std"].to_numpy()
            # mal_stds=np.nan_to_num(mal_stds, nan=1.)
            
            #benign is P mal is Q
            kl_div=np.log(mal_stds/benign_std)+(benign_std**2+(benign_mean-mal_means)**2)/(2*mal_stds**2)-1/2
            kl_divs.append(np.nanmedian(kl_div))
            
        
        
        return (
            pos_benign,
            total_benign,
            pos_malicious,
            total_malicious,
            mean_mal,
            mean_ben,
            kl_divs
        )

    def gen_summary(self):
        for dataset in self.datasets:
            dataframes = []

            for file in self.files:
                dir = f"{dataset}/{self.fe_name}/results/{file}"

                for filename in os.listdir(dir):
                    f = os.path.join(dir, filename)
                    df = pd.read_csv(f)
                    print(f"processing {f}")
                    
                    # measure difference magnitude
                    prev_df=df.shift(1)
                    eps=df["median_score"]-prev_df["median_score"]
                    min_score=np.minimum(df["soft_min_score"], prev_df["soft_min_score"])
                    max_score=np.maximum(df["soft_max_score"], prev_df["soft_max_score"])
                    
                    
                    # Add the results as a new column in the DataFrame
                    #(1/df["batch_size"]+1/prev_df["batch_size"])*
                    
                    df["diff_magnitude"] = 2*eps**2/((max_score-min_score)**2)
                    
                    
                    agg_funcs = {
                    "time": "mean",
                    "detection_rate": "mean",
                    "median_score": "mean",
                    "median_threshold": "mean",
                    "pos_count": "sum",
                    "batch_size": "sum",
                    "diff_magnitude":["mean", "std"],
                }
                    if "pool_size" in df.columns:
                        agg_funcs.update({"pool_size":"mean", "drift_level":"mean"})
                    
                    df = df.replace([np.inf, -np.inf], np.nan)

                    if self.add_dummy_col:
                        # add dummy column 
                        df["protocol"]="All"
                        
                    df_summary = df.groupby(self.col).agg(agg_funcs)
                    df_summary.reset_index(inplace=True)
                    # else:
                    #     df_summary = pd.DataFrame(df.agg(agg_funcs)).T
                        

                    df_summary.columns = ['_'.join(n for n in col if n.strip()) for col in df_summary.columns.values]
                    df_summary["dataset"] = dataset
                    df_summary["file"] = file
                    df_summary["pipeline"] = filename
                    

                    dataframes.append(df_summary)

            all_df = pd.concat(dataframes)
            all_df["pipeline"] = all_df["pipeline"].apply(name_map_func)
            all_df["dr"] = all_df["pos_count_sum"] / all_df["batch_size_sum"]
            all_df["acc"] = all_df["dr"].where(
                all_df["file"].str.startswith("malicious"), other=1 - all_df["dr"]
            )

            path = Path(f"{dataset}/{self.fe_name}/plots")
            path.mkdir(exist_ok=True, parents=True)

            if self.calc_f1:
                data = defaultdict(list)

                if not self.add_dummy_col:
                    # calculate average across all protocols
                    for pipeline, group in all_df.groupby("pipeline"):
                        (
                            pos_benign,
                            total_benign,
                            pos_malicious,
                            total_malicious,
                            mean_mal,
                            mean_ben,
                            kl_divs
                        ) = self.calc_stats(group)
                        
                        f1 = f_beta(
                            pos_malicious,
                            total_malicious - pos_malicious,
                            pos_benign,
                            beta=1,
                        )

                        data["protocol"].append("All")
                        data["pos_mal"].append(pos_malicious)
                        data["total_mal"].append(total_malicious)
                        data["pos_ben"].append(pos_benign)
                        data["total_ben"].append(total_benign)

                        data["pipelines"].append(pipeline)
                        data["f1_scores"].append(f1)

                        data["mean_mal"].append(mean_mal)
                        data["mean_ben"].append(mean_ben)
                        
                        data["mean_kl_div"].append(np.median(kl_divs)) 
                        data["max_kl_div"].append(np.max(kl_divs)) 
                        data["min_kl_div"].append(np.min(kl_divs)) 
                        
                        
                        data["hmean_1_acc"].append(
                            weighted_hmean(mean_ben, mean_mal, beta=1)
                        )
                        data["hmean_0.5_acc"].append(
                            weighted_hmean(mean_ben, mean_mal, beta=0.5)
                        )
                        data["hmean_2_acc"].append(
                            weighted_hmean(mean_ben, mean_mal, beta=2)
                        )
                        data["time"].append(group["time_mean"].median())
                        
                
                for (pipeline, col), group in all_df.groupby(
                    ["pipeline", self.col]
                ):
                    (
                        pos_benign,
                        total_benign,
                        pos_malicious,
                        total_malicious,
                        mean_mal,
                        mean_ben,
                        kl_divs
                    ) = self.calc_stats(group)
                    f1 = f_beta(
                        pos_malicious,
                        total_malicious - pos_malicious,
                        pos_benign,
                        beta=1,
                    )

                    data["protocol"].append(col)
                    data["pos_mal"].append(pos_malicious)
                    data["total_mal"].append(total_malicious)
                    data["pos_ben"].append(pos_benign)
                    data["total_ben"].append(total_benign)

                    data["pipelines"].append(pipeline)
                    data["f1_scores"].append(f1)

                    data["mean_mal"].append(mean_mal)
                    data["mean_ben"].append(mean_ben)
                    
                    data["mean_kl_div"].append(np.mean(kl_divs)) 
                    data["max_kl_div"].append(np.max(kl_divs)) 
                    data["min_kl_div"].append(np.min(kl_divs)) 
                    
                    data["hmean_1_acc"].append(
                        weighted_hmean(mean_ben, mean_mal, beta=1)
                    )
                    data["hmean_0.5_acc"].append(
                        weighted_hmean(mean_ben, mean_mal, beta=0.5)
                    )
                    data["hmean_2_acc"].append(
                        weighted_hmean(mean_ben, mean_mal, beta=2)
                    )
                    data["time"].append(group["time_mean"].median())

                summary_df = pd.DataFrame(data)

                g = sns.catplot(
                    kind="bar",
                    data=summary_df,
                    x="pipelines",
                    y="hmean_0.5_acc",
                    col=self.col,
                )
                # Rotate x-axis labels
                g.set_xticklabels(rotation=90)

                # Adjust layout to prevent overlap
                plt.tight_layout()
                g.savefig(f"{path}/f1_summary.png")
                plt.close()
                print(f"F1 plot saved to {path}/f1_summary.png")
                summary_df.to_csv(
                    f"{dataset}/{self.fe_name}/results/summary.csv", index=False
                )
                print(
                    f"resulting csv saved to {dataset}/{self.fe_name}/results/summary.csv"
                )

            # g = sns.catplot(
            #     data=all_df, kind="bar", x="pipeline", y="detection_rate", row="file", col=self.col
            # )
            # g.savefig(f"{path}/dr_summary.png")
            # plt.close()
            # print(f"DR plot saved to {path}/dr_summary.png")

            g = sns.catplot(
                data=all_df, kind="bar", x="pipeline", y="time_mean", col=self.col
            )
            g.set_xticklabels(rotation=90)
            g.savefig(f"{path}/time_summary.png")
            plt.close()
            print(f"time plot saved to {path}/time_summary.png")

            all_df.to_csv(f"{dataset}/{self.fe_name}/results/overview.csv", index=False)
            print(
                f"resulting csv saved to {dataset}/{self.fe_name}/results/overview.csv"
            )

        return all_df
