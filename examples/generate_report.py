import json
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict
from ANIDSC.utils import calc_quantile
from matplotlib.ticker import NullFormatter

from itertools import product

plt.style.use(["science", "ieee"])


def get_results(dataset_name, gnn=False):
    results_path = Path(f"../datasets/{dataset_name}/results.json")

    with open(str(results_path)) as f:
        results = json.load(f)

    records = pd.DataFrame.from_dict(
        {(i, j): results[i][j] for i in results.keys() for j in results[i].keys()},
        orient="index",
    )

    records = records.reset_index()
    records = records.replace("benign/whole_week", "benign/all_device/whole_week")
    records = records.rename(columns={"level_0": "model", "level_1": "file"})

    # records[["label", "device", "name"]] = records["file"].str.split("/", expand=True)

    if gnn:
        records["model"] = records["model"].str.replace(
            "log-normal", "log_normal", regex=False
        )
        records[["model", "GNN", "dist", "od_model"]] = records["model"].str.rsplit(
            "-", n=3, expand=True
        )
        records["pps"] = records["count"] / records["time"]
        # records = records.drop(["model", "file"], axis=1)
        records.fillna(0, inplace=True)
    print(records.to_csv())


def compare_spikes(dataset_name, fe_name, file_name, model_name):
    results_file = Path(
        f"../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}_raw_scores.csv"
    )
    df = pd.read_csv(str(results_file), header=None)

    meta_file = Path(f"../datasets/{dataset_name}/{fe_name}/{file_name}_meta.csv")
    meta_df = pd.read_csv(str(meta_file))

    scores = df.to_numpy()

    od_idx = (
        np.where(scores[500000:] > calc_quantile(scores[500000:], 0.999))[0] + 500000
    )

    all_df = meta_df.iloc[od_idx]
    all_df = all_df.drop(columns=["time_stamp", "scr_ip", "dst_ip", "packet_size"])

    print(all_df.value_counts(sort=True)[:20].to_csv())


def find_minimum_distances(A, B):
    min_distances = []
    j = 0  # Pointer for array B
    B = np.unique(B)
    # Iterate over each element in array A
    for a in A:
        # Move pointer j to the position where elements in B are closest to a
        while j < len(B) - 1 and abs(B[j + 1] - a) < abs(B[j] - a):
            j += 1
        # Append the minimal distance found for this element
        min_distances.append(abs(B[j] - a))

    return np.array(min_distances)


def plot_concept_drift(
    dataset_name,
    fe_name,
    file_name,
    model_name,
    protocols=None,
    drift_idx=None,
    xrange=None,
    yrange=None,
    col="protocol",
    row=None,
    draw_ticks=True,
    draw_legend=True,
    draw_title=True,
    titles=None,
):
    
    dfs=[]
    for f, m in product(file_name, model_name):
        results_file = Path(
            f"../datasets/{dataset_name}/{fe_name}/outputs/{f}/{m}.csv"
        )
        df = pd.read_csv(str(results_file))
        df["file"]=f.split("/")[0]
        df["model"]=m
        df["relative_batch_num"] = df["batch_num"] - df.iloc[0]["batch_num"]
        dfs.append(df)
    
    df=pd.concat(dfs)
    
    x_str = "relative_batch_num"

    if "model_idx" not in df:
        df["model_idx"] = 1

    df["model_idx"] = df["model_idx"].astype(str)

    if protocols is not None:
        df = df[df["protocol"].isin(protocols)]

    if draw_legend:
        hue = "model_idx"
    else:
        hue = None

    if len(dfs)==1 and len(protocols)==1:
        aspect=2 
    else:
        aspect=1.2
    
    aspect=1.5
    
        
    g = sns.relplot(
        data=df,
        kind="scatter",
        x=x_str,
        y="score",
        hue=hue,
        col=col,
        row=row, 
        # row="model_idx",
        aspect=aspect, #1.2,
        # errorbar=None,
        height=1.5,
        s=2,
        linewidth=0,
        facet_kws={'sharey': True, 'sharex': False}
    )

    if drift_idx is not None:
        drift_idx = [0] + drift_idx + [df[x_str].max()]

    all_idx = []
    for ax, (protocol, file, model) in zip(g.axes.flat, product(df["protocol"].unique(), df["file"].unique(), df["model"].unique())):
        data = df[(df["protocol"] == protocol) & (df["file"]==file) & (df["model"]==model)]
        ax.scatter(x=data[x_str], y=data["threshold"], color="red", s=1, linewidth=0)
        if drift_idx is not None:

            changes = data["model_idx"] != data["model_idx"].shift(1)
            pred_drift = data[changes]["batch_num"].to_numpy()
            dist = find_minimum_distances(drift_idx, pred_drift)
            all_idx.append(pred_drift)
            print(protocol, dist.mean())

            idle = True
            for start, end in zip(drift_idx[:-1], drift_idx[1:]):
                if idle:
                    ax.axvspan(start, end, facecolor="grey", alpha=0.3)
                idle = not idle
                
    if drift_idx is not None:
        all_idx = np.sort(np.hstack(all_idx))
        print("all", find_minimum_distances(drift_idx, all_idx).mean())

    if xrange is not None:
        for ax, xlim in zip(g.axes.flat, xrange):
            ax.set_xlim(*xlim)  # Example limits
            
    if yrange is not None:
        g.set(ylim=yrange)

    g.set_ylabels("Anomaly Score")
    g.set_xlabels("Batch Number")

    if draw_title:
        if titles is None:
            g.set_titles("{col_name}")
        else:
            for ax, title in zip(g.axes.flat, titles):
                ax.set_title(title)
    else:
        g.set_titles("")

        # g._legend.remove()
        # g.set_titles("{row_name}")
        # g.set_titles("")
    if draw_legend:
        sns.move_legend(
            g,
            title="Model",
            bbox_to_anchor=(0.95, 0.5),
            markerscale=3,
            handlelength=0.5,
            frameon=False,
            # columnspacing=1,
            # loc="lower center",
            loc="center right",
            # ncol=7,
        )
        

    # if "ARCUS" not in model_name:
    #     for ax in g.axes.flat:
    #         ax.set_yscale("log")

    g.set(yscale="log")

    if not draw_ticks:
        for ax in g.axes.flatten():
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

    # for ax in g.axes.flat:
    #     ax.axvline(x=16000, color='g')

    g.tight_layout()

    if len(file_name) ==1 and len(model_name)==1:
        fig_path = f"../datasets/{dataset_name}/{fe_name}/plots/{file_name[0]}/{model_name[0]}.png"
    else:
        fig_path = f"../datasets/{dataset_name}/{fe_name}/plots/sample_cd.png"
    g.savefig(fig_path)
    print(f"concept drift plot saved at {fig_path}")
    plt.close()






def dummy_uq_graph():
    G_b=nx.DiGraph()
    G_m=nx.DiGraph()


    G_m.add_nodes_from([
        ("Router", {"node_as": 90,"idx":"1", "updated":"False"}),
        ("Attacker", {"node_as": 120,"idx":"5", "updated":"False"}),
        ("Smart TV", {"node_as": 130,"idx":"6", "updated":"False"}),
    ])

    G_b.add_nodes_from([
       
        ("Smart Clock", {"node_as": 0.24, "idx":"2", "updated":"False"}),
        ("Google Nest", {"node_as": 0.23, "idx":"3", "updated":"False"}),
        ("Camera", {"node_as": 0.20, "idx":"0", "updated":"False"}),
        ("Router", {"node_as": 0.19, "idx":"1", "updated":"False"}),
        ("Smart Plug", {"node_as": 0.19, "idx":"4", "updated":"False"}),
    ])
    G_b.graph["threshold"]="0.1"
    G_m.graph["threshold"]="0.1"
    
    edges_b = [
        ("Smart Plug", "Router"),
        ("Camera", "Router"),
        ("Smart Clock", "Router"),
        ("Smart Clock", "Google Nest"),
        ("Google Nest", "Router"),
    ]

    edges_m = [
        ("Attacker", "Smart TV"),
        ("Smart TV", "Router"),
    ]

    for i, j in edges_b:
        G_b.add_edge(i, j)
        G_b.add_edge(j, i)
        
    for i, j in edges_m:
        G_m.add_edge(i, j)
        G_m.add_edge(j, i)
        
    fig, ax = plt.subplots(ncols=2, figsize=(5,2.5))

    draw_graph(G_b, fig, ax[0], False, "Benign", {}, {0:"Camera",1:"Router",2:"Smart Clock",3:"Google Nest", 4:"Smart Plug",
                                             5:"Attacker",6:"Smart TV"})
    draw_graph(G_m, fig, ax[1], False, "ACK Flooding", {}, {0:"Camera",1:"Router",2:"Smart Clock",3:"Google Nest", 4:"Smart Plug",
                                             5:"Attacker",6:"Smart TV"})
        

    fig.tight_layout()
    fig_name = f"tmp.png"
    fig.savefig(fig_name)
    print(fig_name)



if __name__ == "__main__":
    dummy_uq_graph()
    raise
    # get_results("UQ_IoT_IDS21/AfterImage")
    # get_results("CIC_IDS_2017/AfterImageGraph", gnn=True)
    # compare_spikes("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "AE")
    
    # plot_concept_drift(
    #     "UQ_IoT_IDS21",
    #     "AfterImage",
    #     ["benign/whole_week","malicious/Smart_TV/ACK_Flooding"],
    #     ["AE"],
    #     col="file",
    #     draw_legend=False,
    #     xrange=[(9300, 14000), (-10,4000)],
    #     yrange=(1,1e5),
    #     draw_title=True,
    #     titles=["Benign","ACK Flooding"]
    # )
    
    # raise

    # models = ["AE","ICL","KitNET","SLAD","VAE"]  # "KitNET","SLAD", "ICL"
    # dist = ["gaussian"]  # ,"log-normal","uniform"
    # gnn = ["LinearNodeEncoder"] #"GATNodeEncoder",, "LinearNodeEncoder"
    
    # plot_concept_drift(
    #     "FakeGraphData",
    #     "SyntheticFeatureExtractor",
    #     ["benign/feature_correlation_test"],
    #     # "UQ_IoT_IDS21",
    #     # "AfterImageGraph_multi_layer",
    #     # ["benign/whole_week"], #"malicious/Smart_TV/ACK_Flooding",
    #     [f"MultiLayerOCDModel-1000-50-{g}-{d}-{m}" for g, d, m in product(gnn, dist, models)],
    #     # protocols=["TCP","UDP"],
    #     draw_title=True,
    #     draw_legend=False,
    #     titles=models,
    #     col="model",
    #     # titles=models,
    #     # drift_idx=[
    #     #     3653,
    #     #     9284,
    #     #     14028,
    #     #     19799,
    #     #     24285,
    #     #     30052,
    #     #     33641,
    #     #     38407,
    #     #     42492,
    #     #     47599,
    #     #     50287,
    #     #     51505,
    #     #     55157,
    #     #     56983,
    #     #     60713,
    #     #     63163,
    #     # ],
    # )
    # raise 

    # plot_graph(
    #     "CIC_IDS_2017",
    #     "AfterImageGraph",
    #     "Friday-WorkingHours",
    #     "MultiLayerOCDModel-1000-50-GATNodeEncoder-gaussian-AE",
    #     56332633,
    #     relative_as=False,
    #     # protocols=["SSH"]
    # )
    # raise
    cic_files = [
        "Monday-WorkingHours",
        "Tuesday-WorkingHours",
        "Wednesday-WorkingHours",
        "Thursday-WorkingHours",
        "Friday-WorkingHours",
    ]

    drift_idxes = [
        [],
        [2685, 26565, 35189, 39421],
        [16000, 24725, 24892, 26257, 26725, 36387, 38654, 39825, 48695, 50231],
        [10210, 14538, 15080, 15828, 16163, 16301, 25936, 27107, 28539, 29300, 29648, 33023],
        [14925, 19259, 24776, 26506, 27272, 30526, 31748, 37497]
    ]

    protocols=[["Other","HTTPS"],["SSH","FTP"],["HTTPS","HTTP"],["TCP","HTTP"],["TCP","HTTP"]]
    
    for file, idx, proto in zip(cic_files, drift_idxes,protocols):
        plot_concept_drift(
            "CIC_IDS_2017",
            "AfterImageGraph",
            [file],
            ["MultiLayerOCDModel-1000-50-GATNodeEncoder-gaussian-AE"],
            protocols=proto,
            col=None,
            row="protocol",
            draw_legend=False,
            draw_title=True,
            titles=proto,
            # yrange=[1e-6,10e6],
            drift_idx=idx,
        )
