import json
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from collections import defaultdict
from utils import *
from matplotlib.ticker import NullFormatter
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import os
from adjustText import adjust_text


plt.style.use(["science", "ieee"])


def get_results(dataset_name, gnn=False):
    results_path = Path(f"../../datasets/{dataset_name}/results.json")

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
        f"../../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}_raw_scores.csv"
    )
    df = pd.read_csv(str(results_file), header=None)

    meta_file = Path(f"../../datasets/{dataset_name}/{fe_name}/{file_name}_meta.csv")
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
    draw_ticks=True,
    draw_legend=True,
):
    results_file = Path(
        f"../../datasets/{dataset_name}/{fe_name}/outputs/{file_name}/{model_name}.csv"
    )
    df = pd.read_csv(str(results_file))
    
    df["relative_batch_num"] = df["batch_num"] - df.iloc[0]["batch_num"]

    x_str = "relative_batch_num"

    if "model_idx" not in df:
        df["model_idx"] = 1

    df["model_idx"] = df["model_idx"].astype(str)

    if protocols is not None:
        df = df[df["protocol"].isin(protocols)]

    if draw_legend:
        hue="model_idx"
    else:
        hue=None

    g = sns.relplot(
        data=df,
        kind="scatter",
        x=x_str,
        y="score",
        hue=hue,
        row="protocol",
        # row="model_idx",
        aspect=1.618,
        # errorbar=None,
        height=1.5,
        s=2,
        linewidth=0,
    )

    if drift_idx is not None:
        drift_idx = [0] + drift_idx + [df[x_str].max()]

    all_idx = []
    for ax, protocol in zip(g.axes.flat, df["protocol"].unique()):
        data = df[df["protocol"] == protocol]
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
        g.set(xlim=xrange)

    if yrange is not None:
        g.set(ylim=yrange)

    g.set_ylabels("")
    g.set_xlabels("")

    if not draw_legend:
        # g._legend.remove()
        g.set_titles("{row_name}")
        # g.set_titles("")
    else:
        sns.move_legend(
            g,
            title="Model Index",
            bbox_to_anchor=(0.47, -0.06),
            markerscale=3,
            handlelength=2,
            frameon=False,
            columnspacing=1,
            loc="lower center",
            ncol=7,
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

    fig_path = f"../../datasets/{dataset_name}/{fe_name}/plots/{file_name}/{model_name}_drift.png"
    g.savefig(fig_path)
    print(f"concept drift plot saved at {fig_path}")
    plt.close()


def plot_graph(
    dataset_name,
    fe_name,
    file_name,
    model_name,
    processed,
    protocols=None,
    mac_to_device_map={},
    relative_as=False
):

    idx_to_mac_map = get_node_map(dataset_name, fe_name, file_name)
    if idx_to_mac_map is not None:
        idx_to_mac_map = {v: k for k, v in idx_to_mac_map.items()}
    else:
        idx_to_mac_map = {}

    if protocols is None:
        paths = list(
            Path(
                f"../../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}"
            ).glob("*.dot")
        )
    else:
        paths = [
            Path(
                f"../../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}/{p}.dot"
            )
            for p in protocols
        ]

    fig, ax = plt.subplots(ncols=len(paths), figsize=(5, 5), squeeze=False)

    for i, p in enumerate(paths):
        G = nx.drawing.nx_agraph.read_dot(p)

        # convert attributes
        for node, attrs in G.nodes(data=True):
            for attr in attrs:
                G.nodes[node][attr] = eval(G.nodes[node][attr])
        G.graph['threshold'] = eval(G.graph['threshold'])
            
            
        pos = nx.spiral_layout(G, resolution=0.8, equidistant=True)

        ax[0][i].margins(x=0.1)

        node_as = np.array(list(nx.get_node_attributes(G, "node_as").values()))
        if relative_as:
            node_as -= G.graph['threshold']
            
            node_sm = ScalarMappable(
                norm = TwoSlopeNorm(vcenter=0, vmin=np.min(node_as), vmax=np.max(node_as)),
                cmap = matplotlib.colormaps["coolwarm"],
            )
        else:
            

            node_sm = ScalarMappable(
                norm=Normalize(vmin=np.min(node_as), vmax=np.max(node_as)),
                cmap=plt.cm.winter,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[node_sm.to_rgba(i) for i in node_as],
            node_size=500,
            ax=ax[0][i],
        )

        nx.draw_networkx_edges(
            G,
            pos,
            node_size=500,
            arrowstyle="->",
            arrowsize=20,
            ax=ax[0][i],
        )

        # Draw the labels with conditional formatting

        labels = nx.get_node_attributes(G, "idx")
        texts = []
        for node, label in labels.items():
            node_label = mac_to_device_map.get(
                idx_to_mac_map.get(label, label), idx_to_mac_map.get(label, label)
            )

            fc = "red" if G.nodes[node]["updated"] else "white"
            text = plt.text(
                pos[node][0],
                pos[node][1],
                s=node_label,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"ec": "k", "fc": fc, "alpha": 0.7},
            )
            texts.append(text)

        # Adjust the text to avoid overlap
        adjust_text(texts, only_move={"points": "xy", "texts": "xy"}, autoalign="xy")

        

        node_cbar = fig.colorbar(node_sm, location="right", ax=ax[0][i])
        node_cbar.ax.set_ylabel("node anomaly score", rotation=90)

        fig.tight_layout()
        file_name = f"../../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}/graph.png"
        fig.savefig(file_name)
        print(file_name)

    # nodes = [
    #     ("Camera", {"as": 0.20}),
    #     ("Router", {"as": 0.19}),
    #     ("Smart Clock", {"as": 0.24}),
    #     ("Google Nest", {"as": 0.23}),
    #     ("Smart Plug", {"as": 0.19}),
    # ]

    # nodes = [
    #     ("Router", {"as": 90}),
    #     ("Attacker", {"as": 120}),
    #     ("Smart TV", {"as": 130}),
    # ]

    # G.add_nodes_from(nodes)

    # edges = [
    #     ("Smart Plug", "Router"),
    #     ("Camera", "Router"),
    #     ("Smart Clock", "Router"),
    #     ("Smart Clock", "Google Nest"),
    #     ("Google Nest", "Router"),
    # ]

    # edges = [
    #     ("Attacker", "Smart TV"),
    #     ("Smart TV", "Router"),
    # ]

    # for i, j in edges:
    #     G.add_edge(i, j)
    #     G.add_edge(j, i)

    # pos = nx.circular_layout(G)

    # ax.margins(x=0.3, y=0.2)

    # node_as = np.array(list(nx.get_node_attributes(G, "as").values()))

    # node_sm = ScalarMappable(
    #     norm=Normalize(vmin=np.min(node_as), vmax=np.max(node_as)), cmap=plt.cm.winter
    # )

    # nx.draw_networkx_nodes(
    #     G, pos, node_color=[node_sm.to_rgba(i) for i in node_as], node_size=500, ax=ax
    # )
    # nx.draw_networkx_edges(G, pos, arrows=True, node_size=500, ax=ax)

    # label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    # nx.draw_networkx_labels(G, pos, font_size=10, bbox=label_options, ax=ax)

    # node_cbar = fig.colorbar(node_sm, location="right", ax=ax)
    # node_cbar.ax.set_ylabel("node anomaly score", rotation=90)

    # fig.tight_layout()
    # fig.savefig("tmp.png")
    # print("tmp.png")


if __name__ == "__main__":
    # get_results("UQ_IoT_IDS21/AfterImage")
    # get_results("CIC_IDS_2017/AfterImageGraph", gnn=True)
    # compare_spikes("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "AE")
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImage", "benign/whole_week", "AE", draw_idle=False,)
    # plot_concept_drift("UQ_IoT_IDS21", "AfterImage", "malicious/Smart_TV/ACK_Flooding", "AE")
    # plot_graph(
    #     "CIC_IDS_2017",
    #     "AfterImageGraph",
    #     "Monday-WorkingHours",
    #     "MultiLayerOCDModel-1000-50-GATNodeEncoder-gaussian-AE",
    #     58368,
    #     relative_as=True
    # )

    # models = ["AE"]  # "KitNET","SLAD", "VAE", "ICL"
    # dist = ["gaussian"]  # ,"log-normal","uniform"
    # gnn = ["GATNodeEncoder", "GCNNodeEncoder", "LinearNodeEncoder"]
    # for m in models:
    #     for d in dist:
    #         for g in gnn:
    #             plot_concept_drift(
    #                 # "FakeGraphData",
    #                 # "SyntheticFeatureExtractor",
    #                 # "benign/feature_correlation_test",
    #                 "UQ_IoT_IDS21",
    #                 "AfterImageGraph_multi_layer",
    #                 "malicious/Smart_TV/ACK_Flooding",
    #                 # "benign/whole_week",
    #                 f"MultiLayerOCDModel-1000-50-{g}-{d}-{m}",
    #                 # protocols=["UDP","TCP"],
    #                 draw_idle=False,
    #                 draw_legend=False,
    #                 drift_idx=[
    #                     3653,
    #                     9284,
    #                     14028,
    #                     19799,
    #                     24285,
    #                     30052,
    #                     33641,
    #                     38407,
    #                     42492,
    #                     47599,
    #                     50287,
    #                     51505,
    #                     55157,
    #                     56983,
    #                     60713,
    #                     63163,
    #                 ],
    #             )

    # plot_concept_drift("UQ_IoT_IDS21", "AfterImageGraph_multi_layer", "malicious/Smart_TV/ACK_Flooding", "MultiLayerGNNIDS-gaussian")
    # plot_concept_drift("FakeGraphData", "SyntheticFeatureExtractor", "benign/mean_std_drift_4", "MultiLayerGNNIDS")

    cic_files = [
        "Monday-WorkingHours",
        "Tuesday-WorkingHours",
        "Wednesday-WorkingHours",
        "Thursday-WorkingHours",
        "Friday-WorkingHours",
    ]

    drift_idxes=[
        [],
        [2685, 26565, 35189, 39421],
        [23139, 24725, 24892, 26257, 26725, 36387, 38654, 39825, 48695, 50231],
        [10210, 14538, 15080, 15828, 16163, 16301, 25936, 27107, 28539, 29300, 29648, 33023],
        [14925, 19259, 24776, 26506, 27272, 30526, 31748, 37497]
        ]

    for file, idx in zip(cic_files, drift_idxes):
        plot_concept_drift(
                        "CIC_IDS_2017",
                        "AfterImageGraph",
                        file,
                        f"MultiLayerOCDModel-1000-50-GATNodeEncoder-gaussian-AE",
                        # protocols=[],
                        draw_legend=True,
                        drift_idx=idx)
