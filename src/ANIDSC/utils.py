import json
from pathlib import Path
from io import TextIOWrapper
import numpy as np 
import torch 
import pickle
import scipy
from pytdigest import TDigest
from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import os
from adjustText import adjust_text
import textwrap
from datetime import datetime, timedelta, time
import pytz
import matplotlib.pyplot as plt
import matplotlib


# def plot_graph(
#     dataset_name,
#     fe_name,
#     file_name,
#     model_name,
#     processed,
#     protocols=None,
#     mac_to_device_map={},
#     relative_as=False,
# ):

#     idx_to_mac_map = get_node_map(dataset_name, fe_name, file_name)
#     if idx_to_mac_map is not None:
#         idx_to_mac_map = {v: k for k, v in idx_to_mac_map.items()}
#     else:
#         idx_to_mac_map = {}

#     if protocols is None:
#         paths = list(
#             Path(
#                 f"../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}"
#             ).glob("*.dot")
#         )
#     else:
#         paths = [
#             Path(
#                 f"../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}/{p}.dot"
#             )
#             for p in protocols
#         ]

#     fig, ax = plt.subplots(ncols=len(paths), figsize=(5 * len(paths), 3), squeeze=False)

#     for i, p in enumerate(paths):
#         G = nx.drawing.nx_agraph.read_dot(p)
#         draw_graph(G, fig, ax[0][i], relative_as, p.stem, mac_to_device_map, idx_to_mac_map)
        

#     fig.tight_layout()
#     fig_name = f"../datasets/{dataset_name}/{fe_name}/graph_data/{file_name}/{model_name}/{processed}/graph.png"
#     fig.savefig(fig_name)
#     print(fig_name)


def fig_to_array(fig):
    """convert fig to plot used in tensorboard

    Args:
        fig (_type_): matplotlib figure

    Returns:
        _type_: _description_
    """
    # Convert a Matplotlib figure to a PNG image and return it
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data.transpose(2, 0, 1)  # Convert to [C, H, W] format

def draw_graph(G, fig, ax, threshold, node_as, relative_as, title, mac_to_device_map, idx_to_mac_map, cmap=plt.cm.Blues):
    # convert attributes
    for node, attrs in G.nodes(data=True):
        for attr in attrs:
            if isinstance(G.nodes[node][attr], str):
                G.nodes[node][attr] = eval(G.nodes[node][attr])

    # pos = nx.spiral_layout(G, resolution=0.8, equidistant=True)
    
    pos=nx.circular_layout(G)
    
    ax.margins(x=0.2, y=0.3)
    ax.set_title(title)

    if relative_as:
        node_as-=threshold

        node_sm = ScalarMappable(
            norm=TwoSlopeNorm(
                vcenter=0, vmin=np.min(node_as), vmax=np.max(node_as)
            ),
            cmap=matplotlib.colormaps["coolwarm"],
        )
    else:

        node_sm = ScalarMappable(
            norm=Normalize(vmin=min(0,np.min(node_as)), vmax=np.max(node_as)),
            cmap=cmap,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_sm.to_rgba(i) for i in node_as],
        node_size=500,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        node_size=500,
        # arrowstyle="-|>",
        arrowsize=20,
        ax=ax,
    )

    # Draw the labels with conditional formatting

    labels = nx.get_node_attributes(G, "idx")
    texts = []
    for node, label in labels.items():
        node_label = mac_to_device_map.get(
            idx_to_mac_map.get(label, label), idx_to_mac_map.get(label, label)
        )

        # node_label="\n".join(textwrap.wrap(str(node_label), width=16))
        
        
        fc = "white" if G.nodes[node]["updated"] else "green"
        
        if pos[node][1]>0.5:
            direction=1
        else:
            direction=-1
        
        text = ax.text(
            pos[node][0],
            pos[node][1]+0.35*direction,
            s=node_label,
            fontsize=8,
            ha="center",
            va="center",
            fontweight=700,
            bbox={"ec": "k", "fc": fc, "alpha": 0.5},
            wrap=True
        )
        texts.append(text)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    

    node_cbar = fig.colorbar(node_sm, location="right", ax=ax)
    node_cbar.ax.set_ylabel("Node Anomaly Score", rotation=90)
    
    if not relative_as:
        node_cbar.ax.axhline(threshold, c='r')


def calc_quantile(x, p, dist="lognorm"):
    if dist=="lognorm":
        eps=1e-6
        x=np.log(np.array(x)+eps)
        mean=np.mean(x) 
        std=np.std(x)
        quantile=np.exp(mean+np.sqrt(2)*std*scipy.special.erfinv(2*p-1))
    elif dist=="norm":
        mean=np.mean(x) 
        std=np.std(x)
        quantile=mean+np.sqrt(2)*std*scipy.special.erfinv(2*p-1)
        
        
    return quantile

def is_stable(x, p=0.95, return_quantile=False):
    if len(x)!=x.maxlen:
        stability=False
        quantile=0.
    else:
        quantile=calc_quantile(x, p)
        stability=np.mean(np.array(x)<quantile)<p
        
    if return_quantile:
        return stability, quantile
    else:
        return stability

def uniqueXT(x, sorted=True, return_index=False, return_inverse=False, return_counts=False,
             occur_last=False, dim=None):
    if return_index or (not sorted and dim is not None):
        unique, inverse, counts = torch.unique(x, sorted=True,
            return_inverse=True, return_counts=True, dim=dim)
        inv_sorted, inv_argsort = inverse.flatten().sort(stable=True)

        if occur_last and return_index:
            tot_counts = (inverse.numel() - 1 - 
                torch.cat((counts.new_zeros(1),
                counts.flip(dims=[0]).cumsum(dim=0)))[:-1].flip(dims=[0]))
        else:
            tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        
        index = inv_argsort[tot_counts]
        
        if not sorted:
            index, idx_argsort = index.sort()
            unique = (unique[idx_argsort] if dim is None else
                torch.index_select(unique, dim, idx_argsort))
            if return_inverse:
                idx_tmp = idx_argsort.argsort()
                inverse.flatten().index_put_((inv_argsort,), idx_tmp[inv_sorted])
            if return_counts:
                counts = counts[idx_argsort]

        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_inverse:
            ret += (inverse,)
        if return_counts:
            ret += (counts,)
        return ret if len(ret)>1 else ret[0]
    
    else:
        return torch.unique(x, sorted=sorted, return_inverse=return_inverse,
            return_counts=return_counts, dim=dim)

def find_concept_drift_times(dataset_name, fe_name, file_name, timezone, schedule):
    times = pd.read_csv(
        f"../datasets/{dataset_name}/{fe_name}/{file_name}.csv",
        skiprows=lambda x: x % 256 != 0,
    )
    times = times["timestamp"]
    timezone = pytz.timezone(timezone)
    idle = True
    drift_idx = []
    for idx, time in times.items():
        # find time in brisbane, and adjusted time period
        pkt_time = datetime.fromtimestamp(
            float(time), tz=timezone
        )  # -timedelta(hours=17)

        # weekday schedule
        prev_idle = idle

        conditions = schedule[pkt_time.weekday()]

        for c in conditions:
            # print(c[0], pkt_time.time(), c[1])
            if c[0] <= pkt_time.time() <= c[1]:
                idle = False
                break
            else:
                idle = True

        if idle != prev_idle:
            drift_idx.append(idx)
    print(drift_idx)

