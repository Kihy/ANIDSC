"""Pure helper / stats utilities (no UI dependencies)."""
from __future__ import annotations

import re
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


# ── Plotting layout ─────────────────────────────────────────────────────────

def horizontal_band_layout(G: nx.Graph) -> dict:
    """Force-directed layout per layer, then placed on fixed horizontal bands."""
    layer_y = {"Physical": 0, "Internet": 200, "Transport": 400}
    groups: dict[str, list] = {}
    for node, data in G.nodes(data=True):
        groups.setdefault(data.get("layer"), []).append(node)

    pos: dict = {}
    for layer, nodes in groups.items():
        subG = G.subgraph(nodes)
        sub_pos = nx.kamada_kawai_layout(subG)

        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        min_x, span_x = min(xs), max(max(xs) - min(xs), 1e-6)
        min_y, span_y = min(ys), max(max(ys) - min(ys), 1e-6)

        base_y = layer_y.get(layer, 0)
        for node in nodes:
            x = (sub_pos[node][0] - min_x) / span_x * 800
            y = (sub_pos[node][1] - min_y) / span_y * 100 + base_y
            pos[node] = (x, y)
    return pos


# ── Name helpers ─────────────────────────────────────────────────────────────

def simple_pipeline_name(name: str) -> Optional[str]:
    """Extract a human-readable model name from a pipeline path string."""
    parts: list[str] = []

    def _find(pattern: str) -> None:
        m = re.search(pattern, name)
        if m:
            parts.extend(m.groups())

    _find(r"GraphProcessor\(([^)]+)\)")
    _find(r"BaseNodeEmbedder\(([^)]+)\)")
    _find(r"OnlineOD\(([^)]+)\)")
    if "LivePercentile" in name:
        parts.append("scaler")
    return "+".join(parts)


# ── Stats ────────────────────────────────────────────────────────────────────

def f_beta(tp: float, fn: float, fp: float, beta: float = 1) -> float:
    beta2 = beta ** 2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return (1 + beta2) * tp / denom if denom else 0.0


def weighted_hmean(precision: float, recall: float, beta: float = 1) -> float:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def calc_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats: dict = {}
    mask_b = df["file"].str.startswith("benign")
    mask_m = df["file"].str.startswith("attack")

    stats["pos_benign"] = df.loc[mask_b, "pos_count"].sum()
    stats["total_benign"] = df.loc[mask_b, "batch_size"].sum()
    stats["pos_malicious"] = df.loc[mask_m, "pos_count"].sum()
    stats["total_malicious"] = df.loc[mask_m, "batch_size"].sum()
    stats["f1"] = f_beta(
        stats["pos_malicious"],
        stats["total_malicious"] - stats["pos_malicious"],
        stats["pos_benign"],
    )

    df = df.copy()
    df["accuracy"] = df["pos_count"] / df["batch_size"]
    df.loc[mask_b, "accuracy"] = 1 - df.loc[mask_b, "accuracy"]
    stats["acc_benign"] = df.loc[mask_b, "accuracy"].mean()
    stats["acc_malicious"] = df.loc[mask_m, "accuracy"].mean()
    stats["hmean_acc"] = weighted_hmean(stats["acc_benign"], stats["acc_malicious"])
    return pd.DataFrame([stats])
