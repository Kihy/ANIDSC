from abc import abstractmethod
import json
import os
from typing import Any, Dict, List

from ..save_mixin.pickle import PickleSaveMixin
import networkx as nx
import numpy as np

from ..component.feature_buffer import CompressedOutputWriter

from ..save_mixin.null import NullSaveMixin
from ..component.pipeline_component import PipelineComponent
from . import od_metrics
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


METRICS = [
    "detection_rate",
    "lower_quartile_score",
    "upper_quartile_score",
    "soft_min_score",
    "soft_max_score",
    "median_score",
    "median_threshold",
    "pos_count",
    "batch_size",
]

class BaseResultWriter(CompressedOutputWriter):
    @property 
    def output_path(self):
        pipeline_name = self.parent_pipeline.name

        return Path(
            f"{self.dataset_name}/{self.comp_name}/{self.folder_name}/{self.file_name}/{pipeline_name}.{self.file_type}"
        )

class CSVResultWriter(PickleSaveMixin, BaseResultWriter):
    @property
    def folder_name(self):
        return "results"

    @property
    def file_type(self):
        return "csv"


    
    def setup(self):
        super().setup()

        self.metrics = [getattr(od_metrics, m) for m in METRICS]
        if os.stat(self.feature_path).st_size == 0:
            self.save_file.write(
                ",".join(["process_time", "batch_num", "timestamp"] + METRICS) + "\n"
            )

    def process(self, data):
        current_time = time.time()
        duration = current_time - self.request_attr("start_time")

        result_dict = {
            "process_time": duration,
            "batch_num": self.request_attr("batch_evaluated"),
            "timestamp": self.request_attr("timestamp"),
        }

        for metric_name, metric in zip(METRICS, self.metrics):

            result_dict[metric_name] = metric(data)

        self.save_file.write(",".join(map(str, result_dict.values())) + "\n")

        return data


class GraphResultWriter(PickleSaveMixin, BaseResultWriter):

    @property
    def folder_name(self):
        return "graphs"

    @property
    def file_type(self):
        return "ndjson"

    def __init__(self, graph_period=1):
        super().__init__()
        self.graph_period = graph_period
        self.last_timestamp = None

    def process(self, data):
        cur_time = self.request_attr("timestamp")

        if self.last_timestamp is None:
            self.last_timestamp = self.request_attr("timestamp")

        if cur_time - self.last_timestamp > self.graph_period:
            self.save_graph(data)

            self.last_timestamp = cur_time

        return data

    def save_graph(self, data):
        G = self.request_attr("networkx")

        # get threshold
        G.graph["threshold"] = data["threshold"]

        # add anomaly score
        if data["score"] is None:
            nx.set_node_attributes(G, float("inf"), "node_as")
        else:
            # assign edge nodes
            score_map = {i: float(v) for i, v in zip(G.nodes, data["score"])}
            nx.set_node_attributes(G, score_map, "node_as")

        json.dump(nx.cytoscape_data(G), self.save_file)
        self.save_file.write("\n")
