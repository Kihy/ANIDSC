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
from datetime import datetime

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
            f"{self.dataset_name}/{self.folder_name}/{self.comp_name}-{self.request_attr('run_identifier')}/{self.file_name}/{pipeline_name}.{self.file_type}"
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
            "timestamp": self.request_attr("timestamp")[-1],
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

    def __init__(self):
        super().__init__()


    def process(self, data):
        graphs = self.request_attr("transformed_graph")
        
        for g, score in zip(graphs, data["score"]):
            # get threshold
            g.graph["threshold"] = data["threshold"]
            g.graph["graph_as"]=score
        

            json.dump(nx.cytoscape_data(g), self.save_file)
            self.save_file.write("\n")
