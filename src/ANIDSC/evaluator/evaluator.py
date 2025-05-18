import json
import os
from typing import Any, Dict, List
from networkx.drawing.nx_pydot import write_dot
import networkx as nx
import numpy as np

from ..save_mixin.null import NullSaveMixin
from ..component.pipeline_component import PipelineComponent
from . import od_metrics
from pathlib import Path
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
from ..templates import METRICS


class BaseEvaluator(NullSaveMixin, PipelineComponent):
    def __init__(
        self,
        metric_list: List[str] = METRICS,
        log_to_tensorboard: bool = True,
        graph_period=1e4,
    ):
        """base evaluator that evaluates the output of a single model
        Args:
            metric_list (List[str]): list of metric names in string format
            log_to_tensorboard (bool, optional): whether to write to tensorboard. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            draw_graph_rep_interval (bool, optional): whether to draw the graph representation. only available if pipeline contains graph representation. Defaults to False.
        """
        super().__init__()
        self.metrics = [getattr(od_metrics, m) for m in metric_list]
        self.metric_list = metric_list
        self.log_to_tensorboard = log_to_tensorboard
        self.graph_period = graph_period
        self.last_anomaly=0
        self.comparable = False

    def setup(self):
        super().setup()

        dataset_name = self.request_attr("data_source", "dataset_name")
        fe_name = self.request_attr("data_source", "fe_name")
        file_name = self.request_attr("data_source", "file_name")
        pipeline_name = str(self.parent_pipeline)

        self.file_template = (
            f"{dataset_name}/{fe_name}/{{}}/{file_name}/{pipeline_name}"
        )
        
        if self.graph_period:
            graph_plot_path = Path(self.file_template.format("graphs"))
            graph_plot_path.mkdir(parents=True, exist_ok=True)

        # file to store outputs
        scores_and_thresholds_path = Path(self.file_template.format("results") + ".csv")
        scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = open(str(scores_and_thresholds_path), "a")
        if os.stat(scores_and_thresholds_path).st_size == 0:
            self.output_file.write(
                ",".join(["time", "batch_num"] + self.metric_list) + "\n"
            )

        if self.log_to_tensorboard:
            log_dir = self.file_template.format("runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            print("tensorboard logging to", log_dir)

        self.prev_timestamp = time.time()

    def on_load(self):
        self.setup()

    def save(self):

        self.output_file.close()
        print("results file saved at", self.output_file.name)
        self.write_header = True

    def process(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """processes results and log them accordingly

        Args:
            results (Dict[str, Any]): gets metric values in metric_list based on results

        Returns:
            Dict[str, Any]: dictionary of metric, value pair
        """
        # records time

        current_time = time.time()
        duration = current_time - self.parent_pipeline.start_time

        result_dict = {"time": duration, "batch_num": results["batch_num"]}

        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name] = metric(results)

            if self.log_to_tensorboard:
                self.writer.add_scalar(
                    metric_name,
                    result_dict[metric_name],
                    results["batch_num"],
                )

        self.output_file.write(",".join(map(str, result_dict.values())) + "\n")

        
        if self.graph_period:
            # plot graphs periodically
            if (results["batch_num"] % self.graph_period == 0):
                self.save_graph(results)
            
            # plot if there is an anomaly
            if results['score'] is not None:
                if (results['batch_num']-self.last_anomaly)>self.graph_period and np.median(results['score']) > results['threshold']:
                    self.save_graph(results)
                    self.last_anomaly=results['batch_num']
            
    def save_graph(self, results):
        G = self.request_attr("graph_rep", "G", None)
        G = to_networkx(
            G, node_attrs=["x", "idx", "updated"], edge_attrs=["edge_attr"]
        )
        
        # get mac_to_idx_map
        mac_to_idx_map=self.request_attr("data_source","fe_attrs")["mac_to_idx_map"]
        idx_to_max_map={v:k for k,v in mac_to_idx_map.items()}
        nx.set_node_attributes(G, idx_to_max_map, "mac_address")
        
        # get threshold
        G.graph["threshold"]=results["threshold"]
        
        # add anomaly score
        if results["score"] is None:
            nx.set_node_attributes(G, float("inf"), "node_as")
        else:
            # assign edge nodes
            score_map={i:v for i,v in enumerate(results["score"].tolist())}
            nx.set_node_attributes(G, score_map, "node_as")
        
        with open(self.file_template.format("graphs")+f"/{results['batch_num']}.json", "w") as f:
            json.dump(nx.cytoscape_data(G), f)
           