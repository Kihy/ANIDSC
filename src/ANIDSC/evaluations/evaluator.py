from typing import Any, Dict, List
from networkx.drawing.nx_pydot import write_dot
import networkx as nx
import numpy as np

from ..save_mixin.null import NullSaveMixin
from ..components.pipeline_component import PipelineComponent
from . import od_metrics 
from pathlib import Path
import time
from ..utils2 import draw_graph, fig_to_array
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class CollateEvaluator(PipelineComponent):
    def __init__(self, log_to_tensorboard:bool=True, save_results:bool=True):
        """Evaluator to aggregate results from multiple base evaluators

        Args:
            log_to_tensorboard (bool, optional): whether to log results in tensorboard. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. Defaults to True.
        """        
        super().__init__()
        self.log_to_tensorboard=log_to_tensorboard
        self.save_results=save_results
        
        
    def setup(self):
        
        if self.save_results:
            # file to store outputs
            layerwise_path = Path(
                f"{self.context['dataset_name']}/{self.context['fe_name']}/results/{self.context['file_name']}/{self.context['pipeline_name']}.csv"
            )
            layerwise_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(layerwise_path), "w")
            self.write_output_header=True 
            
            
            # folder to dot folder
            self.dot_folder=Path(f"{self.context['dataset_name']}/{self.context['fe_name']}/dot/{self.context['file_name']}/{self.context['pipeline_name']}")
            self.dot_folder.mkdir(parents=True, exist_ok=True)
            print(f"dot folder {str(self.dot_folder)}")
            
            
        if self.log_to_tensorboard:
            log_dir=f"{self.context['dataset_name']}/{self.context['fe_name']}/runs/{self.context['file_name']}/{self.context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    protocol: ["Multiline", [f"median_score/{protocol}", f"median_threshold/{protocol}"]]
                for protocol in self.context['protocols'].keys()},
            }

            self.writer.add_custom_scalars(layout)
    
    def teardown(self):
        self.output_file.close()
        print("results file saved at", self.output_file.name)
        return self.output_file.name
    
    def process(self, results:Dict[str, Dict[str, Any]]):
        """process the results dictionary

        Args:
            results (Dict[str, Dict[str, Any]]): layer: results pairs for each layer
        """        
        
        #layerwise results
        for protocol, result_dict in results.items():
            if result_dict is None:
                continue
            
            if self.log_to_tensorboard:
                for name, value in list(result_dict.items()):
                    if name=="batch_num":
                        continue 
                    
                    elif name=="graph_rep":
                        self.writer.add_image(
                           protocol, value, global_step=result_dict['batch_num'])
                        del result_dict["graph_rep"]    
                    elif isinstance(value, (int, float, np.generic)):       
                        self.writer.add_scalar(
                                        f"{name}/{protocol}",
                                        value,
                                        result_dict['batch_num'],
                                    )
                
                    
            if self.save_results:
                if "G" in result_dict.keys():
                    write_dot(result_dict["G"], f"{str(self.dot_folder)}/{protocol}_{result_dict['batch_num']}.dot")
                    del result_dict["G"]
                    
                    
                result_dict["protocol"]=protocol
                
                if self.write_output_header:
                    self.output_file.write(",".join(result_dict.keys()) + "\n")
                    self.write_output_header=False
                    
                
                self.output_file.write(
                    ",".join(list(map(str, result_dict.values()))) + "\n"
                )     
        
class BaseEvaluator(NullSaveMixin, PipelineComponent): 
    def __init__(self, metric_list:List[str], log_to_tensorboard:bool=True, save_results:bool=True):
        """base evaluator that evaluates the output of a single model
        Args:
            metric_list (List[str]): list of metric names in string format
            log_to_tensorboard (bool, optional): whether to write to tensorboard. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            draw_graph_rep_interval (bool, optional): whether to draw the graph representation. only available if pipeline contains graph representation. Defaults to False.
        """        
        super().__init__()
        self.metrics=[getattr(od_metrics, m) for m in metric_list]
        self.metric_list=metric_list
        self.log_to_tensorboard=log_to_tensorboard
        self.save_results=save_results
        self.write_header=True
        
    def setup(self):
        super().setup()
        
        self.context["metric_list"]=self.metric_list
        
        if self.save_results:
            # file to store outputs
            scores_and_thresholds_path = Path(
                f"{self.context['dataset_name']}/{self.context['fe_name']}/results/{self.context['file_name']}/{self.context['pipeline_name']}.csv"
            )
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(scores_and_thresholds_path), "w")
            
        if self.log_to_tensorboard:
            log_dir=f"{self.context['dataset_name']}/{self.context['fe_name']}/runs/{self.context['file_name']}/{self.context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
        self.prev_timestamp = time.time()
        
    def teardown(self):
        if self.save_results:
            self.output_file.close()
            print("results file saved at", self.output_file.name)
            self.write_header=True
    
    def process(self, results:Dict[str, Any])->Dict[str, Any]:
        """processes results and log them accordingly

        Args:
            results (Dict[str, Any]): gets metric values in metric_list based on results

        Returns:
            Dict[str, Any]: dictionary of metric, value pair
        """        
        # records time

        current_time=time.time()
        duration=current_time - self.context["start_time"]
        
        result_dict={"time":duration}
        
        header=["time"]
        values=[str(duration)]
        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name]=metric(results)
            
            if self.log_to_tensorboard:
                self.writer.add_scalar(
                                metric_name,
                                result_dict[metric_name],
                                results['batch_num'],
                            )
    
            if self.save_results:
                if self.write_header:
                    header.append(metric_name)
                values.append(str(result_dict[metric_name]))
    
        if self.save_results:
            if self.write_header:
                self.output_file.write(",".join(header) + "\n")
                self.write_header=False
            self.output_file.write(",".join(values) + "\n")
        
        result_dict['batch_num']=results['batch_num']
        return result_dict