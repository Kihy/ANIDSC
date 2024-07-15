from typing import Any, Dict, List

import numpy as np
from .pipeline import PipelineComponent
from .. import evaluations 
from pathlib import Path
import time
from ..utils import draw_graph, fig_to_array
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
        context=self.get_context()
        if self.save_results:
            # file to store outputs
            layerwise_path = Path(
                f"{context['dataset_name']}/{context['fe_name']}/results/{context['file_name']}/{context['pipeline_name']}.csv"
            )
            layerwise_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(layerwise_path), "w")
            self.write_output_header=True 
            
            
            
        if self.log_to_tensorboard:
            log_dir=f"{context['dataset_name']}/{context['fe_name']}/runs/{context['file_name']}/{context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    protocol: ["Multiline", [f"median_score/{protocol}", f"median_threshold/{protocol}"]]
                for protocol in context['protocols'].keys()},
            }

            self.writer.add_custom_scalars(layout)
    
    def teardown(self):
        self.output_file.close()
        print("results file saved at", self.output_file.name)
    
    def process(self, results:Dict[str, Dict[str, Any]]):
        """process the results dictionary

        Args:
            results (Dict[str, Dict[str, Any]]): layer: results pairs for each layer
        """        
        # all results
        pos_idx=[]
        
        #layerwise results
        for protocol, result_dict in results.items():
            if result_dict is None:
                continue
            
            if self.log_to_tensorboard:
                for name, value in result_dict.items():
                    if name=="batch_num":
                        continue 
                    elif name=="graph_rep":
                        self.writer.add_image(
                           protocol, value, global_step=result_dict['batch_num'])
                    elif isinstance(value, (int, float)):       
                        self.writer.add_scalar(
                                        f"{name}/{protocol}",
                                        value,
                                        result_dict['batch_num'],
                                    )
                    
            if self.save_results:
                result_dict["protocol"]=protocol
                
                # remove pos_idx
                pos_idx.append(result_dict["pos_idx"])
                del result_dict['pos_idx']
                
                if self.write_output_header:
                    self.output_file.write(",".join(result_dict.keys()) + "\n")
                    self.write_output_header=False
                self.output_file.write(
                    ",".join(list(map(str, result_dict.values()))) + "\n"
                )
                
        unique_values = np.unique(np.hstack(pos_idx))
        pos_counts=len(unique_values)
        detection_rate=pos_counts/result_dict["batch_size"]
        
        # modify result dict 
        result_dict["protocol"]="Overall"
        result_dict["detection_rate"]=detection_rate
        result_dict["pos_count"]=pos_counts 
        
        # List of keys to calculate the average for
        keys_to_average = ['time', 'model_idx', 'num_model', 'median_score', 'pos_count', 'batch_size']

        # Calculate the averages
        for key in keys_to_average:
            result_dict[key]= sum(d[key] for d in results.values()) / len(results) 
        
        self.output_file.write(
                    ",".join(list(map(str, result_dict.values()))) + "\n"
                )

        
        
        
class BaseEvaluator(PipelineComponent): 
    def __init__(self, metric_list:List[str], log_to_tensorboard:bool=True, save_results:bool=True, draw_graph_rep_interval:bool=False):
        """base evaluator that evaluates the output of a single model
        Args:
            metric_list (List[str]): list of metric names in string format
            log_to_tensorboard (bool, optional): whether to write to tensorboard. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            draw_graph_rep_interval (bool, optional): whether to draw the graph representation. only available if pipeline contains graph representation. Defaults to False.
        """        
        super().__init__()
        self.metrics=[getattr(evaluations, m) for m in metric_list]
        self.metric_list=metric_list
        self.log_to_tensorboard=log_to_tensorboard
        self.save_results=save_results
        self.draw_graph_rep_interval=draw_graph_rep_interval
        self.write_header=True
        
    def setup(self):
        super().setup()
        context=self.get_context()
        self.parent.context["metric_list"]=self.metric_list
        if self.save_results:
            # file to store outputs
            scores_and_thresholds_path = Path(
                f"{context['dataset_name']}/{context['fe_name']}/results/{context['file_name']}/{context['pipeline_name']}.csv"
            )
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(scores_and_thresholds_path), "w")
            
    
        if self.log_to_tensorboard:
            log_dir=f"{context['dataset_name']}/{context['fe_name']}/runs/{context['file_name']}/{context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
            
            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    "All": ["Multiline", ["median_score", "median_threshold"]]
                },
            }

            self.writer.add_custom_scalars(layout)
            
            

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
        duration=current_time -self.prev_timestamp
        self.prev_timestamp=current_time
        
        result_dict={"time":duration}
        if self.get_context().get('concept_drift_detection',False):
            result_dict['model_idx']= results['model_idx']
            result_dict['num_model']=results['num_model']
        
        header=[]
        values=[]
        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name]=metric(results)
            if isinstance(result_dict[metric_name], (int, float)):
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
        
                
        context=self.get_context()
        if context.get('G',False) and self.draw_graph_rep_interval and results['batch_num']%self.draw_graph_rep_interval==0:
            G=context['G']
            fig, ax = plt.subplots()

            idx_to_max_map={value: key for key, value in context['mac_to_idx_map'].items()}
            # draw graph
            draw_graph(
                G,  results['threshold'],results['score'], fig, ax, False, context["protocol"], {}, idx_to_max_map
            )

            image = fig_to_array(fig)
            plt.close(fig)
            
            if self.log_to_tensorboard:
                self.writer.add_image(
                    context["protocol"], image, global_step=results['batch_num']
                )
            else:
                result_dict["graph_rep"]=image
            
        
        result_dict['batch_num']=results['batch_num']
        return result_dict