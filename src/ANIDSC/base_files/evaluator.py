from typing import Any, Dict, List
from networkx.drawing.nx_pydot import write_dot
import networkx as nx
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
            
            
            # folder to dot folder
            self.dot_folder=Path(f"{context['dataset_name']}/{context['fe_name']}/dot/{context['file_name']}/{context['pipeline_name']}")
            self.dot_folder.mkdir(parents=True, exist_ok=True)
            print(f"dot folder {str(self.dot_folder)}")
            
            
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
            
            # folder to dot folder
            self.dot_folder=Path(f"{context['dataset_name']}/{context['fe_name']}/dot/{context['file_name']}/{context['pipeline_name']}")
            self.dot_folder.mkdir(parents=True, exist_ok=True)
            print(f"dot folder {str(self.dot_folder)}")
            
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
        context=self.get_context()
        current_time=time.time()
        duration=current_time - context["start_time"]
        
        
        result_dict={"time":duration}
        if context.get('concept_drift_detection',False):
            result_dict['model_idx']= results['model_idx']
            result_dict['num_model']=results['num_model']
        
        header=["time"]
        values=[str(duration)]
        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name]=metric(results)
            if isinstance(result_dict[metric_name], (int, float, np.generic)):
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
            if results["score"] is None:
                results['score']=[0. for _ in G.nodes()]
                results['threshold']=0.
            
            nx.set_node_attributes(G, { n: {"score": score} for n, score in zip(G.nodes(), results["score"]) })
                
            G.graph["graph"]={'threshold':results["threshold"],
                              'protocol':context['protocol']}
            
            if self.save_results:
                # save graph as dot file 
                write_dot(G, f"{str(self.dot_folder)}/{context['protocol']}_{results['batch_num']}.dot")
            
            fig, ax = plt.subplots()
            
            idx_to_max_map={value: key for key, value in context['mac_to_idx_map'].items()}
            # draw graph
            draw_graph(
                G, fig, ax, results["threshold"], results["score"], False, context["protocol"], {}, idx_to_max_map
            )

            image = fig_to_array(fig)
            plt.close(fig)
            
            if self.log_to_tensorboard:
                self.writer.add_image(
                    context["protocol"], image, global_step=results['batch_num']
                )
            else:
                result_dict["graph_rep"]=image
            
            result_dict["G"]=G
        
        result_dict['batch_num']=results['batch_num']
        return result_dict