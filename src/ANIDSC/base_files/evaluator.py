from typing import Any, Dict, List
from .pipeline import PipelineComponent
from .. import metrics 
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
        self.write_header=True 
        
    def setup(self):
        context=self.get_context()
        if self.save_results:
            # file to store outputs
            scores_and_thresholds_path = Path(
                f"{context['dataset_name']}/{context['fe_name']}/results/{context['file_name']}/{context['model_name']}.csv"
            )
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(scores_and_thresholds_path), "w")
            
        if self.log_to_tensorboard:
            log_dir=f"{context['dataset_name']}/runs/{context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    protocol: ["Multiline", [f"average_score/{protocol}", f"average_threshold/{protocol}"]]
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
        # records time
        
        for protocol, result_dict in results.items():
            if result_dict is None:
                continue
            if self.log_to_tensorboard:
                #skip protocol information
                for name, value in result_dict.items():
                    if name=="batch_num":
                        continue 
                    elif name=="graph_rep":
                        self.writer.add_image(
                           protocol, value, global_step=result_dict['batch_num'])
                    else:       
                        self.writer.add_scalar(
                                        f"{name}/{protocol}",
                                        value,
                                        result_dict['batch_num'],
                                    )
                    
            if self.save_results:
                result_dict["protocol"]=protocol
                if self.write_header:
                    self.output_file.write(",".join(result_dict.keys()) + "\n")
                    self.write_header=False
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
        self.metrics=[getattr(metrics, m) for m in metric_list]
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
                f"{context['dataset_name']}/{context['fe_name']}/results/{context['file_name']}/{context['model_name']}.csv"
            )
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_file = open(str(scores_and_thresholds_path), "w")
            
    
        if self.log_to_tensorboard:
            log_dir=f"{context['dataset_name']}/runs/{context['pipeline_name']}"
            self.writer = SummaryWriter(
                log_dir=log_dir
                )
            print("tensorboard logging to", log_dir)
            
            
            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    "All": ["Multiline", ["average_score", "average_threshold"]]
                },
            }

            self.writer.add_custom_scalars(layout)

        self.prev_timestamp = time.time()
        
    def teardown(self):
        if self.save_results:
            self.output_file.close()
            print("results file saved at", self.output_file.name)
    
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
        
        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name]=metric(results)
        
        if self.save_results:
            if self.write_header:
                self.output_file.write(
                ",".join(list(map(str, result_dict.keys()))) + "\n"
            )
                self.write_header=False 
                
            self.output_file.write(
                ",".join(list(map(str, result_dict.values()))) + "\n"
            )
        
        if self.log_to_tensorboard:
            for name, value in result_dict.items():
                self.writer.add_scalar(
                                name,
                                value,
                                results['batch_num'],
                            )
                
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