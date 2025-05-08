
import importlib
import time
from typing import Dict

from tqdm import tqdm
import yaml
from ..component.pipeline_component import PipelineComponent
from ..save_mixin.yaml import YamlSaveMixin


class Pipeline(YamlSaveMixin, PipelineComponent):
    
    def __init__(self, **kwargs):
        """A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """        
        super().__init__(component_type="pipeline")
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.save_attr.append(key)
        self.start_time=None
        self.prefix=[]
        
    

    
    def load_components(self, manifest):
        components = {}        
        for type, meta in manifest.items():
            module = importlib.import_module(f"ANIDSC.{type}")
            component_cls = getattr(module, meta["class"])
            if meta.get("file", False):
                file_path = meta["file"]
                comp = component_cls.load(file_path)
                
            else:
                comp = component_cls(**meta.get("attrs", {}))
            comp.parent_pipeline=self
            components[type] = comp
        return components
        
    def on_load(self):
        self.components=self.load_components(self.manifest)
        for key, comp in self.components.items():
            comp.on_load()
    
    def add_prefix(self, prefix):
        self.prefix.append(prefix)
    
    def setup(self):
        
        self.components=self.load_components(self.manifest)
        
        for key, comp in self.components.items():
            comp.setup()
            

    def perform_action(self, comp_type, action):
        if comp_type in self.components:
            return getattr(self.components[comp_type], action)()
    
    
    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time=time.time()
        for comp_type, component in self.components.items():
            data = component.preprocess(data)
            data = component.process(data)
            data = component.postprocess(data)
            if data is None: 
                break
        return comp_type
            
        

    def start(self):
        
        pbar = tqdm()
        
        while True:
            comp_type=self.process()
            
            pbar.update(1)
            if comp_type=="data_source":
                break

        self.save()
            
    def get_save_path_template(self):
        fe_name=self.perform_action('feature_extractor', '__str__')
        if fe_name is None:
            fe_name=self.get_attr("data_source","fe_name")
        
        if len(self.prefix)==0:
        
            return f"{self.get_attr('data_source','dataset_name')}/{fe_name}/saved_components/{{}}/{self.get_attr('data_source','file_name')}/{{}}.{{}}"

        else:
            prefix_str="/".join(self.prefix)
            return f"{self.get_attr('data_source','dataset_name')}/{fe_name}/saved_components/{{}}/{self.get_attr('data_source','file_name')}/{prefix_str}/{{}}.{{}}"


    def get_attr(self, comp_type, attr, default=None):
        if comp_type in self.components:
            return getattr(self.components[comp_type], attr, default)
        else:
            return self.request_attr(comp_type, attr, default)
        
    def __eq__(self, other: 'Pipeline'):
        same_class=self.__class__==other.__class__ 
        if not same_class:
            return False
        
       
        return self.components==other.components

    def __str__(self):
        if len(self.prefix)==0:
            return "->".join([str(component) for _, component in self.components.items()])
        else:
            return "/".join(self.prefix) +"/"+ "->".join([str(component) for _, component in self.components.items()])
    
