from typing import Any, Dict, List, Optional, Callable, Set, get_type_hints
from dataclasses import dataclass
from functools import wraps
import inspect
import yaml

# ============================================================================
# COMPONENT SPEC + INTROSPECTION
# ============================================================================

@dataclass
class ComponentSpec:
    name: str
    category: str
    required: List[str]
    optional: Dict[str, Any]
    types: Dict[str, Any]
    doc: str


def infer_kwargs(func: Callable):
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    required = []
    optional = {}

    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            optional[name] = param.default

    return required, optional, hints


# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

class ComponentRegistry:
    """Registry for component creation functions with explicit contracts."""

    _components: Dict[str, ComponentSpec] = {}
    _functions: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, category: str = "component"):
        def decorator(func: Callable):
            required, optional, types = infer_kwargs(func)

            cls._components[name] = ComponentSpec(
                name=name,
                category=category,
                required=required,
                optional=optional,
                types=types,
                doc=func.__doc__ or f"{name} component",
            )

            cls._functions[name] = func
            setattr(ComponentFactory, name, staticmethod(func))
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._functions:
            raise ValueError(
                f"Unknown component '{name}'. "
                f"Available: {list(cls._functions.keys())}"
            )
        return cls._functions[name]

    @classmethod
    def validate_kwargs(cls, name: str, kwargs: dict):
        spec = cls._components[name]
        missing = set(spec.required) - kwargs.keys()
        if missing:
            raise ValueError(
                f"❌ Component '{name}' missing required parameters: "
                f"{sorted(missing)}\n"
                f"Required: {spec.required}\n"
                f"Optional: {list(spec.optional.keys())}",
                f"Provided: {kwargs}"
            )

    @classmethod
    def list_components(cls, category: Optional[str] = None) -> List[str]:
        if category:
            return [
                name for name, spec in cls._components.items()
                if spec.category == category
            ]
        return list(cls._components.keys())

    @classmethod
    def show_component_kwargs(cls, name: str) -> str:
        if name not in cls._components:
            raise ValueError(f"Unknown component: {name}")

        spec = cls._components[name]
        lines = [
            f"\nComponent: {name}",
            f"Category: {spec.category}",
            "\nRequired:"
        ]

        for r in spec.required:
            lines.append(f"  ✓ {r}: {spec.types.get(r, Any)}")

        if spec.optional:
            lines.append("\nOptional:")
            for k, v in spec.optional.items():
                lines.append(f"  • {k}: {spec.types.get(k, Any)} = {v}")

        lines.append(f"\nDescription: {spec.doc}")
        return "\n".join(lines)


# ============================================================================
# COMPONENT FACTORY (STATIC HELPERS)
# ============================================================================

class ComponentFactory:
    PROTOCOL_MAP = {
        "TCP": 0,
        "UDP": 1,
        "ICMP": 2,
        "ARP": 3,
        "Other": 4,
    }


# ============================================================================
# COMPONENT DEFINITIONS
# ============================================================================
  
@ComponentRegistry.register("reader", category="data_source")
def reader(
    reader_type: str,
    dataset_name: str,
    file_name: str,
    prev_pipeline:str=None,
    **kwargs
) -> dict:
    return {
        "type": "data_source",
        "class": reader_type,
        "attrs": {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "prev_pipeline": prev_pipeline
        },
    }


@ComponentRegistry.register("protocol_meta_extractor", category="meta_extractor")
def protocol_meta_extractor(**kwargs) -> dict:
    return {
        "type": "meta_extractor",
        "class": "ProtocolMetaExtractor",
    }
    
    
@ComponentRegistry.register("dict_feature_buffer", category="feature_buffer")
def feature_buffer(
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "DictFeatureBuffer",
        "attrs": {
            "buffer_size": buffer_size,
        },
    }
    
        
@ComponentRegistry.register("json_feature_buffer", category="feature_buffer")
def json_feature_buffer(
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "JsonFeatureBuffer",
        "attrs": {
            "buffer_size": buffer_size,
        },
    }


    
@ComponentRegistry.register("numpy_feature_buffer", category="feature_buffer")
def numpy_feature_buffer(
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "NumpyFeatureBuffer",
        "attrs": {
            "buffer_size": buffer_size,
        },
    }




@ComponentRegistry.register("feature_extractor", category="feature_extractor")
def feature_extractor(
    feature_extractor: str,
    fe_attr: Optional[dict] = None,
    **kwargs
) -> dict:
    return {
        "type": "feature_extractor",
        "class": feature_extractor,
        "attrs": fe_attr or {},
    }


    
@ComponentRegistry.register("graph_feature_extractor", category="feature_extractor")
def graph_feature_extractor(
    fe_attr: Optional[dict] = None,
    **kwargs
) -> dict:
    return {
        "type": "feature_extractor",
        "class": "GraphFeatureExtractor",
        "attrs": fe_attr or {},
    }





@ComponentRegistry.register("scaler", category="scaler")
def scaler(scale_idx=0, **kwargs) -> dict:
    return {"type": "scaler", "class": "LivePercentile", "attrs": {"scale_idx": scale_idx}}




@ComponentRegistry.register("od_model", category="model")
def od_model(model_name: str, model_params: dict, **kwargs) -> dict:
    return {
        "type": "model",
        "class": "BaseOnlineODModel",
        "attrs": {"model_name": model_name, "model_params": model_params},
    }


@ComponentRegistry.register("csv_evaluator", category="evaluator")
def csv_evaluator(**kwargs) -> dict:
    return {"type": "evaluator", "class": "CSVResultWriter"}

@ComponentRegistry.register("graph_evaluator", category="evaluator")
def graph_evaluator(**kwargs) -> dict:
    return {"type": "evaluator", "class": "GraphResultWriter"}



@ComponentRegistry.register("node_embedder", category="node_encoder")
def node_embedder(embedder_name: str, embedder_params: Optional[dict] = None, **kwargs) -> dict:
    return {
        "type": "node_encoder",
        "class": "BaseNodeEmbedder",
        "attrs": {"embedder_name": embedder_name, "embedder_params": embedder_params or {}},
    }


@ComponentRegistry.register("graph_rep", category="graph_rep")
def graph_rep(graph_rep: str, **kwargs) -> dict:
    return {
        "type": "graph_rep",
        "class": "GraphProcessor",
        "attrs": {"rep_name": graph_rep},
    }
    
@ComponentRegistry.register("time_remover", category="misc_component")
def time_remover(**kwargs) -> dict:
    return {
        "type": "misc_component",
        "class": "RemoveTimestamp",
        "attrs": {},

    }
@ComponentRegistry.register("splitter", category="pipeline_component")
def splitter(
    protocol_map: dict,
    **kwargs
) -> dict:
    return {
        "type": "pipeline",
        "class": "MultilayerSplitter",
        "attrs": {
            "name": kwargs["pipeline_name"],
            "components": kwargs["components"],
            "run_identifier": kwargs["run_identifier"],
            "protocol_map": protocol_map,
        },
    }

@ComponentRegistry.register("pipeline", category="pipeline")
def pipeline(pipeline_name:str, components: Any, run_identifier: str=None, **kwargs) -> dict:
    return {
        "type": "pipeline",
        "class": "Pipeline",
        "attrs": {
            "name": pipeline_name,
            "components": components,
            "run_identifier": run_identifier,
        },
    }


# ============================================================================
# PIPELINE BUILDER
# ============================================================================

class PipelineBuilder:
    def __init__(self):
        self.components: List[dict] = []

    def add(self, component_name: str, **kwargs):
        ComponentRegistry.validate_kwargs(component_name, kwargs)
        self.components.append(
            ComponentRegistry.get(component_name)(**kwargs)
        )
        return self

    def build(self, run_identifier: str) -> str:
        pipeline_dict = pipeline(self.components, run_identifier)
        return yaml.safe_dump(
            pipeline_dict,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )


# ============================================================================
# PIPELINE REGISTRY
# ============================================================================

class PipelineRegistry:
    _templates: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, description: str = "", required_kwargs=None):
        def decorator(func: Callable):
            if required_kwargs is None:
                sig = inspect.signature(func)
                required = [
                    p.name for p in sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind != inspect.Parameter.VAR_KEYWORD
                ]
            else:
                required = list(required_kwargs)

            cls._templates[name] = {
                "function": func,
                "description": description or func.__doc__ or name,
                "required": required,
            }
            return func
        return decorator

    @classmethod
    def get_template(cls, template_name: str, **kwargs) -> str:
        if template_name not in cls._templates:
            raise ValueError(f"Unknown pipeline template '{template_name}'. Available templates {cls._templates}")

        info = cls._templates[template_name]
        missing = set(info["required"]) - kwargs.keys()
        if missing:
            raise ValueError(
                f"❌ Pipeline '{template_name}' missing required parameters: "
                f"{sorted(missing)}"
            )

        components = info["function"](**kwargs)
        pipeline_dict = create_component("pipeline", components=components, **kwargs)
        return yaml.safe_dump(
            pipeline_dict,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_component(component_type: str, **kwargs) -> dict:
    ComponentRegistry.validate_kwargs(component_type, kwargs)
    return ComponentRegistry.get(component_type)(**kwargs)


def create_pipeline(template_name: str, **kwargs) -> str:
    return PipelineRegistry.get_template(template_name, **kwargs)



# ============================================================================
# REGISTER STANDARD TEMPLATES WITH REQUIRED KWARGS
# Templates should end with template
# ============================================================================

@PipelineRegistry.register(
    "metadata-extraction-template",
    "Extract metadata from pcap files",
)
def meta_extraction_template(**kwargs) -> List[dict]:
    """Create meta extraction pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("protocol_meta_extractor", **kwargs),
        create_component("dict_feature_buffer", buffer_size=1),
    ]



@PipelineRegistry.register(
    "feature-extraction-template",
    "Extract features from metadata",
)
def feature_extraction_template(**kwargs) -> List[dict]:
    """Create feature extraction pipeline components."""

    return [create_component("reader", **kwargs), 
            create_component("feature_extractor", **kwargs),
            create_component("numpy_feature_buffer", buffer_size=512)]


@PipelineRegistry.register(
    "graph-feature-extraction-template",
    "Extract features from data",
)
def graph_feature_extraction(**kwargs) -> List[dict]:
    """Create feature extraction pipeline components."""

    return [create_component("reader", **kwargs), 
            create_component("multi_feature_extractor", **kwargs),
            create_component("json_feature_buffer", buffer_size=1)]
    
    


@PipelineRegistry.register(
    "basic-detection-template",
    "Basic anomaly detection pipeline"
)
def boxplot_detection_template(**kwargs) -> List[dict]:
    """Create basic detection pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator"),
    ]
    

@PipelineRegistry.register(
    "scaled-detection-template",
    "Basic anomaly detection pipeline"
)
def tabular_detection_template(**kwargs) -> List[dict]:
    """Create basic detection pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("time_remover", **kwargs),
        create_component("scaler", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator"),
    ]


@PipelineRegistry.register(
    "basic-graph-detection-template",
    "Basic anomaly detection pipeline"
)
def basic_graph_detection_template(**kwargs) -> List[dict]:
    """Create basic detection pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("scaler", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator"),
    ]


@PipelineRegistry.register(
    "lager-layer-template",
    "Pipeline for each layer of Lager"
)
def lager_layer_template(**kwargs) -> List[dict]:
    return [
        create_component("scaler", scale_idx=4, **kwargs), # skip first few columns that are not features
        create_component("graph_rep", **kwargs),
        create_component("node_embedder", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator", **kwargs)] 

@PipelineRegistry.register(
    "lager-template",
    "LAGER pipeline with multilayer splitter")
def lager_template(**kwargs) -> List[dict]:
    """Create LAGER pipeline configuration."""
    components=PipelineRegistry.get_template("lager-layer-template", **kwargs)
    
    return [create_component("reader", **kwargs),
            create_component("time_remover", **kwargs),
            create_component("splitter", components=components, **kwargs),
            create_component("csv_evaluator", **kwargs)]
   
    


@PipelineRegistry.register(
    "multilayer-graph-feature-template",
    "Multilayer graph representation with feature pipeline"
)
def multilayer_graph_feature_template(**kwargs) -> List[dict]:
    """Create multilayer pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("graph_rep", **kwargs),
        create_component("graph_feature_extractor", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator", **kwargs),
        create_component("graph_evaluator", **kwargs),
    ]


@PipelineRegistry.register(
    "multilayer-graph-recon-template",
    "Multilayer graph representation with reconstruction based learning"
)
def multilayer_graph_recon_template(**kwargs) -> List[dict]:
    """Create multilayer pipeline components."""
    return [
        create_component("reader", **kwargs),
        create_component("graph_rep", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator", **kwargs),
        create_component("graph_evaluator", **kwargs),
    ]

