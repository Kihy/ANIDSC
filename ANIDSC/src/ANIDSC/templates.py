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

@ComponentRegistry.register("packet_reader", category="data_source")
def packet_reader(
    dataset_name: str,
    file_name: str,
    **kwargs
) -> dict:
    return {
        "type": "data_source",
        "class": "PacketReader",
        "attrs": {
            "dataset_name": dataset_name,
            "file_name": file_name,
        },
    }

@ComponentRegistry.register("graph_reader", category="data_source")
def graph_reader(
    dataset_name: str,
    file_name: str,
    fe_name: str,
    **kwargs
) -> dict:
    return {
        "type": "data_source",
        "class": "JsonGraphReader",
        "attrs": {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "fe_name": fe_name
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
    folder_name: str,
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "DictFeatureBuffer",
        "attrs": {
            "folder_name": folder_name,
            "buffer_size": buffer_size,
        },
    }
    
        
@ComponentRegistry.register("json_feature_buffer", category="feature_buffer")
def json_feature_buffer(
    folder_name: str,
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "JsonFeatureBuffer",
        "attrs": {
            "folder_name": folder_name,
            "buffer_size": buffer_size,
        },
    }


    
@ComponentRegistry.register("numpy_feature_buffer", category="feature_buffer")
def numpy_feature_buffer(
    folder_name: str,
    buffer_size: int,
    **kwargs
) -> dict:
    return {
        "type": "feature_buffer",
        "class": "NumpyFeatureBuffer",
        "attrs": {
            "folder_name": folder_name,
            "buffer_size": buffer_size,
        },
    }


@ComponentRegistry.register("csv_reader", category="data_source")
def csv_reader(
    dataset_name: str,
    file_name: str,
    fe_name:str,
    **kwargs
) -> dict:
    return {
        "type": "data_source",
        "class": "CSVReader",
        "attrs": {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "fe_name": fe_name
        },
    }


@ComponentRegistry.register("multi_feature_extractor", category="feature_extractor")
def feature_extractor(
    fe_attr: Optional[dict] = None,
    **kwargs
) -> dict:
    return {
        "type": "feature_extractor",
        "class": "MultiLayerGraphExtractor",
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





@ComponentRegistry.register("lp_scaler", category="scaler")
def lp_scaler(**kwargs) -> dict:
    return {"type": "scaler", "class": "LivePercentile"}




@ComponentRegistry.register("od_model", category="model")
def od_model(model_name: str, **kwargs) -> dict:
    return {
        "type": "model",
        "class": "BaseOnlineODModel",
        "attrs": {"model_name": model_name},
    }


@ComponentRegistry.register("csv_evaluator", category="evaluator")
def csv_evaluator(**kwargs) -> dict:
    return {"type": "evaluator", "class": "CSVResultWriter"}

@ComponentRegistry.register("graph_evaluator", category="evaluator")
def graph_evaluator(**kwargs) -> dict:
    return {"type": "evaluator", "class": "GraphResultWriter"}



@ComponentRegistry.register("node_embedder", category="node_encoder")
def node_embedder(embedder: str, **kwargs) -> dict:
    return {
        "type": "node_encoder",
        "class": "BaseNodeEmbedder",
        "attrs": {"model_name": embedder},
    }


@ComponentRegistry.register("graph_rep", category="graph_rep")
def graph_rep(graph_rep: str, **kwargs) -> dict:
    return {
        "type": "graph_rep",
        "class": "GraphProcessor",
        "attrs": {"rep_name": graph_rep},
    }


@ComponentRegistry.register("pipeline", category="pipeline")
def pipeline(manifest: Any, run_identifier: str, **kwargs) -> dict:
    return {
        "type": "pipeline",
        "class": "Pipeline",
        "attrs": {
            "manifest": manifest,
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

            if "run_identifier" not in required:
                required.append("run_identifier")

            cls._templates[name] = {
                "function": func,
                "description": description or func.__doc__ or name,
                "required": required,
            }
            return func
        return decorator

    @classmethod
    def get_template(cls, name: str, **kwargs) -> str:
        if name not in cls._templates:
            raise ValueError(f"Unknown pipeline template '{name}'. Available templates {cls._templates}")

        info = cls._templates[name]
        missing = set(info["required"]) - kwargs.keys()
        if missing:
            raise ValueError(
                f"❌ Pipeline '{name}' missing required parameters: "
                f"{sorted(missing)}"
            )

        components = info["function"](**kwargs)
        pipeline_dict = pipeline(components, kwargs["run_identifier"])
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
# ============================================================================

@PipelineRegistry.register(
    "meta_extraction",
    "Extract metadata from data",
)
def meta_extraction_template(**kwargs) -> List[dict]:
    """Create meta extraction pipeline components."""
    return [
        create_component("packet_reader", **kwargs),
        create_component("protocol_meta_extractor", **kwargs),
        create_component("dict_feature_buffer", folder_name="features", buffer_size=1),
    ]


@PipelineRegistry.register(
    "feature_extraction",
    "Extract features from data",
)
def feature_extraction_template(**kwargs) -> List[dict]:
    """Create feature extraction pipeline components."""

    return [create_component("csv_reader", **kwargs), 
            create_component("feature_extractor", **kwargs),
            create_component("numpy_feature_buffer", 
                            folder_name="features", buffer_size=1)]


@PipelineRegistry.register(
    "graph_feature_extraction",
    "Extract features from data",
)
def graph_feature_extraction(**kwargs) -> List[dict]:
    """Create feature extraction pipeline components."""

    return [create_component("csv_reader", **kwargs), 
                  create_component("multi_feature_extractor", **kwargs),
                  create_component("json_feature_buffer", 
                                 folder_name="features", buffer_size=1)]



@PipelineRegistry.register(
    "basic_detection",
    "Basic anomaly detection pipeline"
)
def basic_detection_template(**kwargs) -> List[dict]:
    """Create basic detection pipeline components."""
    return [
        create_component("graph_reader", **kwargs),
        create_component("lp_scaler", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator"),
    ]


# @PipelineRegistry.register(
#     "lager",
#     "LAGER pipeline with multilayer splitter")
# def lager_template(**kwargs) -> dict:
#     """Create LAGER pipeline configuration."""
#     components = create_component("data_reader", **kwargs)
    
#     # Build inner components
#     inner_components = {}
#     inner_components.update(create_component("scaler", **kwargs))
#     inner_components.update(create_component("graph_rep", **kwargs))
#     inner_components.update(create_component("node_embedder",
#         embedder=kwargs.get("node_encoder", ""), n_features=15, **kwargs
#     ))
#     inner_components.update(create_component("model", **kwargs))
#     inner_components.update(create_component("evaluator", eval_type="CSVResultWriter", 
#                                             graph_period=100, **kwargs))
    
#     # Wrap in pipeline
#     inner_pipeline = create_component("pipeline", manifest=inner_components, 
#                                      run_identifier=kwargs["run_identifier"])
    
#     # Create splitter
#     split_keys = list(ComponentFactory.PROTOCOL_MAP.keys())
#     name = f"{kwargs['node_encoder']}->{kwargs['model_name']}"
#     components.update(create_component("splitter", manifest=inner_pipeline, 
#                                       split_keys=split_keys, name=name, **kwargs))
    
#     return components


# @PipelineRegistry.register(
#     "homogeneous",
#     "Homogeneous graph pipeline"
# )
# def homogeneous_template(**kwargs) -> List[dict]:
#     """Create homogeneous pipeline components."""
#     components = [
#         create_component("data_reader", **kwargs),
#         create_component("graph_rep", **kwargs),
#         create_component("node_embedder", embedder=kwargs["node_embed"], **kwargs),
#     ]
    
#     # Add scaler conditionally
#     needs_scaler = (
#         kwargs["model_name"] != "MedianDetector" and 
#         kwargs["graph_rep"] != "CDD"
#     )
#     if needs_scaler:
#         components.append(create_component("scaler", **kwargs))
    
#     components.extend([
#         create_component("model", **kwargs),
#         create_component("evaluator", eval_type="CSVResultWriter", **kwargs),
#         create_component("evaluator", eval_type="GraphResultWriter", **kwargs),
#     ])
    
#     return components


@PipelineRegistry.register(
    "multilayer-graph-feature",
    "Multilayer graph representation with feature pipeline"
)
def multilayer_template(**kwargs) -> List[dict]:
    """Create multilayer pipeline components."""
    return [
        create_component("graph_reader", **kwargs),
        create_component("graph_rep", **kwargs),
        create_component("graph_feature_extractor", **kwargs),
        create_component("od_model", **kwargs),
        create_component("csv_evaluator", **kwargs),
        create_component("graph_evaluator", **kwargs),
    ]


# Convenience function for backward compatibility
def get_template(template_name: str, **kwargs) -> str:
    return PipelineRegistry.get_template(template_name, **kwargs)
