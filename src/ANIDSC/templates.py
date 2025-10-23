import sys
from typing import Any, Dict, List


import yaml


def make_data_reader(
    reader_type,
    dataset_name: str,
    file_name: str,
    fe_name: str = "",
    **kwargs,
) -> dict:
    return {
        "type": "data_source",
        "class": reader_type,
        "attrs": {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "fe_name": fe_name,
        },
    }


def make_feature_extractor(feature_extractor: str, **kwargs) -> dict:
    return {"type": "feature_extractor", "class": feature_extractor}


def make_meta_extractor(meta_extractor: str, **kwargs) -> dict:
    return {
        "type": "meta_extractor",
        "class": meta_extractor,
        "attrs": {
            "protocol_map": {
                "TCP": 0,
                "UDP": 1,
                "ICMP": 2,
                "ARP": 3,
                "Other": 4,
            }
        },
    }


def make_feature_buffer(buffer_type, folder_name, buffer_size) -> dict:

    return {
        "type": "feature_buffer",
        "class": buffer_type,
        "attrs": {"folder_name": folder_name, "buffer_size": buffer_size},
    }


def make_scaler(**kwargs) -> dict:
    return {"type": "scaler", "class": "LivePercentile"}


def make_model(model_name: str, **kwargs) -> dict:

    if model_name == "BoxPlot":
        return {"type": "model", "class": model_name}
    else:
        return {
            "type": "model",
            "class": "BaseOnlineODModel",
            "attrs": {"model_name": model_name},
        }


def make_evaluator(eval_type, **kwargs) -> dict:
    return {"type": "evaluator", "class": eval_type}


def make_splitter(manifest, split_keys, name, **kwargs):
    return {
        "pipeline": {
            "class": "MultilayerSplitter",
            "attrs": {
                "manifest": {key: manifest for key in split_keys},
                "name": name,
                "protocol_map": {key: i for i, key in enumerate(split_keys)},
            },
        }
    }


def make_node_embedder(embedder, **kwargs):
    return {
        "type": "node_encoder",
        "class": "BaseNodeEmbedder",
        "attrs": {"model_name": embedder},
    }


def make_graph_rep(**kwargs):
    return {
        "type": "graph_rep",
        "class": "GraphProcessor",
        "attrs": {"rep_name": kwargs["graph_rep"]},
    }


def make_pipeline(manifest) -> Dict[str, Any]:
    return {"type": "pipeline", "class": "Pipeline", "attrs": {"manifest": manifest}}


def dict_to_yaml(pipeline_dict):
    return yaml.safe_dump(
        pipeline_dict, sort_keys=False, default_flow_style=False, indent=2
    )


def make_aggregator(**kwargs):
    return {"custom_processor": {"class": "Aggregator"}}


def get_template(template_name, **kwargs):
    components = []
    if template_name == "meta_extraction":

        components.append(make_data_reader(**kwargs))
        components.append(make_meta_extractor(**kwargs))
        components.append(
            make_feature_buffer("DictFeatureBuffer", "features", buffer_size=1)
        )
        pipeline = make_pipeline(components)

    elif template_name == "feature_extraction":
        components.append(make_data_reader(**kwargs))
        components.append(make_feature_extractor(**kwargs))

        if "Graph" in kwargs["feature_extractor"]:
            components.append(
                make_feature_buffer("JsonFeatureBuffer", "features", buffer_size=1)
            )
        else:
            components.append(
                make_feature_buffer("NumpyFeatureBuffer", "features", buffer_size=1024)
            )
        pipeline = make_pipeline(components)

    elif template_name == "basic_detection":
        components.append(make_data_reader(**kwargs))
        components.append(make_scaler(**kwargs))
        components.append(make_model(**kwargs))
        components.append(make_evaluator("CSVResultWriter"))
        pipeline = make_pipeline(components)

    elif template_name == "lager":
        components = make_data_reader(**kwargs)

        inner_components = make_scaler(**kwargs)
        inner_components.update(make_graph_rep(**kwargs))
        inner_components.update(make_node_embedder(n_features=15, **kwargs))
        inner_components.update(make_model(**kwargs))
        inner_components.update(make_evaluator(graph_period=100, **kwargs))
        inner_components = make_pipeline(inner_components)

        split_keys = ["TCP", "UDP", "ICMP", "ARP", "Other"]
        name = f"{kwargs['node_encoder']}->{kwargs['model_name']}"
        components.update(make_splitter(inner_components, split_keys, name, **kwargs))

        pipeline = make_pipeline(components)

    elif template_name == "homogeneous":

        components.append(make_data_reader(**kwargs))
        components.append(
            make_graph_rep(
               **kwargs
            )
        )

        components.append(make_node_embedder(embedder=kwargs["node_embed"], **kwargs))

        # add scaler, no need for median dectector
        if kwargs["model_name"] != "MedianDetector":
            # no need for CDD
            if kwargs["graph_rep"]!= "CDD":
                components.append(make_scaler(**kwargs))

        components.append(make_model(**kwargs))
        components.append(make_evaluator("CSVResultWriter", **kwargs))
        components.append(make_evaluator("GraphResultWriter", **kwargs))
        pipeline = make_pipeline(components)
    else:
        raise ValueError("Unknown pipeline name")
    return dict_to_yaml(pipeline)
