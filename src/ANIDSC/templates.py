import sys
from typing import Any, Dict, List

import yaml


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


def make_packet_reader(dataset_name: str, file_name: str, **kwargs) -> dict:
    return {
        "data_source": {
            "class": "PacketReader",
            "attrs": {
                "dataset_name": dataset_name,
                "file_name": file_name,
            },
        }
    }


def make_data_reader(
    dataset_name: str,
    file_name: str,
    feature_extractor: str,
    fe_attrs: dict,
    reader_type="CSVReader",
    **kwargs,
) -> dict:
    return {
        "data_source": {
            "class": reader_type,
            "attrs": {
                "dataset_name": dataset_name,
                "file_name": file_name,
                "fe_name": feature_extractor,
                "fe_attrs": fe_attrs,
            },
        }
    }


def make_feature_extractor(feature_extractor: str, **kwargs) -> dict:
    return {"feature_extractor": {"class": feature_extractor}}


def make_meta_extractor(meta_extractor: str, **kwargs) -> dict:
    return {
        "meta_extractor": {
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
    }


def make_feature_buffer(
    buffer_type, folder_name
) -> dict:

    return {
        "feature_buffer": {
            "class": buffer_type,
            "attrs": {
                "folder_name": folder_name,
            },
        }
    }


def make_scaler(**kwargs) -> dict:
    return {"scaler": {"class": "LivePercentile"}}


def make_model(model_name: str, **kwargs) -> dict:

    if model_name == "BoxPlot":
        return {"model": {"class": model_name}}
    else:
        return {
            "model": {"class": "BaseOnlineODModel", "attrs": {"model_name": model_name}}
        }


def make_evaluator(graph_period, **kwargs) -> dict:
    return {
        "evaluator": {"class": "BaseEvaluator", "attrs": {"graph_period": graph_period}}
    }


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


def make_node_encoder(node_encoder, n_features, **kwargs):
    return {
        "model.node_encoder": {
            "class": node_encoder,
            "attrs": {"n_features": n_features},
        }
    }


def make_graph_rep(rep_class="HomoGraphRepresentation", **kwargs):
    return {"graph_rep": {"class": f"{rep_class}"}}


def make_pipeline(manifest: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    return {"class": "Pipeline", "attrs": {"manifest": manifest}}


def dict_to_yaml(pipeline_dict):
    return yaml.safe_dump(
        pipeline_dict, sort_keys=False, default_flow_style=False, indent=2
    )


def make_aggregator(**kwargs):
    return {"custom_processor": {"class": "Aggregator"}}


def get_template(template_name, **kwargs):
    if template_name == "feature_extraction":
        components = make_packet_reader(**kwargs)
        components.update(make_meta_extractor(**kwargs))
        components.update(make_feature_buffer("TabularFeatureBuffer","metadata"))
        components.update(make_feature_extractor(**kwargs))
        components.update(make_feature_buffer("TabularFeatureBuffer","features"))
        pipeline = make_pipeline(components)

    elif template_name == "basic_detection":
        components = make_data_reader(**kwargs)
        components.update(make_scaler(**kwargs))
        components.update(make_model(**kwargs))
        components.update(make_evaluator(graph_period=0, **kwargs))
        pipeline = make_pipeline(components)

    elif template_name == "lager":
        components = make_data_reader(**kwargs)

        inner_components = make_scaler(**kwargs)
        inner_components.update(make_graph_rep(**kwargs))
        inner_components.update(make_node_encoder(n_features=15, **kwargs))
        inner_components.update(make_model(**kwargs))
        inner_components.update(make_evaluator(graph_period=100, **kwargs))
        inner_components = make_pipeline(inner_components)

        split_keys = ["TCP", "UDP", "ICMP", "ARP", "Other"]
        name = f"{kwargs['node_encoder']}->{kwargs['model_name']}"
        components.update(make_splitter(inner_components, split_keys, name, **kwargs))

        pipeline = make_pipeline(components)

    elif template_name == "graph_feature_detection":
        components = make_data_reader(reader_type="JsonGraphReader", **kwargs)
        components.update(
            make_graph_rep(rep_class="AutoScaleGraphRepresentation", **kwargs)
        )
        components.update(make_node_encoder(n_features=2, **kwargs))
        components.update(make_model(**kwargs))
        components.update(make_evaluator(graph_period=1, **kwargs))
        pipeline = make_pipeline(components)

    elif template_name=="homogeneous":
        components = make_data_reader(reader_type="JsonGraphReader", **kwargs)
        components.update(
            make_graph_rep(rep_class="PlainGraphRepresentation", **kwargs)
        )
        components.update(make_model(**kwargs))
        components.update(make_evaluator(graph_period=1, **kwargs))
        pipeline = make_pipeline(components)
    elif template_name=="homogeneous_scaled":
        components = make_data_reader(reader_type="JsonGraphReader", **kwargs)
        components.update(
            make_graph_rep(rep_class="PlainGraphRepresentation", **kwargs)
        )
        components.update(make_scaler())
        components.update(make_model(**kwargs))
        components.update(make_evaluator(graph_period=1, **kwargs))
        pipeline = make_pipeline(components)

    else:
        raise ValueError("Unknown pipeline name")
    return dict_to_yaml(pipeline)
