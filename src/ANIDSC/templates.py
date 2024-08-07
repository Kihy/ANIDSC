
from . import cdd_frameworks, models, feature_extractors
from .base_files import BaseEvaluator, CollateEvaluator, FeatureBuffer, MultilayerSplitter

from .models.gnnids import HomoGraphRepresentation, NodeEncoderWrapper
from .normalizer import LivePercentile


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

def get_pipeline(
    pipeline_components=["feature_extraction","detection"],
    pipeline_desc={"fe_cls":"AfterImage", "model_name":"AE"},
    load_existing=False
):

    def load_or_create(class_type, folder, name=None, **kwargs):
        if load_existing:
            return class_type.load_pickle(folder, *load_existing, name or load_existing[1])
        return class_type(**kwargs)
    
    if "feature_extraction" in pipeline_components:
        extractor_class = getattr(feature_extractors, pipeline_desc["fe_cls"])
        feature_extractor = load_or_create(extractor_class, "feature_extractors")
        feature_buffer = FeatureBuffer(buffer_size=256)
        pipeline = feature_extractor | feature_buffer
    else:
        pipeline = None

    if "detection" in pipeline_components:
        preprocessors = [] if "Graph" in pipeline_desc["fe_cls"] else ["to_float_tensor", "to_device"]

        model = load_or_create(getattr(models, pipeline_desc["model_name"]), "models",name=pipeline_desc["model_name"], preprocessors=preprocessors, profile=False)

        if "cdd" in pipeline_components:
            model = load_or_create(getattr(cdd_frameworks, pipeline_desc["cdd_type"]), "models", model)

        if "Graph" in pipeline_desc["fe_cls"]:
            evaluator = BaseEvaluator(
            METRICS,
            log_to_tensorboard=False,
            save_results=False,
            draw_graph_rep_interval=100,
        )
            if load_existing:
                protocol_splitter = load_or_create(
                    MultilayerSplitter,
                    "models",
                    name=f"MultlayerSplitter(LivePercentile-HomoGraphRepresentation-NodeEncoderWrapper({pipeline_desc['node_encoder']}({pipeline_desc['distribution']})-{pipeline_desc['model_name']})-BaseEvaluator)"
                )
            else:
                node_encoder = getattr(models, pipeline_desc['node_encoder'])(15, pipeline_desc['distribution'])
                encoder_model = NodeEncoderWrapper(node_encoder, model)
                graph_rep = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"])
                standardizer = LivePercentile()
                protocol_splitter = MultilayerSplitter(
                    pipeline=(standardizer | graph_rep | encoder_model | evaluator)
                )
            collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
            
            if pipeline is not None:
                pipeline = pipeline | protocol_splitter | collate_evaluator
            else:
                pipeline = protocol_splitter | collate_evaluator
        else:
            evaluator = BaseEvaluator(
            METRICS,
            log_to_tensorboard=True,
            save_results=True,
            draw_graph_rep_interval=100,
        )
            standardizer = load_or_create(LivePercentile, "scalers", "LivePercentile")
            if pipeline is not None:
                pipeline = pipeline |standardizer| model | evaluator
            else:
                pipeline=standardizer| model | evaluator

    return pipeline
