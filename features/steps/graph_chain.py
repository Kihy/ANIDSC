from behave import given, when, then

from ANIDSC import models
from ANIDSC.base_files.evaluator import BaseEvaluator, CollateEvaluator
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.base_files.model import MultilayerSplitter
from ANIDSC.data_source.offline_sources import CSVReader
from ANIDSC.feature_extractors.after_image import AfterImageGraph
from ANIDSC.models.autoencoder import AE
from ANIDSC.models.gnnids import (
    GCNNodeEncoder,
    HomoGraphRepresentation,
    LinearNodeEncoder,
    NodeEncoderWrapper,
)
from ANIDSC.normalizer.t_digest import LivePercentile


@given("a feature extraction pipeline with AfterImageGraph")
def step_given_feature_extraction_pipeline_with_AfterImageGraph(context):
    context.protocols = []
    feature_extractor = AfterImageGraph(context.protocols)
    context.fe_name = f"AfterImageGraph({','.join(context.protocols)})"
    feature_buffer = FeatureBuffer(buffer_size=256)
    context.pipeline = feature_extractor | feature_buffer

@given(
    'a csv file initialized with dataset "{dataset}", file "{file}", and feature extractor AfterImageGraph'
)
def step_given_csv_file(context, dataset, file):
    fe_name="AfterImageGraph(Other)"
    context.data_source = CSVReader(dataset, "AfterImageGraph", fe_name, file)
    context.dataset = dataset
    context.file = file
    context.fe_name = fe_name

@given("a graph NIDS model with {model_name}")
def step_given_basic_NIDS_model_with_AE(context, model_name):
    if model_name == "KitNET":
        preprocessors = []
    else:
        preprocessors = ["to_float_tensor", "to_device"]

    model = getattr(models, model_name)(preprocessors=[])

    node_encoder=LinearNodeEncoder(15, "gaussian")

    encoder_model=NodeEncoderWrapper(node_encoder, model)

    standardizer = LivePercentile()
    graph_rep = HomoGraphRepresentation(preprocessors=preprocessors)

    evaluator = BaseEvaluator(
        [
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )

    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)

    pipeline = standardizer | graph_rep | encoder_model | evaluator

    if hasattr(context, "pipeline"):
        context.pipeline = (
            context.pipeline | MultilayerSplitter(pipeline=pipeline) | collate_evaluator
        )
    else:
        context.pipeline = MultilayerSplitter(pipeline=pipeline) | collate_evaluator
    context.model_name = f"MultlayerSplitter({pipeline})"
