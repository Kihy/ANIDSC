from behave import given, when, then

from ANIDSC.base_files.evaluator import BaseEvaluator, CollateEvaluator
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.base_files.model import MultilayerSplitter
from ANIDSC.data_source.offline_sources import CSVReader
from ANIDSC.feature_extractors.after_image import AfterImageGraph
from ANIDSC.models.autoencoder import AE
from ANIDSC.models.gnnids import GCNNodeEncoder, HomoGraphRepresentation, NodeEncoderWrapper
from ANIDSC.normalizer.t_digest import LivePercentile


@given('a feature extraction pipeline with AfterImageGraph')
def step_given_feature_extraction_pipeline_with_AfterImageGraph(context):
    context.protocols=["TCP","UDP"]
    feature_extractor = AfterImageGraph(context.protocols)
    context.fe_name="AfterImageGraph"
    feature_buffer = FeatureBuffer(buffer_size=256)
    context.feature_extraction = feature_extractor | feature_buffer



    

@given('a graph NIDS model with AE')
def step_given_basic_NIDS_model_with_AE(context):
    model = AE(
        preprocessors=[],
        profile=False,
    )

    node_encoder=GCNNodeEncoder(10, "gaussian")

    encoder_model=NodeEncoderWrapper(node_encoder, model)
    
    standardizer = LivePercentile()
    graph_rep = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"])
    
   
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )
    
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
    
    context.detection = MultilayerSplitter(
        pipeline=(standardizer | graph_rep | encoder_model | evaluator),
        
    )|collate_evaluator
    
    context.model_name="MultlayerSplitter(LivePercentile-HomoGraphRepresentation-NodeEncoderWrapper(GCNNodeEncoder-AE)-BaseEvaluator)"
    