from behave import given, when, then

from ANIDSC.base_files.evaluator import BaseEvaluator
from ANIDSC.base_files.model import ConceptDriftWrapper
from ANIDSC.models.autoencoder import AE
from ANIDSC.normalizer.t_digest import LivePercentile

@given('a CDD NIDS model with AE')
def step_given_CDD_NIDS_model_with_AE(context):
    model = AE(
        preprocessors=["to_float_tensor", "to_device"],
    )
    
    cd_model=ConceptDriftWrapper(model, 1000, 50)
    standardizer = LivePercentile()
   
    evaluator = BaseEvaluator(
        ["detection_rate", "average_score", "average_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=10,
    )
    context.detection = standardizer | cd_model | evaluator
    context.model_name="CDD(AE)"
