from behave import given, when, then

from ANIDSC import models
from ANIDSC.base_files.evaluator import BaseEvaluator
from ANIDSC.base_files.model import ConceptDriftWrapper
from ANIDSC.models.autoencoder import AE
from ANIDSC.normalizer.t_digest import LivePercentile

@given('a CDD NIDS model with {model_name}')
def step_given_CDD_NIDS_model_with_AE(context, model_name):
    if model_name=="KitNET":
        preprocessors=[]
    else:
        preprocessors=["to_float_tensor", "to_device"]
        
    model = getattr(models, model_name)(
        preprocessors=preprocessors
    )
    
    if model_name=="ARCUS":
        cd_model=model 
        context.model_name="ARCUS"
    else:
        cd_model=ConceptDriftWrapper(model, 1000, 50)
        context.model_name=f"CDD({model_name})"

    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
    )
    context.detection = standardizer | cd_model | evaluator
    