from behave import given, when, then
from ANIDSC import models
from ANIDSC.adversarial_attacks.liuer_mihou import LiuerMihouAttack
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.feature_extractors.after_image import AfterImage
from ANIDSC.normalizer.t_digest import LivePercentile


@given("a Liuer Mihou attack with trained '{nids}' model and AfterImage feature extractor from '{file}'")
def step_given_LM(context, nids, file):
    fe_name="AfterImage"
    model = getattr(models, nids).load_pickle(
            "models", context.dataset, fe_name, file, nids
        )
    
    standardizer = LivePercentile.load_pickle(
            "scalers", context.dataset, fe_name, file, "LivePercentile"
        )
    
    feature_extractor = AfterImage.load_pickle(
                "feature_extractors", context.dataset, fe_name, file, fe_name
            )
    feature_buffer = FeatureBuffer(buffer_size=256)
    
    lm_attack=LiuerMihouAttack(feature_extractor, model, standardizer)
    
    context.pipeline = lm_attack | feature_extractor | feature_buffer
    
    
    