from behave import given, when, then

from ANIDSC.templates import get_pipeline


@given("a feature extraction pipeline with AfterImage")
def step_given_feature_extraction_pipeline_with_AfterImage(context):
    context.pipeline = get_pipeline(
        pipeline_components=["feature_extraction"],
        pipeline_desc={"fe_cls": "AfterImage"},
    )
    context.fe_name = "AfterImage"
    context.pipeline_components = ["feature_extraction", "detection"]
    
@given("a feature extraction pipeline with AfterImageGraph")
def step_given_feature_extraction_pipeline_with_AfterImageGraph(context):
    context.protocols = ["TCP","UDP","ARP","ICMP"]
    context.fe_name = "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"
    
    context.pipeline = get_pipeline(pipeline_components=["feature_extraction"],
                                    pipeline_desc={"fe_cls": "AfterImageGraph"})
    
    