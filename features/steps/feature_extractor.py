from behave import *
from src.ANIDSC.templates import extract_network_features

@given('we want to extract features from {filename} in {dataset} dataset')
def step_impl(context, filename, dataset):
    context.filename=filename
    context.dataset=dataset 
    
@when('The feature extractor is AfterImageGraph with {state} information')
def step_impl(context, state):
    extract_network_features(
            context.dataset,
            "AfterImageGraph",
            {
                "graph_type": "multi_layer",
                "protocols": ["TCP", "DNS", "SSH", "FTP", "HTTP","UDP","HTTPS"],
            },
            context.filename,
            state=eval(state),
        )

@then('The process should not report error')
def step_impl(context):
    assert context.failed is False


    
    