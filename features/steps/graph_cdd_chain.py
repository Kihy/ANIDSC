from behave import given, when, then


from ANIDSC.templates import get_pipeline


@given('a {state} cdd graph pipeline with {model_name} and {cdd_framework}')
def step_given_CDD_graph_pipeline_with_AE(context, state, cdd_framework, model_name):
    
    if state == "new":
        load_existing=False 
    else:
        load_existing=[context.dataset, context.fe_name, context.benign_file]
    
    context.pipeline_components.append('cdd')
    
    context.pipeline = get_pipeline(
        pipeline_components=context.pipeline_components,
        pipeline_desc={"fe_cls": "AfterImageGraph", "model_name": model_name,
                       "cdd_framework":cdd_framework,
                       "node_encoder":"LinearNodeEncoder",
                       "distribution":"gaussian"},
        load_existing=load_existing
    )

    context.model_name = model_name