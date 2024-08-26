from behave import given, when, then

from ANIDSC.data_source.offline_sources import CSVReader
from ANIDSC.templates import get_pipeline



@given(
    'a csv file initialized with dataset "{dataset}", file "{file}", and feature extractor AfterImageGraph'
)
def step_given_csv_file(context, dataset, file):
    fe_name=f"AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"
    context.data_source = CSVReader(dataset, "AfterImageGraph", fe_name, file)
    context.dataset = dataset
    context.file = file
    context.fe_name = fe_name
    context.benign_file="benign_lenovo_bulb"
    context.pipeline_components=["detection"]

@given("a {state} graph pipeline with {model_name}")
def step_given_graph_NIDS_model_with_AE(context, state, model_name):
    if state=="new":
        load_existing=False
    else:
        load_existing=[context.dataset, context.fe_name, context.benign_file]
    
    context.pipeline = get_pipeline(
        pipeline_components=context.pipeline_components,
        pipeline_desc={"fe_cls": "AfterImageGraph", "model_name": model_name,
                       "node_encoder":"LinearNodeEncoder",
                       "distribution":"gaussian"},
        load_existing=load_existing
    )

    context.model_name = model_name