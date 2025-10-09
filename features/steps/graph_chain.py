from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml

@given("Node Embedder: {node_embed}")
def step_given_feature_extractor(context, node_embed):
    context.pipeline_vars["node_embed"]=node_embed 