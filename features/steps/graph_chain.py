from ANIDSC.pipeline import Pipeline
from ANIDSC.templates import get_template
from behave import given, when, then


import os
import yaml

@given("Node Encoder: {node_encoder}")
def step_given_feature_extractor(context, node_encoder):
    context.custom_vars["node_encoder"]=node_encoder 