#!/bin/bash
set -e

source "$(dirname "$0")/common.sh"

JOB_ID="base"

DATASET="test_dataset"


run_experiment run $DATASET --config experiments/configs/test_configs/meta_extraction.yaml
run_experiment run $DATASET --config experiments/configs/test_configs/graph_feature_extraction.yaml
run_experiment tune $DATASET --config experiments/configs/test_configs/structural_gae.yaml --hyperparam_spec experiments/hyper_specs/structural_gae.yaml 