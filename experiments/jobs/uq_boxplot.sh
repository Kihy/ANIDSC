#!/bin/bash
set -e

source "$(dirname "$0")/common.sh"

DATASET="uq_dataset"

meta_config="experiments/config/metadata-extraction-template/protocol-meta-extraction.yaml"
fe_config="experiments/config/feature-extraction-template/frequency.yaml"
detection_config="experiments/config/basic-detection-template/boxplot.yaml"

check_path_exists $meta_config
check_path_exists $fe_config
check_path_exists $detection_config


run_experiment run $DATASET --config $meta_config --run_identifier meta_extraction

run_experiment run $DATASET --config $fe_config --prev_pipeline meta_extraction/protocol-meta-extraction  --run_identifier feature-extraction 

run_experiment run $DATASET --config $detection_config --prev_pipeline feature-extraction/frequency --run_identifier detection 