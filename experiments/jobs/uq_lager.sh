#!/bin/bash
set -e

source "$(dirname "$0")/common.sh"

DATASET="uq_dataset"

fe_name="afterimage-graph"

fe_config="experiments/configs/templates/feature-extraction-template/$fe_name.yaml"

detection_template="lager-template"
check_path_exists $fe_config



# uses the same meta data as boxplot, so we can skip it and directly run the feature extraction and detection steps

# run_experiment run $DATASET --config $fe_config --prev_pipeline meta_extraction/protocol-meta-extraction  --run_identifier feature-extraction 

for file in experiments/configs/templates/$detection_template/*; do    
    run_experiment tune $DATASET --config $file --prev_pipeline feature-extraction/$fe_name --run_identifier detection-lager --n_trials 20
done
