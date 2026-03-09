#!/bin/bash
set -e

source "$(dirname "$0")/common.sh"

DATASET="uq_dataset"


fe_config="experiments/configs/templates/feature-extraction-template/afterimage.yaml"

check_path_exists $fe_config



# uses the same meta data as boxplot, so we can skip it and directly run the feature extraction and detection steps

# run_experiment run $DATASET --config $fe_config --prev_pipeline meta_extraction/protocol-meta-extraction  --run_identifier feature-extraction 

for file in experiments/configs/templates/scaled-detection-template/*; do
    [[ "$file" == *"AE"* || "$file" == *"GOAD.yaml"* || "$file" == *"ICL.yaml"* ]] && continue
    run_experiment tune $DATASET --config $file --prev_pipeline feature-extraction/afterimage --run_identifier detection --n_trials 20
done
