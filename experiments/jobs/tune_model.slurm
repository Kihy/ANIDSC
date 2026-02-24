#!/bin/bash
#SBATCH --job-name=experiments
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# Get command and config path from positional arguments
DATASET="$1"
config_path="$2"
hyper_path="$3"

# Pass JSON content as string directly
docker run --rm \
    --gpus device=0 \
    -e TQDM_MININTERVAL=60 \
    -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
    -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
    -v "$(pwd)/experiments/scripts":/workspace/intrusion_detection/experiments \
    -w /workspace/intrusion_detection/experiments \
    -u $(id -u):$(id -g) \
    kihy/anidsc_image \
    python3 tune_model.py $DATASET --config "$config_content" --hyperparam_spec "$hyperparam_content"

