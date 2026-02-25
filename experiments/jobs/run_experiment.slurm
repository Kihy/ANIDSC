#!/bin/bash
#SBATCH --job-name=experiments
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# Get command and config path from positional arguments
script="$1"
shift

# Pass JSON content as string directly
docker run --rm \
    --gpus device=0 \
    -e TQDM_MININTERVAL=60 \
    -e SLURM_JOB_ID="$SLURM_JOB_ID" \
    -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
    -v "$(pwd)/runs":/workspace/intrusion_detection/runs \
    -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
    -v "$(pwd)/experiments/":/workspace/intrusion_detection/experiments \
    -w /workspace/intrusion_detection \
    -u $(id -u):$(id -g) \
    kihy/anidsc_image \
    python3 "$script" "$@"

