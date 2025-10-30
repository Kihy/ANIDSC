#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --partition=debug               # Use GPU partition (adjust name as needed)
#SBATCH --gpus=1                      # Request 1 GPU (adjust as needed)
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/logs/%x_%j.out        # Output file (%j = job ID)
#SBATCH --error=experiments/logs/%x_%j.err         # Error file


# Set up environment
export CUDA_VISIBLE_DEVICES=$SLURM_GPUS_ON_NODE


# Run feature extraction
docker run --rm \
    --gpus all \
    -e TQDM_MININTERVAL=60 \
    -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
    -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
    -v "$(pwd)/experiments/scripts":/workspace/intrusion_detection/experiments \
    -w /workspace/intrusion_detection/experiments \
    -u $(id -u):9016  \
    kihy/anidsc_image \
    python3 sample_experiment.py feature_extraction test_dataset --config '{"feature_extractor": "NetworkAccessGraphExtractor", "reader_type": "CSVReader", "fe_name":"ProtocolMetaExtractor"}'

