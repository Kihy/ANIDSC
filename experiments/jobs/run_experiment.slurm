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

# Get job ID
job_id="${SLURM_JOB_ID}"

# Load JSON file and add run_identifier
config_content=$(jq --arg job_id "$job_id" '. + {run_identifier: $job_id}' "$config_path")
pipeline_name=$(echo "$config_content" | jq -r '.pipeline_name')


#----------------------------------

# Create log directory
LOG_DIR="experiments/logs/${pipeline_name}/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

exec 1>"${LOG_DIR}/${SLURM_JOB_ID}.out"
exec 2>"${LOG_DIR}/${SLURM_JOB_ID}.err"

echo "Job ID: $job_id"
echo "Config path: $config_path"
echo "Config content: $config_content"
echo "Log directory: $LOG_DIR"
echo "Start time: $(date)"

echo -e "\n ========================================================== \n"

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
    python3 run_experiment.py $DATASET --config "$config_content"


echo -e "\n ========================================================== \n"

echo "End time: $(date)"