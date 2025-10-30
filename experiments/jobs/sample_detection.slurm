#!/bin/bash
#SBATCH --job-name=homogenous
#SBATCH --partition=debug
#SBATCH --gpus=3
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --output=experiments/logs/%x_%A/%a.out
#SBATCH --error=experiments/logs/%x_%A/%a.err
#SBATCH --array=0-59%3          # 5 models × 4 embedders × 3 graph_reps = 60 combinations

MODELS=("torch_model.AE" "torch_model.GOAD" "torch_model.SLAD" "torch_model.VAE" "MedianDetector")
EMBEDDER=("PassThroughEmbedder" "GCNEmbedder" "GATEmbedder" "MLPEmbedder")
GRAPH_REPS=("Plain" "Filter" "CDD")

# compute indices
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / ( ${#EMBEDDER[@]} * ${#GRAPH_REPS[@]} ) ))
EMBED_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#GRAPH_REPS[@]}) % ${#EMBEDDER[@]} ))
GRAPH_IDX=$(( SLURM_ARRAY_TASK_ID % ${#GRAPH_REPS[@]} ))

MODEL=${MODELS[$MODEL_IDX]}
EMBED=${EMBEDDER[$EMBED_IDX]}
GRAPH=${GRAPH_REPS[$GRAPH_IDX]}

echo "Task $SLURM_ARRAY_TASK_ID running: MODEL=$MODEL, EMBEDDER=$EMBED, GRAPH=$GRAPH"
echo "Start time: $(date)"

docker run --rm \
  --gpus all \
  -e TQDM_MININTERVAL=60 \
  -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
  -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
  -v "$(pwd)/experiments/scripts":/workspace/intrusion_detection/experiments \
  -w /workspace/intrusion_detection/experiments \
  -u $(id -u):9016 \
  kihy/anidsc_image \
  python3 sample_experiment.py homogeneous test_dataset\
  --config "{\"fe_name\": \"NetworkAccessGraphExtractor\",\"graph_rep\": \"$GRAPH\", \"reader_type\":\"JsonGraphReader\", \"model_name\": \"$MODEL\", \"node_embed\": \"$EMBED\"}"


echo "Completed at $(date)"
