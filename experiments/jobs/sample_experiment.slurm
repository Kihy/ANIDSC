#!/bin/bash
#SBATCH --job-name=experiments
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Usage: YOU MUST REPLACE META_RUN_ID AND FE_RUN_ID with the actual RUN_ID. 
#        You can also change --array=1-3 to selectively run jobs
#        sbatch --export=EXP_TYPE=meta_extraction experiments/jobs/sample_experiment.slurm       
#        sbatch --array=0-2 --export=EXP_TYPE=feature_extraction,META_RUN_ID=1256 experiments/jobs/sample_experiment.slurm
#        sbatch --array=0-29 --export=EXP_TYPE=detection,FE_RUN_ID=1257 experiments/jobs/sample_experiment.slurm

# ---------------------------------
# CUSTOM PARAMETERS
DATASET="test_dataset"
RUN_SCRIPT="sample_experiment.py"

#----------------------------------

# Create log directory
LOG_DIR="experiments/logs/${EXP_TYPE}/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Redirect logs
TASK_ID=${SLURM_ARRAY_TASK_ID:-$SLURM_JOB_ID}
exec 1>"${LOG_DIR}/${TASK_ID}.out"
exec 2>"${LOG_DIR}/${TASK_ID}.err"

echo "Experiment type: $EXP_TYPE"
echo "Task ID: $TASK_ID"
echo "Start time: $(date)"

# Common Docker run function
run_docker() {
    local cmd=$1
    local config=$2
    
    docker run --rm \
        --gpus device=0 \
        -e TQDM_MININTERVAL=60 \
        -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
        -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
        -v "$(pwd)/experiments/scripts":/workspace/intrusion_detection/experiments \
        -w /workspace/intrusion_detection/experiments \
        -u $(id -u):$(id -g) \
        kihy/anidsc_image \
        python3 $RUN_SCRIPT $cmd $DATASET --config "$config"
}

# Generate all combinations and select by index
generate_combinations() {
    local -n arrays=$1
    local result=()
    
    eval "
    for combo in $(printf '{%s}..' "${arrays[@]}"); do
        result+=(\"\$combo\")
    done
    "
    printf '%s\n' "${result[@]}"
}

# Execute based on experiment type
case $EXP_TYPE in
    detection)
        # Define all parameter arrays
        MODELS=("MedianDetector" "torch_model.AE" "torch_model.GOAD" "torch_model.SLAD" "torch_model.VAE")
        EMBEDDERS=("PassThroughEmbedder")
        GRAPH_REPS=("CDD" "Plain")
        LAYERS=("physical" "internet" "transport")
        
        # Generate all combinations
        combinations=()
        for model in "${MODELS[@]}"; do
            for embedder in "${EMBEDDERS[@]}"; do
                for graph in "${GRAPH_REPS[@]}"; do
                    for layer in "${LAYERS[@]}"; do
                        combinations+=("$model|$embedder|$graph|$layer")
                    done
                done
            done
        done
                
        echo "Total combinations: ${#combinations[@]}"

        # Select combination for this task
        IFS='|' read -r MODEL EMBED GRAPH LAYER <<< "${combinations[$SLURM_ARRAY_TASK_ID]}"
        
        echo "MODEL=$MODEL, EMBEDDER=$EMBED, GRAPH=$GRAPH, LAYER=$LAYER"
        
        read -r -d '' CONFIG << EOF
{
  "fe_name": "SingleLayerGraphExtractor($LAYER)-$FE_RUN_ID",
  "graph_rep": "$GRAPH",
  "reader_type": "JsonGraphReader",
  "model_name": "$MODEL",
  "node_embed": "$EMBED",
  "run_identifier": "${SLURM_ARRAY_JOB_ID}"
}
EOF
        run_docker "homogeneous" "$CONFIG"
        ;;
        
    meta_extraction)
        read -r -d '' CONFIG << EOF
{
  "meta_extractor": "ProtocolMetaExtractor",
  "reader_type": "PacketReader",
  "run_identifier": "${SLURM_JOB_ID}"
}
EOF
        run_docker "meta_extraction" "$CONFIG"
        ;;
        
    feature_extraction)
        # Generate all combinations (in case you want to add more parameters later)
        LAYERS=("physical" "internet" "transport")
        
        combinations=()
        for layer in "${LAYERS[@]}"; do
            combinations+=("$layer")
        done
        
        # Select combination for this task
        LAYER="${combinations[$SLURM_ARRAY_TASK_ID]}"
        
        echo "LAYER=$LAYER"
        
        read -r -d '' CONFIG << EOF
{
  "feature_extractor": "SingleLayerGraphExtractor",
  "fe_attr": {
    "layer": "$LAYER"
  },
  "reader_type": "CSVReader",
  "fe_name": "ProtocolMetaExtractor-$META_RUN_ID",
  "run_identifier": "${SLURM_ARRAY_JOB_ID}"
}
EOF
        run_docker "feature_extraction" "$CONFIG"
        ;;
        
    *)
        echo "Error: Unknown experiment type '$EXP_TYPE'"
        echo "Valid types: detection, meta_extraction, feature_extraction"
        exit 1
        ;;
esac

echo "Completed at $(date)"