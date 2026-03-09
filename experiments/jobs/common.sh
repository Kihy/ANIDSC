#!/bin/bash

run_experiment() {
    local command="$1"
    shift

    docker run --rm \
        --gpus all \
        -e TQDM_MININTERVAL=60 \
        -e SLURM_JOB_ID="$JOB_ID" \
        -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
        -v "$(pwd)/runs":/workspace/intrusion_detection/runs \
        -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
        -v "$(pwd)/experiments":/workspace/intrusion_detection/experiments \
        -w /workspace/intrusion_detection \
        -u "$(id -u)":"$(id -g)" \
        kihy/anidsc_image \
        python3 experiments/scripts/main.py "$command" "$@"
}

check_path_exists() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Error: path does not exist: $path"
        exit 1
    fi
}
