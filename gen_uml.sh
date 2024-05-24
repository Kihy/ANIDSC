#!/bin/bash

# Define the root directory where the folders are located
ROOT_DIR="src"
OUTPUT_DIR="docs/class_diagrams"

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each folder in the root directory
for dir in "$ROOT_DIR"/*/; do
  # Check if the directory is a folder
  if [ -d "$dir" ] && [ "$(basename "$dir")" != "__pycache__" ]; then
    # Get the base name of the directory
    folder_name=$(basename "$dir")
    echo "Running pyreverse on folder: $folder_name"
    
    # Run pyreverse on the directory and specify the output directory
    pyreverse -o dot -p "$folder_name" -d "$OUTPUT_DIR" "$dir"

    if [ $? -eq 0 ]; then
      echo "Successfully generated diagram for $folder_name"
    else
      echo "Error running pyreverse on $folder_name"
    fi
  fi
done