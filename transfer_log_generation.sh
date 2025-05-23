#!/bin/bash

# Parent directory containing subdirectories
PARENT_DIR="/app/benchmark_data/lc_bench/"

# Iterate over subdirectories in the parent directory
for SUBDIR in "$PARENT_DIR"/*/; do
    # Extract the name of the subdirectory
    TASK=$(basename "$SUBDIR")
    
    # Construct the command
    COMMAND="numactl -C \"16-25\" python main.py --optimizer pc --exp lc --task ${TASK} --log-dir ./data/htl-data/lc_${TASK}_pc/ --num-experiments 5 --num-procs 1"
    
    # Print the command being executed
    echo "Executing: $COMMAND"
    
    # Execute the command and wait for it to finish
    eval "$COMMAND"
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Command failed for task: ${TASK}"
        exit 1
    fi
done
