#!/bin/bash

# Record start time
start_time=$(date +%s)

# Define the list of federation types
datasets=("FD") #"HAR" "EEG"
fed_types=("FedAvg" "FedProx" "SCAFFOLD" "MOON" "noFL_multitargets") #"FedMA" "IFCA"

# Wait time between starting each process (in seconds)
wait_time=5

# GPU device ID
gpu_id=0

# Iterate over federation types
for dataset in "${datasets[@]}" 
do
      echo "Running for dataset: $dataset"
      
      for fed_type in "${fed_types[@]}" 
      do  
            echo "Running with fed_type: $fed_type"
            
            # Run each command in background
            nohup python -u trainers/train.py --da_method MAPU --fed_type "$fed_type" --dataset "$dataset" --backbone CNN --num_runs 3 --device cuda:"$gpu_id" > logs/train_log_${dataset}_${fed_type}.out 2>&1 &
            
            echo "----------------------------------------"
            # Wait for the specified wait time
            sleep "$wait_time"
            
            ((gpu_id++))
            # Reset gpu_id to 0 if it becomes greater than 2
            if [ "$gpu_id" -gt 2 ]; then
                  gpu_id=0
            fi
      done
done

# Wait for all background processes to finish
wait

# Calculate total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"
