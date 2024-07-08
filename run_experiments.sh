#!/bin/bash

# Define the array of batch sizes and learning rates you want to iterate over
batch_sizes=(32)
decay_db_per_mm_per_mhz=(0.1)
learning_rates=(0.00001)

# Path to your Python script
script_path="config.py"

for batch_size in "${batch_sizes[@]}"; do
	for lr in "${learning_rates[@]}"; do
		for decay in "${decay_db_per_mm_per_mhz[@]}"; do
	        echo "Running experiment with batch_size=${batch_size}, learning_rate=${lr}, decay=${decay}"

			# Use sed to modify batch_size  in the script
			sed -i "s/^batch_size = .*/batch_size = ${batch_size}/" $script_path
			sed -i "s/^decay_db_per_mm_per_mhz = .*/decay_db_per_mm_per_mhz = ${decay}/" $script_path
			sed -i "s/^model_lr = .*/model_lr = ${lr}/" $script_path
			
			# Run the training script
			# python train.py
			
			# Calculate log_decay
			log_decay=$(echo "$decay * 100 / 1" | bc)
			log_lr=$(echo "$lr * 100000 / 1" | bc)
			export LEARNING_RATE=${lr}
			
			log_file="results/batch_${batch_size}_lr_${log_lr}_decay_${log_decay}.log"    
			# Run the training script in the background using nohup
            # nohup python train.py > $log_file 2>&1 &
			python train.py > $log_file
		done
	done
done

# Wait for all background processes to finish
wait

echo "All experiments are completed."