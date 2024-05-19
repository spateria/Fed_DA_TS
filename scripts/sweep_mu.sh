#!/bin/bash

dataset=EEG

fed_type=MOON

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --moon_mu 0.01 --backbone CNN --num_runs 3 --device cuda:0 > logs/train_log_${dataset}_${fed_type}_1.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --moon_mu 0.1 --backbone CNN --num_runs 3 --device cuda:0 > logs/train_log_${dataset}_${fed_type}_2.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --moon_mu 1.0 --backbone CNN --num_runs 3 --device cuda:1 > logs/train_log_${dataset}_${fed_type}_3.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5

fed_type=FedProx

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --fedprox_mu 0.001 --backbone CNN --num_runs 3 --device cuda:1 > logs/train_log_${dataset}_${fed_type}_4.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --fedprox_mu 0.01 --backbone CNN --num_runs 3 --device cuda:2 > logs/train_log_${dataset}_${fed_type}_5.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5

nohup python -u trainers/train.py --da_method MAPU --fed_type $fed_type --dataset $dataset --fedprox_mu 0.1 --backbone CNN --num_runs 3 --device cuda:2 > logs/train_log_${dataset}_${fed_type}_6.out 2>&1 &
echo "----------------------------------------"
# Wait for the specified wait time
sleep 5
