#!/bin/bash

# 第一组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior scale" | tee ours_task1.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior scale 2>&1 | tee -a ours_task1.txt

# 第二组命令
# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior signflip" | tee ours_task2.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior signflip 2>&1 | tee -a ours_task2.txt

# # 第三组命令
# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior zero" | tee ours_task3.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior zero 2>&1 | tee -a ours_task3.txt

# # 第四组命令
# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior noise" | tee ours_task4.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior noise 2>&1 | tee -a ours_task4.txt