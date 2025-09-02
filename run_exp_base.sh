#!/bin/bash

# 第一组命令
echo "python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior scale" | tee base_task1.txt
python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior scale 2>&1 | tee -a base_task1.txt

# 第二组命令
echo "python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior signflip" | tee base_task2.txt
python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior signflip 2>&1 | tee -a base_task2.txt

# 第三组命令
echo "python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior zero" | tee base_task3.txt
python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior zero 2>&1 | tee -a base_task3.txt

# 第四组命令
echo "python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior noise" | tee base_task4.txt
python flsim/run_experiment_base.py --config configs/exp_base.yaml --model cnn_mnist --mal-behavior noise 2>&1 | tee -a base_task4.txt