#!/bin/bash

# 第一组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior scale" | tee flame_incentives_task1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior scale 2>&1 | tee -a flame_incentives_task1.txt

# 第二组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior signflip" | tee flame_incentives_task2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior signflip 2>&1 | tee -a flame_incentives_task2.txt

# 第三组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior zero" | tee flame_incentives_task3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior zero 2>&1 | tee -a flame_incentives_task3.txt

# 第四组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior noise" | tee flame_incentives_task4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-behavior noise 2>&1 | tee -a flame_incentives_task4.txt