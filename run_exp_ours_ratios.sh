#!/bin/bash

# 第一组命令
# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior scale" | tee ours_task1_0.1.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior scale 2>&1 | tee -a ours_task1_0.1.txt

# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior scale" | tee ours_task1_0.2.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior scale 2>&1 | tee -a ours_task1_0.2.txt

# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior scale" | tee ours_task1_0.4.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior scale 2>&1 | tee -a ours_task1_0.4.txt

# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior scale" | tee ours_task1_0.5.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior scale 2>&1 | tee -a ours_task1_0.5.txt

# 第二组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior signflip" | tee ours_signflip_task1_0.1.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior signflip 2>&1 | tee -a ours_signflip_task1_0.1.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior signflip" | tee ours_signflip_task1_0.2.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior signflip 2>&1 | tee -a ours_signflip_task1_0.2.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior signflip" | tee ours_signflip_task1_0.4.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior signflip 2>&1 | tee -a ours_signflip_task1_0.4.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior signflip" | tee ours_signflip_task1_0.5.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior signflip 2>&1 | tee -a ours_signflip_task1_0.5.txt

# 第3组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior zero" | tee ours_zero_task1_0.1.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior zero 2>&1 | tee -a ours_zero_task1_0.1.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior zero" | tee ours_zero_task1_0.2.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior zero 2>&1 | tee -a ours_zero_task1_0.2.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior zero" | tee ours_zero_task1_0.4.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior zero 2>&1 | tee -a ours_zero_task1_0.4.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior zero" | tee ours_zero_task1_0.5.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior zero 2>&1 | tee -a ours_zero_task1_0.5.txt

# 第四组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior noise" | tee ours_noise_task1_0.1.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior noise 2>&1 | tee -a ours_noise_task1_0.1.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior noise" | tee ours_noise_task1_0.2.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior noise 2>&1 | tee -a ours_noise_task1_0.2.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior noise" | tee ours_noise_task1_0.4.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior noise 2>&1 | tee -a ours_noise_task1_0.4.txt

echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior noise" | tee ours_noise_task1_0.5.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior noise 2>&1 | tee -a ours_noise_task1_0.5.txt
