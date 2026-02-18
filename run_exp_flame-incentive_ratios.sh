#!/bin/bash

# 第一组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior scale" | tee flame-incentives_scale_0.1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_0.1.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior scale" | tee flame-incentives_scale_0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior scale" | tee flame-incentives_scale_0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_0.4.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior scale" | tee flame-incentives_scale_0.5.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_0.5.txt

# 第二组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior signflip" | tee flame-incentives_signflip_0.1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_0.1.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior signflip" | tee flame-incentives_signflip_0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior signflip" | tee flame-incentives_signflip_0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_0.4.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior signflip" | tee flame-incentives_signflip_0.5.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_0.5.txt

# 第3组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior zero" | tee flame-incentives_zero_0.1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_0.1.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior zero" | tee flame-incentives_zero_0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior zero" | tee flame-incentives_zero_0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_0.4.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior zero" | tee flame-incentives_zero_0.5.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_0.5.txt

# 第四组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior noise" | tee flame-incentives_noise_0.1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.1 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_0.1.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior noise" | tee flame-incentives_noise_0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.2 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior noise" | tee flame-incentives_noise_0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.4 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_0.4.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior noise" | tee flame-incentives_noise_0.5.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --mal-frac 0.5 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_0.5.txt
