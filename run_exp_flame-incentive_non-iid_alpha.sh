#!/bin/bash

# 第一组命令
# echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior scale" | tee flame-incentives_scale_iid.txt
# python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_iid.txt

# echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior scale" | tee flame-incentives_scale_alpha0.2.txt
# python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_alpha0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior scale" | tee flame-incentives_scale_alpha0.3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_alpha0.3.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior scale" | tee flame-incentives_scale_alpha0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior scale 2>&1 | tee -a flame-incentives_scale_alpha0.4.txt

# 第二组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior signflip" | tee flame-incentives_signflip_iid.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_iid.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior signflip" | tee flame-incentives_signflip_alpha0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_alpha0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior signflip" | tee flame-incentives_signflip_alpha0.3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_alpha0.3.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior signflip" | tee flame-incentives_signflip_alpha0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior signflip 2>&1 | tee -a flame-incentives_signflip_alpha0.4.txt

# 第3组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior zero" | tee flame-incentives_zero_iid.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_iid.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior zero" | tee flame-incentives_zero_alpha0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_alpha0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior zero" | tee flame-incentives_zero_alpha0.3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_alpha0.3.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior zero" | tee flame-incentives_zero_alpha0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior zero 2>&1 | tee -a flame-incentives_zero_alpha0.4.txt

# 第四组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior noise" | tee flame-incentives_noise_iid.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --iid --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_iid.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior noise" | tee flame-incentives_noise_alpha_0.2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.2 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_alpha_0.2.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior noise" | tee flame-incentives_noise_alpha_0.3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.3 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_alpha_0.3.txt

echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior noise" | tee flame-incentives_noise_alpha_0.4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model cnn_mnist --alpha 0.4 --mal-behavior noise 2>&1 | tee -a flame-incentives_noise_alpha_0.4.txt
