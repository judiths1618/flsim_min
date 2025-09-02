#!/bin/bash

# 第一组命令
# echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model cnn_mnist --mal-behavior scale" | tee ours_task1.txt
# python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior scale 2>&1 | tee -a ours_cifar10_task1.txt

# 第二组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior signflip" | tee ours_cifar10_task2.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior signflip 2>&1 | tee -a ours_cifar10_task2.txt

# # 第三组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior zero" | tee ours_cifar10_task3.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior zero 2>&1 | tee -a ours_cifar10_task3.txt

# # 第四组命令
echo "python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior noise" | tee ours_cifar10_task4.txt
python flsim/run_experiment_ours.py --config configs/exp_ours.yaml --model resnet18_light --mal-behavior noise 2>&1 | tee -a ours_cifar10_task4.txt

# 第一组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior scale" | tee flame_incentives_cifar10_task1.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior scale 2>&1 | tee -a flame_incentives_cifar10_task1.txt

# 第二组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior signflip" | tee flame_incentives_cifar10_task2.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior signflip 2>&1 | tee -a flame_incentives_cifar10_task2.txt

# 第三组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior zero" | tee flame_incentives_cifar10_task3.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior zero 2>&1 | tee -a flame_incentives_cifar10_task3.txt

# 第四组命令
echo "python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior noise" | tee flame_incentives_cifar10_task4.txt
python flsim/run_experiment_flame_incentives.py --config configs/exp_flame_incentives.yaml --model resnet18_light --mal-behavior noise 2>&1 | tee -a flame_incentives_cifar10_task4.txt