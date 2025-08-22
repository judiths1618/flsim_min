# flsim (slim, fixed)

Install:
```bash
pip install -e .[dev,detector,data]
```

Run tests:
```bash
pytest -q tests
```

Run with Flower IID partitions:
```bash
python flsim/run_experiment.py --config configs/exp_default.yaml   --rounds 3 --nodes 8 --malicious_ratio 0.25 --dim 100   --use_flower --dataset cifar10 --iid
```

Run with Flower non-IID (Dirichlet alpha=0.3):
```bash
python flsim/run_experiment.py --config configs/exp_default.yaml   --rounds 3 --nodes 8 --malicious_ratio 0.25 --dim 100   --use_flower --dataset cifar10 --alpha 0.3
```
# 在项目根目录
python -m flsim.run_local_training \
  --config configs/exp_default.yaml \
  --dataset cifar10 --clients 8 --iid \
  --model logreg --epochs 1 --batch 64 --lr 0.1 \
  --rounds 3
# flsim_min
