
from __future__ import annotations
import argparse
import numpy as np
import random

from flsim.config import build_contract_from_yaml
from flsim.train.local import train_locally_on_partitions
from flsim.data.flower import load_flower_arrays
# from flsim.aggregation.base import _zeros_like
from flsim.aggregation.flame import AggregationStrategy as _Flame  # typing only
from flsim.attack.malicious import choose_malicious_nodes, apply_malicious_updates
from flsim.eval.global_evaluation import evaluate_global_on_flower, evaluate_global_params
# 使用示例
from flsim.aggregation.base import AggregationStrategy


def main():
    ap = argparse.ArgumentParser(description="Local model training on Flower partitions (keeps original logic intact)")
    ap.add_argument("--config", required=True, help="Path to YAML config for detection/aggregation/etc.")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=3)

    # —— 新增恶意相关参数 ——
    ap.add_argument("--mal-frac", type=float, default=0.2, help="恶意节点比例（0~1）")
    ap.add_argument("--mal-ids", type=str, default="", help="显式恶意节点ID，逗号分隔，如 1,3,5")
    ap.add_argument("--mal-behavior", type=str, default="scale",
                    choices=["scale", "signflip", "zero", "noise"])
    ap.add_argument("--mal-scale", type=float, default=-10.0, help="scale 行为的缩放因子")
    ap.add_argument("--mal-noise-std", type=float, default=0.1, help="noise 行为的噪声标准差")
    
    ap.add_argument("--defense", default="flame")
    args = ap.parse_args()

    # Build composed contract from YAML (unchanged)
    # contract = build_contract_from_yaml(args.config)

    contract = build_contract_from_yaml(args.config)
    nodes=args.clients

    print("nodes", nodes)

    # Register nodes with initial stake and reputation
    for nid in range(1, nodes+1):
        contract.register_node(nid, stake=100.0, reputation=50.0)
    

    # Load client partitions (X, y)
    Xp, yp, D, K, X_eval, y_eval = load_flower_arrays(
        dataset=args.dataset,
        n_clients=args.clients,
        iid=args.iid,
        alpha=args.alpha,
        split=args.split,
        flatten=True,
        normalize=True,
        # 关键：所有客户端与评估集共享同一组全局 min/max
        normalization="global",
    )

    # set malicious nodes
    explicit = {int(x) for x in args.mal_ids.split(",") if x.strip()} if args.mal_ids else None
    true_mal = choose_malicious_nodes(
        all_node_ids= [client_id for client_id in Xp.keys()],
        mal_frac=float(args.mal_frac),
        explicit_ids=explicit,
        seed=42,
    )
    print(f"[Info] Malicious nodes: {sorted(true_mal)} (behavior={args.mal_behavior})")


    # Initialize global parameters as None (the trainer will create them based on model)
    global_params = None
    # run_local_training.py 的训练循环里，在第一次调用 run_round 之前


    results = []
    for r in range(1, args.rounds + 1):

        
        # Local training per client on their partition
        updates, reference_global = train_locally_on_partitions(
            model_name=args.model,
            partitions={cid: (Xp[cid], yp[cid], X_eval, y_eval) for cid in Xp},
            global_params=global_params,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=42 + r,
        )
        # print(reference_global)
        print()
                # Ensure the contract's baseline matches the parameters used for training
        if contract.prev_global is None:
            contract.prev_global = reference_global
        else:
            contract.prev_global = global_params

        # if r == 1 and getattr(contract, "prev_global", None) is None:
        #     contract.prev_global = {k: v.copy() for k, v in reference_global.items()}

        for u in updates:
            m = evaluate_global_params(args.model, u.params, X_eval, y_eval)
            print(f"client {u.node_id} tested: acc={m['acc']:.4f}, loss={m['loss']:.4f}")

        # 初始化聚合策略
        fedavg = AggregationStrategy(debug=True)

        # # 训练并获取更新
        # updates, global_params = train_locally_on_partitions(
        #     model_name=args.model,
        #     partitions={cid: (Xp[cid], yp[cid]) for cid in Xp},
        #     global_params=global_params,
        #     epochs=10,
        #     batch_size=32
        # )

        # 聚合更新
        aggregated_params = fedavg.aggregate(updates)

        # res = contract.run_round(r, updates=updates, true_malicious=true_mal)
        # global_params = res["global_params"]
        contract.prev_global = aggregated_params
        # print(f"Round {r} results: {res}")
        # contract.prev_global = global_params
        mG = evaluate_global_params(args.model, global_params, X_eval, y_eval)
        print(f"GLOBAL: acc={mG['acc']:.4f}, loss={mG['loss']:.4f}")

if __name__ == "__main__":
    main()
