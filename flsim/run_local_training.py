
#!/usr/bin/env python3
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
        normalize=False,
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
        if r == 1 and getattr(contract, "prev_global", None) is None:
            contract.prev_global = {k: v.copy() for k, v in reference_global.items()}

        if global_params is None and r==1:
            # Optional: evaluate the initial reference global before first aggregation
            base = evaluate_global_params(args.model, reference_global,
                                            X_eval, y_eval)
            print(f"[Eval] Base global (round 0): acc={base['acc']:.4f}, loss={base['loss']:.4f}")
    
        # 
        if true_mal is not None: 
            # —— 对恶意节点应用篡改 ——
            updates, true_mal_round = apply_malicious_updates(
                updates,
                malicious_ids=true_mal,
                behavior=args.mal_behavior,
                scale=args.mal_scale,
                noise_std=args.mal_noise_std,
                seed=42 + r,
            )
        print(f"Round {r} updates: {len(updates)} clients")

        eval_accs: list[float] = []
        eval_losses: list[float] = []
        node_ids: list[int] = []
        eval_metrics_map: dict[int, dict[str, float]] = {}

        # 1. Evaluate all clients and store metrics per node
        for u in updates:
            m = evaluate_global_params(args.model, u.params, X_eval, y_eval)
            print(f"Evaluated client {u.node_id}: acc={m['acc']:.4f}, loss={m['loss']:.4f}")
            eval_accs.append(m["acc"])
            eval_losses.append(m["loss"])
            node_ids.append(u.node_id)
            eval_metrics_map[u.node_id] = m
        print(eval_accs)

        # 2. Detect malicious clients using a simple accuracy threshold
        threshold = 0.1
        malicious = {nid for nid, acc in zip(node_ids, eval_accs) if acc < threshold}

        print(f"Detected malicious clients (Threshold < {threshold}):", malicious)

        for u in updates:
            m = eval_metrics_map[u.node_id]
            contract.set_features(
                u.node_id,
                flat_update=u.params,
                claimed_acc=float(u.metrics.get("acc")),
                eval_acc=float(m['acc']),
            )
            contract.set_contribution(u.node_id, float(m['acc']))
            contract.credit_reward(u.node_id, 10.0 * float(m['acc']))


        # We pass the updates into the contract per its interface.

        # Note: Contract will call aggregation which expects absolute params.
        result = contract.run_round(
            r, detected_ids=malicious, updates=updates, true_malicious=true_mal
        )  # malicious ground-truth here
        
        print("=== Per-client test acc ===")
        # for u in updates:
        #     m = evaluate_global_params(args.model, u.params, X_eval, y_eval)
        #     print(f"client {u.node_id}: acc={m['acc']:.4f}, loss={m['loss']:.4f}")

        mG = evaluate_global_params(args.model, result["global_params"], X_eval, y_eval)
        print(f"GLOBAL: acc={mG['acc']:.4f}, loss={mG['loss']:.4f}")


        global_params2 = result["global_params"]
        
        eval_metrics2 = evaluate_global_params(args.model, global_params2,
                                         X_eval, y_eval)
        print(f"[Eval] Global after round {r}: acc={eval_metrics2['acc']:.4f}, loss={eval_metrics2['loss']:.4f}")
        
        # exit()
        # The contract stores the new global in result["metrics"] if configured; otherwise,
        # res = c.run_round(r, updates=updates, true_malicious=true_mal)
        # print(f"Results: {result}")
        results.append(result)
        # we compute it from the aggregation strategy directly (using the same as in contract).
        # agg = contract.aggregator  # type: ignore[attr-defined]
        # if hasattr(agg, "aggregate"):
        #     global_params = agg.aggregate(updates, prev_global=reference_global)  # type: ignore[arg-type]
        # else:
        #     # Fallback: average the weights
        #     flats = [np.concatenate([v.ravel() for v in u.params.values()]) for u in updates]
        #     avg = np.mean(np.stack(flats, axis=0), axis=0)
        #     # Reshape like reference_global
        #     template = reference_global
        #     i = 0; shaped = {}
        #     for k, v in template.items():
        #         n = v.size; shaped[k] = avg[i:i+n].reshape(v.shape); i += n
        #     global_params = shaped
    summary = contract.metrics.summary()
    # out_obj = {"results": results, "summary": summary, "config": config, "true_malicious": sorted(true_mal),
    #            "malicious_ratio": malicious_ratio, "nodes": nodes, "rounds": rounds}
    print(f"Final summary: {summary[-1] if summary else {}}")
    print(f"Round {r} summary:", result.get("metrics", {}))

if __name__ == "__main__":
    main()
