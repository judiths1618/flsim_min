#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, os
import numpy as np

from flsim.config import build_contract_from_yaml
from flsim.core.types import ModelUpdate
from flsim.data.flower import load_flower_label_vectors, project_vectors
from flsim.data.flower import load_flower_arrays
from flsim.train.local import train_locally_on_partitions

def simulate_updates(n_nodes:int, dim:int, true_malicious:set[int], benign_mu=1.0, benign_sigma=0.05, mal_sigma=5.0, mal_shift=10.0):
    updates = []
    for nid in range(1, n_nodes+1):
        if nid in true_malicious:
            vec = np.random.randn(dim) * mal_sigma + mal_shift
            acc = float(np.clip(np.random.rand()*0.1 + 0.7, 0, 1))
        else:
            vec = benign_mu + np.random.randn(dim) * benign_sigma
            acc = float(np.clip(np.random.rand()*0.2 + 0.8, 0, 1))
        updates.append(ModelUpdate(node_id=nid, params=vec, weight=1.0, metrics={"eval_acc": acc}))
    return updates

def run(config_path: str, *, rounds:int, nodes:int, malicious_ratio:float, seed:int, dim:int, out:str|None,
        use_flower: bool=False, dataset: str='cifar10', model: str='logreg',iid: bool=True, alpha: float=0.5, split: str='train', epochs:int=10, batch:int=64, lr:float=0.1):
    np.random.seed(seed); random.seed(seed)
    
    # build contract from YAML
    contract = build_contract_from_yaml(config_path)
    
    # Register nodes with initial stake and reputation
    for nid in range(1, nodes+1):
        contract.register_node(nid, stake=100.0, reputation=50.0)

    # Determine number of malicious nodes
    mcount = max(0, int(round(nodes * float(malicious_ratio))))
    mcount = min(nodes, mcount)
    true_mal = set(random.sample(range(1, nodes+1), k=mcount))
    print(f"True malicious nodes: {sorted(true_mal)}")

    # Load client dataset partitions and global evaluation split
    Xp, yp, D, K, X_eval, y_eval = load_flower_arrays(
        dataset=dataset,
        n_clients=nodes,
        iid=iid,
        alpha=alpha,
        split=split,
        flatten=True,
        normalize=True,
    )

    # Initialize global parameters as None (the trainer will create them based on model)
    global_params = None

    results = []
    # Run training rounds
    for r in range(1, rounds + 1):

        # Local training per client on their partition
        updates, reference_global = train_locally_on_partitions(
            model_name=model,
            partitions={cid: (Xp[cid], yp[cid], X_eval, y_eval) for cid in Xp},
            global_params=global_params,
            epochs=epochs,
            batch_size=batch,
            lr=lr,
            seed=42 + r,
        )

        print(f"Round {r} updates: {len(updates)} clients")
        # FL pipeline using the existing contract (detection/aggregation/etc.)
        
        # Set features for each node based on updates
        for u in updates:
            contract.set_features(u.node_id, flat_update=u.params, claimed_acc=float(u.metrics.get("acc")))
        #                    eval_acc=float(u.metrics.get("eval_acc", 0.9)))
        
        # Set contributions and credit rewards 
        for u in updates:
            contract.set_contribution(u.node_id, float(u.metrics["acc"])) 
            contract.credit_reward(u.node_id, 10.0 * float(u.metrics["acc"])) 
    
        print(f"Running {rounds} rounds with {nodes} nodes, malicious ratio {malicious_ratio}, seed {seed}, dim {dim}")
        print(f"True malicious nodes: {sorted(true_mal)}")

        # print("\n", updates)
        # Run the settlement round with the current updates
        res = contract.run_round(r, updates=updates, true_malicious=true_mal)
        print(f"Round {r} results: {res}")

        results.append(res)

        summary = contract.metrics.summary()
        # out_obj = {"results": results, "summary": summary, "config": config_path, "true_malicious": sorted(true_mal),
                # "malicious_ratio": malicious_ratio, "nodes": nodes, "rounds": rounds}
        # print(f"Final summary: {summary[-1] if summary else {}}{out_obj}")
    # if out:
    #     os.makedirs(os.path.dirname(out), exist_ok=True)
    #     with open(out, "w", encoding="utf-8") as f:
    #         json.dump(out_obj, f, indent=2)
    #     print(f"Wrote results to {out}")
    # else:
    #     print(json.dumps(out_obj, indent=2))

def main():
    ap = argparse.ArgumentParser(description="Run FL experiment with YAML configuration and Flower partitions") 
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--malicious_ratio", type=float, default=0.25, help="fraction of malicious clients in [0,1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=100, help="update dimension for synthetic or projected vectors")
    ap.add_argument("--out", type=str, default=None, help="json output path")

    ap.add_argument("--use_flower", action="store_true", help="Use Flower datasets for partitions")
    ap.add_argument("--dataset", type=str, default="cifar10", help="Flower dataset name")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--iid", action="store_true", help="Use IID partitioning (default if flag set)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-IID (effective when --iid not set)")
    ap.add_argument("--split", type=str, default="train", help="Dataset split (train/test/val)")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)


    args = ap.parse_args()
    run(args.config, rounds=args.rounds, nodes=args.nodes, malicious_ratio=args.malicious_ratio, seed=args.seed, dim=args.dim, out=args.out,
        use_flower=args.use_flower, dataset=args.dataset, model=args.model, iid=args.iid, alpha=args.alpha, split=args.split)

if __name__ == "__main__":
    main()
