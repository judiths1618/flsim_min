
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import random
import csv
import os
import sys

random.seed(42)
np.random.seed(42)
# When executed as a script (``python flsim/run_experiment_ours.py``) the
# package-relative imports fail because ``__package__`` is empty. Adjust the
# path in that case so the module can still resolve ``flsim.run_experiment``.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from flsim import run_experiment  # type: ignore
else:  # pragma: no cover - exercised only when run as module
    from . import run_experiment

# Expose helper functions for convenience in ``main``
build_contract_from_yaml = run_experiment.build_contract_from_yaml
train_locally_on_partitions = run_experiment.train_locally_on_partitions
load_flower_arrays = run_experiment.load_flower_arrays

# from flsim.aggregation.base import _zeros_like
from flsim.aggregation.flame import AggregationStrategy as _Flame  # typing only
from flsim.attack.malicious import choose_malicious_nodes, apply_malicious_updates
from flsim.eval.global_evaluation import evaluate_global_on_flower, evaluate_global_params, evaluate_cnn_cifar_on_flower, evaluate_cnn_cifar

# The FlameDetector relies on optional heavy dependencies (torch, hdbscan).
# Provide a small fallback so this module can be imported even when those
# packages are unavailable during testing.
try:  # pragma: no cover - exercised only when deps missing
    from flsim.incentive.detection.flame import FlameDetector  # type: ignore
except Exception:  # pragma: no cover
    print("Warning: optional dependencies for FlameDetector missing, using placeholder.")
    # class FlameDetector:  # type: ignore
    #     """Lightweight placeholder used when optional deps are missing."""

    #     def detect(self, features, scores):
    #         return {}

def robust_z_flags(values, thresh=1, tail="both"):
    v = np.asarray(values, dtype=float)
    print("values:", v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12  # avoid /0
    rz = 0.6745 * (v - med) / mad              # robust z-score
    print("robust z-scores:", rz)
    if tail == "low":
        return rz < -thresh
    elif tail == "high":
        return rz > thresh
    
    return np.abs(rz) > thresh
# 计算 IQR
def outlier_flags(values, k=5.5, tail="both"):
    """
    Return boolean mask for outliers based on IQR method.

    Args:
        values (array-like): numeric values
        k (float): IQR multiplier (default=1.5)
        tail (str): 'low', 'high', or 'both'

    Returns:
        np.ndarray[bool]: True for outliers
    """
    values = np.asarray(values)
    print("values:", values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    print(f"IQR: {iqr}, lower: {lower}, upper: {upper}")
    if tail == "low":
        return values < lower
    elif tail == "high":
        return values > upper
    return (values < lower) | (values > upper)

def jains_fairness(values):
    """Compute Jain's fairness index for a list of values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    num = arr.sum() ** 2
    den = arr.size * np.sum(arr ** 2)
    return float(num / den) if den > 0 else 0.0


def gini_coefficient(values):
    """Compute the Gini coefficient for a list of values."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - arr.min()
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr)) / (n * arr.sum()) - (n + 1) / n)


def run(
    config: str,
    rounds: int,
    nodes: int,
    malicious_ratio: float,
    seed: int,
    dim: int,
    out: str | None = None,
) -> None:
    """Run a minimal FL experiment used in tests.

    The heavy lifting (data loading, local training, contract creation) is
    delegated to the :mod:`flsim.run_experiment` helpers so that tests can
    patch those functions easily.
    """
    contract = run_experiment.build_contract_from_yaml(config)
    for nid in range(nodes):
        contract.register_node(nid, stake=100.0, reputation=50.0)

    Xp, yp, D, K, X_eval, y_eval = run_experiment.load_flower_arrays(
        dataset="cifar10",
        n_clients=nodes,
        iid=True,
        alpha=0.5,
        split="train",
        flatten=True,
        normalize=True,
    )

    global_params = None
    for r in range(1, rounds + 1):
        updates, global_params = run_experiment.train_locally_on_partitions(
            model_name="logreg",
            partitions={cid: (Xp[cid], yp[cid], X_eval, y_eval) for cid in Xp},
            global_params=global_params,
            epochs=1,
            batch_size=1,
            lr=0.1,
            seed=seed + r,
        )
        # In tests the contract interface is minimal, so we avoid using optional
        # keyword arguments like ``detected_ids``.
        contract.run_round(r, updates, true_malicious=set())


def main():
    ap = argparse.ArgumentParser(description="Local model training on Flower partitions (keeps original logic intact)")
    ap.add_argument("--config", required=True, help="Path to YAML config for detection/aggregation/etc.")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=10)

    # —— 新增恶意相关参数 ——
    ap.add_argument("--mal-frac", type=float, default=0.3, help="恶意节点比例（0~1）")
    ap.add_argument("--mal-ids", type=str, default="", help="显式恶意节点ID，逗号分隔，如 1,3,5")
    ap.add_argument("--mal-behavior", type=str, default="scale",
                    choices=["scale", "signflip", "zero", "noise"])
    ap.add_argument("--mal-scale", type=float, default=-10.0, help="scale 行为的缩放因子")
    ap.add_argument("--mal-noise-std", type=float, default=0.5, help="noise 行为的噪声标准差")
    
    # ap.add_argument("--defense", default="flame")
    args = ap.parse_args()

    # --------- Build composed contract from YAML --------
    contract = build_contract_from_yaml(args.config)
    nodes = args.clients

    log_fields = [
        "round",
        "node_id",
        "stake",
        "reputation",
        "train_acc",
        "train_loss",
        "val_acc",
        "val_loss",
        "reward",
        "stake_penalty",
        "rep_penalty",
        "is_committee",
        "is_malicious",
        "detected",
    ]
    log_file = open(f"./results/fl_log_ours_{args.mal_behavior}_{args.dataset}_{args.model}.csv", "w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=log_fields)
    writer.writeheader()

    # ---------- Register nodes with initial stake and reputation ----------
    for nid in range(0, nodes):
        contract.register_node(nid, stake=100.0, reputation=50.0)
    

    # ---------- Load client partitions (X, y) -----------
    Xp, yp, D, K, X_eval, y_eval = load_flower_arrays(
        dataset=args.dataset,
        n_clients=args.clients,
        iid=args.iid,
        alpha=args.alpha,
        split=args.split,
        flatten=True,
        normalize=True,
    )

    # -------- set malicious nodes -----------
    explicit = {int(x) for x in args.mal_ids.split(",") if x.strip()} if args.mal_ids else None
    true_mal = choose_malicious_nodes(
        all_node_ids= [client_id for client_id in Xp.keys()],
        mal_frac=float(args.mal_frac),
        explicit_ids=explicit,
        seed=42,
    )
    print(f"[Info] Malicious nodes: {sorted(true_mal)} (behavior={args.mal_behavior}, ratio={args.mal_frac})")


    # ------------- Initialize global parameters as None (the trainer will create them based on model) -------------
    global_params = None


    results = []
    for r in range(1, args.rounds + 1):
        # Local training per client on their partition
        updates, reference_global = train_locally_on_partitions(
            model_name=args.model,
            partitions={cid: (Xp[cid], yp[cid], X_eval, y_eval) for cid in Xp},
            # partitions={cid: (Xp[cid], yp[cid]) for cid in Xp},
            global_params=global_params,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=42 + r,
        )
        train_metrics_map = {u.node_id: u.metrics for u in updates}

        if r == 1 and getattr(contract, "prev_global", None) is None:
            contract.prev_global = {k: v.copy() for k, v in reference_global.items()}

        if global_params is None and r==1:
            # Optional: evaluate the initial reference global before first aggregation
            verified = evaluate_global_params(args.model, reference_global,
                                            X_eval, y_eval)
            print(f"[Eval] Base global (round 0): acc={verified['acc']:.4f}, loss={verified['loss']:.4f}")
    
        # ------------- apply amlicious updates -----------------
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
        # print(f"Round {r} updates: {len(updates)} clients")

        # ------------- Naive malicious detection method using shared model and updates and outliers ---------------
        eval_accs: list[float] = []
        eval_losses: list[float] = []
        node_ids: list[int] = []
        eval_metrics_map: dict[int, dict[str, float]] = {}
        features_map: dict[int, dict[str, float]] = {}
        score_map: dict[int, float] = {}

        # 1. Evaluate all clients and store metrics per node
        print("=== Per-client test acc ===")
        for u in updates:
            m = evaluate_global_params(args.model, u.params, X_eval, y_eval)
            # print(float(u.metrics.get("val_acc")))
            eval_accs.append(m["acc"])
            eval_losses.append(m["loss"])
            node_ids.append(u.node_id)
            eval_metrics_map[u.node_id] = m
            features_map[u.node_id] = {
                "flat_update": u.params,
                "claimed_acc": float(u.metrics.get("acc")),
                "eval_acc": float(m["acc"]),
                "val_acc":float(u.metrics.get("val_acc"))
            }
            score_map[u.node_id] = float(m["acc"])
            # print(f"Evaluated client {u.node_id}: acc={m['acc']:.4f}, loss={m['loss']:.4f}",u.metrics.get("acc"), u.metrics.get("val_acc"))
                
        # We pass the updates into the contract per its interface.
        for u in updates:
            m = eval_metrics_map[u.node_id]
            contract.set_features(u.node_id, **features_map[u.node_id])
            contract.set_contribution(u.node_id, float(m['acc']))
            # contract.credit_reward(u.node_id, 10.0 * float(m['acc']))

        # 2. Detect suspicious clients using a simple accuracy threshold
        # suspicious={}
        # acc_threshold = 0.5
        # loss_threshold = 10
        # suspicious = {nid for nid, acc, loss in zip(node_ids, eval_accs, eval_losses) if ((acc < acc_threshold) or (acc < acc_threshold and loss > loss_threshold))}
        # print(f"Detected suspicious clients (acc_Threshold < {acc_threshold} and loss_threshold > {loss_threshold}):", suspicious)
        # suspicious = detect_suspicious(node_ids, eval_accs, eval_losses, method="iqr", iqr_k=1.5)
        # print("Suspicious (IQR):", suspicious)

        # print(f"contribution history before round {r}: {contract.contributions}")
        
        # exit()
        # 3. Additional detection via FLAME detector
        flame_detector = FlameDetector()
        flame_flags = flame_detector.detect(features_map, score_map)
        flame_malicious = {nid for nid, flag in flame_flags.items() if flag}
        if flame_malicious:
            print("[Flame] Detected malicious clients:", flame_malicious)
            # suspicious = flame_malicious
        

        # 4. Historical contribution-based detection
        suspicious: set[int] = set()
        history_window = 3
        # collect rolling averages over the window for each client
        history_avgs: dict[int, float] = {}
        for nid, state in contract.nodes.items():
            history = list(state.contrib_history)
            if nid in contract.contributions:
                history.append(contract.contributions[nid])
            if len(history) >= history_window:
                history_avgs[nid] = float(np.mean(history[-history_window:]))

        history_suspicious: set[int] = set()
        if history_avgs:
            # adaptively pick threshold from distribution (10th percentile --> bottom 30% are suspicious)
            history_thresh = np.percentile(list(history_avgs.values()), args.mal_frac * 100)
            history_suspicious = {nid for nid, avg in history_avgs.items() if avg < history_thresh}
            if history_suspicious:
                print("[History] Low contribution clients:", sorted(history_suspicious),
                      f"(threshold={history_thresh:.2f})")
        suspicious = set(suspicious).union(history_suspicious)
        print(f"[history] Detected suspicious clients:", history_suspicious)

        if not suspicious:      # if no suspicious yet, 
            suspicious = set(flame_malicious) # use flame results if any
        elif suspicious.isdisjoint(flame_malicious): # if no overlap, keep history ones
            suspicious = set(suspicious) # keep history ones
        else:   # if overlap, take union
            suspicious = set(suspicious).union(flame_malicious)

        print(f"Final suspicious clients for round {r}:", suspicious)
        # Note: Contract will call aggregation which expects absolute params.
        result = contract.run_round(
            r, detected_ids=suspicious, updates=updates, true_malicious=true_mal
        )  # malicious ground-truth here

        # ---------------- evaluate global model -----------------
        # mG = evaluate_global_params(args.model, result["global_params"], X_eval, y_eval)
        # print(f"GLOBAL: acc={mG['acc']:.4f}, loss={mG['loss']:.4f}")
        global_params2 = result["global_params"]
        eval_metrics2 = evaluate_global_params(args.model, global_params2,
                                         X_eval, y_eval)
        print(f"[Eval] Global after round {r}: acc={eval_metrics2['acc']:.4f}, loss={eval_metrics2['loss']:.4f}\n")

        # Include evaluation metrics in the round results for downstream analysis
        result["metrics"] = eval_metrics2
        results.append(result)
        # print(f"Round {r} result:", result)

        plans = result.get("plans", {})
        rewards_map = plans.get("credit_rewards", {})
        penalties_map = plans.get("apply_penalties", {})
        committee_set = set(result.get("committee", []))
        # detected_set = set(result.get("detected", []))
        truth_set = set(result.get("truth", []))

        for nid, state in result["node_states"].items():
            train_m = train_metrics_map.get(nid, {})
            eval_m = eval_metrics_map.get(nid, {})
            pen = penalties_map.get(nid, {})
            writer.writerow(
                {
                    "round": r,
                    "node_id": nid,
                    "stake": state["stake"],
                    "reputation": state["reputation"],
                    "train_acc": train_m.get("acc", float("nan")),
                    "train_loss": train_m.get("loss", float("nan")),
                    "val_acc": eval_m.get("acc", float("nan")),
                    "val_loss": eval_m.get("loss", float("nan")),
                    "reward": rewards_map.get(nid, 0.0),
                    "stake_penalty": pen.get("stake_mul", 0.0)
                    if nid in penalties_map
                    else 0.0,
                    "rep_penalty": pen.get("rep_mul", 0.0)
                    if nid in penalties_map
                    else 0.0,
                    "is_committee": int(nid in committee_set),
                    "is_malicious": int(nid in truth_set),
                    # "detected": int(nid in detected_set),
                    "detected": int(nid in suspicious),
                }
            )


    summary = contract.metrics.summary()

        # Calculate detection statistics across rounds
    if summary:
        prec = np.array([s.get("precision", 0.0) for s in summary], dtype=float)
        rec = np.array([s.get("recall", 0.0) for s in summary], dtype=float)
        tp = np.array([s.get("tp", 0) for s in summary], dtype=float)
        fp = np.array([s.get("fp", 0) for s in summary], dtype=float)
        fn = np.array([s.get("fn", 0) for s in summary], dtype=float)
        print(
            "Detection over rounds:"
            f"\n  Precision: {prec.mean():.3f} ± {prec.std():.3f}"
            f"\n  Recall:    {rec.mean():.3f} ± {rec.std():.3f}"
            f"\n  TP:        {tp.mean():.3f} ± {tp.std():.3f}"
            f"\n  FP:        {fp.mean():.3f} ± {fp.std():.3f}"
            f"\n  FN:        {fn.mean():.3f} ± {fn.std():.3f}"
        )

    # Balance statistics on final round
    final_balances = results[-1].get("balances", {}) if results else {}
    benign_vals = [v for nid, v in final_balances.items() if nid not in true_mal]
    mal_vals = [v for nid, v in final_balances.items() if nid in true_mal]

    benign_bal = float(sum(benign_vals))
    mal_bal = float(sum(mal_vals))
    benign_mean = float(np.mean(benign_vals)) if benign_vals else 0.0
    benign_std = float(np.std(benign_vals)) if benign_vals else 0.0
    mal_mean = float(np.mean(mal_vals)) if mal_vals else 0.0
    mal_std = float(np.std(mal_vals)) if mal_vals else 0.0

    fairness = jains_fairness(final_balances.values())
    benign_fairness = jains_fairness(benign_vals)
    mal_fairness = jains_fairness(mal_vals)
    gini = gini_coefficient(final_balances.values())
    benign_gini = gini_coefficient(benign_vals)
    mal_gini = gini_coefficient(mal_vals)
    print(
        "Balances - "
        f"benign total: {benign_bal:.4f} (mean {benign_mean:.4f} ± {benign_std:.4f}), "
        f"malicious total: {mal_bal:.4f} (mean {mal_mean:.4f} ± {mal_std:.4f})\n"
        
        f"Benign fairness: {benign_fairness:.4f}, Malicious fairness: {mal_fairness:.4f}\n"
        f"Benign Gini: {benign_gini:.4f}, Malicious Gini: {mal_gini:.4f}\n"
        
        f"Jain's Fairness: {fairness:.4f}\nGini Coefficient: {gini:.4f}\n"
    )

    # out_obj = {"results": results, "summary": summary, "config": config, "true_malicious": sorted(true_mal),
    #    "malicious_ratio": malicious_ratio, "nodes": nodes, "rounds": rounds}
    # print(f"Final summary: {summary[-1] if summary else {}}")

    log_file.close()

if __name__ == "__main__":
    main()
