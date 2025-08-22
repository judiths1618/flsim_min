import numpy as np
from unittest.mock import patch

from flsim.core.types import ModelUpdate
from flsim import run_experiment


def test_run_experiment_smoke(tmp_path):
    nodes = 2
    D = 3
    K = 2
    Xp = {0: np.zeros((2, D), dtype=np.float32), 1: np.ones((2, D), dtype=np.float32)}
    yp = {0: np.array([0, 1]), 1: np.array([1, 0])}
    X_eval = np.zeros((1, D), dtype=np.float32)
    y_eval = np.array([0])

    def fake_train_locally_on_partitions(model_name, partitions, global_params=None, epochs=1, batch_size=1, lr=0.1, seed=0):
        for cid, part in partitions.items():
            assert part[2] is X_eval
            assert part[3] is y_eval
        updates = [
            ModelUpdate(node_id=cid, params=np.zeros(D), metrics={"acc": 1.0, "val_acc": 1.0})
            for cid in partitions
        ]
        global_params = {"W": np.zeros((D, K)), "b": np.zeros(K)}
        return updates, global_params

    class DummyContract:
        def __init__(self):
            self.metrics = type("M", (), {"summary": lambda self: []})()

        def register_node(self, nid, stake, reputation):
            pass

        def set_features(self, nid, **kwargs):
            pass

        def set_contribution(self, nid, score):
            pass

        def credit_reward(self, nid, amount):
            pass

        def run_round(self, r, updates, true_malicious):
            return {"round": r, "global_params": {"W": np.zeros((D, K)), "b": np.zeros(K)}}

    with patch("flsim.run_experiment.load_flower_arrays", return_value=(Xp, yp, D, K, X_eval, y_eval)), \
         patch("flsim.run_experiment.train_locally_on_partitions", side_effect=fake_train_locally_on_partitions), \
         patch("flsim.run_experiment.build_contract_from_yaml", return_value=DummyContract()):
        run_experiment.run(
            "configs/exp_default.yaml",
            rounds=1,
            nodes=nodes,
            malicious_ratio=0.0,
            seed=0,
            dim=D,
            out=None,
        )
