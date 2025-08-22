import numpy as np
from flsim.model.nn import MLP


def test_mlp_fit_local_returns_consistent_metrics():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    X_val = X.copy()
    y_val = y.copy()

    model = MLP(input_dim=2, num_classes=2, hidden_dim=4)
    model.init_parameters()

    metrics = model.fit_local(
        X,
        y,
        epochs=1,
        batch_size=1,
        lr=0.1,
        shuffle=False,
        rng=np.random.default_rng(0),
        X_val=X_val,
        y_val=y_val,
    )
    print(metrics)
    assert metrics["acc"] == metrics["train_acc"]
    assert set(["loss", "acc", "train_acc", "val_acc", "n"]).issubset(metrics)
    assert not np.isnan(metrics["val_acc"])
