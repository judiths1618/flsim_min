import numpy as np
from flsim.model.logreg import LogisticRegression


def test_logreg_fit_local_returns_consistent_metrics():
    # Simple dataset where model can achieve perfect accuracy
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    X_val = X.copy()
    y_val = y.copy()

    model = LogisticRegression(input_dim=2, num_classes=2)
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

    assert metrics["acc"] == metrics["train_acc"]
    assert set(["loss", "acc", "train_acc", "val_acc", "n"]).issubset(metrics)
    assert not np.isnan(metrics["val_acc"])
