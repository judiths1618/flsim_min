from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from ...core.registry import DETECTION
from .utils import (
    to_1d_float,
    flatten_features,
    l2_normalize,
    aggregate_models,
    evaluate_on_trigger_data,
    evaluate_on_clean_data,
)

@DETECTION.register("flame_v2")
class FlameV2Detector:
    def __init__(self, *, embed_dim: int = 30, eps: float = 0.3, min_samples: int = 2,
                 trigger_acc_thresh: float = 0.6, clean_acc_drop_thresh: float = 0.2):
        self.embed_dim = embed_dim
        self.eps = eps
        self.min_samples = min_samples
        self.trigger_acc_thresh = trigger_acc_thresh
        self.clean_acc_drop_thresh = clean_acc_drop_thresh

    def detect(self, updates: Dict[int, Dict[str, Any]], base_model, trigger_loader, clean_loader):
        ids, X = flatten_features(updates)
        if X is None or X.shape[0] < 3:
            return {i: False for i in ids}

        # Step 1: Learn embedding
        Xn = l2_normalize(X)
        pca = PCA(n_components=min(self.embed_dim, X.shape[1]))
        X_embedded = pca.fit_transform(Xn)

        # Step 2: Clustering
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clusterer.fit_predict(X_embedded)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        flagged = {i: False for i in ids}
        if n_clusters == 0:
            return flagged  # No meaningful clustering

        # Step 3: Per-cluster evaluation
        for label in set(labels):
            if label == -1:
                continue
            cluster_ids = [ids[i] for i in range(len(ids)) if labels[i] == label]
            cluster_models = [updates[i]["flat_update"] for i in cluster_ids]

            # Aggregate updates in this cluster
            agg_model = aggregate_models(cluster_models, base_model)

            # Evaluate
            trigger_acc = evaluate_on_trigger_data(agg_model, trigger_loader)
            clean_acc = evaluate_on_clean_data(agg_model, clean_loader)

            if trigger_acc > self.trigger_acc_thresh or clean_acc < (1.0 - self.clean_acc_drop_thresh):
                for cid in cluster_ids:
                    flagged[cid] = True

        return flagged

    # 在聚合前，对客户端提交上来的模型参数进行筛选
	def model_sift(self, round, clients_weight, all_candidates, true_bad, true_good):
		return tpr, tnr
