from __future__ import annotations
from typing import Dict, Set
import numpy as np
import torch
import hdbscan  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
from ...core.registry import DETECTION


from typing import Any, Dict, List, Tuple
import numpy as np

def _to_1d_float(x: Any) -> np.ndarray:
    """Recursively flatten x -> 1D float array.
    Supports dict (sorted by key), list/tuple, array-like, and scalars.
    """
    if isinstance(x, dict):
        parts = []
        # 保持确定性：按键名排序后拼接
        for k in sorted(x.keys()):
            parts.append(_to_1d_float(x[k]))
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    elif isinstance(x, (list, tuple)):
        parts = [_to_1d_float(e) for e in x]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
        return arr.ravel()

def _flattened(features: Dict[int, Dict[str, Any]]) -> Tuple[List[int], np.ndarray | None]:
    ids: List[int] = []
    vecs: List[np.ndarray] = []

    for nid, f in features.items():
        if "flat_update" not in f:
            continue
        v = _to_1d_float(f["flat_update"])
        if v.size > 0 and np.isfinite(v).all():
            ids.append(int(nid))
            vecs.append(v)

    if not vecs:
        return ids, None

    # 对齐长度：用 0 填充到统一长度，避免 stack 失败
    maxd = max(v.size for v in vecs)
    if any(v.size != maxd for v in vecs):
        vecs = [v if v.size == maxd else np.pad(v, (0, maxd - v.size), constant_values=0.0) for v in vecs]

    X = np.stack(vecs, axis=0)
    return ids, X

# def _flattened(features: Dict[int, Dict[str, float]]):
#     """Flatten features to a list of node IDs and a 2D array of vectors."""
#     ids, vecs = [], []
#     for nid, f in features.items():
#         if "flat_update" in f:
#             v = np.asarray(f["flat_update"], dtype=float).ravel()
#             if v.size > 0 and np.isfinite(v).all():
#                 ids.append(int(nid)); vecs.append(v)
#     return ids, (np.stack(vecs, axis=0) if vecs else None)

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / nrm

def _largest_label(labels: np.ndarray) -> int | None:
    labs, counts = np.unique(labels[labels != -1], return_counts=True)
    if labs.size == 0:
        return None
    return int(labs[np.argmax(counts)])


def _pairwise_euclidean(Xn: np.ndarray) -> np.ndarray:
    G = Xn @ Xn.T
    nrm = np.sum(Xn * Xn, axis=1, keepdims=True)
    D2 = np.maximum(0.0, nrm + nrm.T - 2.0 * G)
    return np.sqrt(D2) + 1e-12


@DETECTION.register("flame")
class FlameDetector:
    """FLAME-style filtering via clustering on cosine distance with fallbacks."""
    def __init__(self, *, min_points: int = 4, min_cluster_frac: float = 0.2, dbscan_eps: float = 0.3,
                 detect_score_thresh: float = 0.05):
        self.min_points = int(min_points)
        self.min_cluster_frac = float(np.clip(float(min_cluster_frac), 0.05, 0.8))
        self.dbscan_eps = float(dbscan_eps)
        self.detect_score_thresh = float(detect_score_thresh)


    # 在聚合前，对客户端提交上来的模型参数进行筛选
    def model_sift(self, round, clients_weight, all_candidates, true_bad, true_good):
        # 用来存储筛选后模型参数和
        weight_accumulator = {}
        for name, params in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 0. 数据预处理，将clients_weight展开成二维tensor, 方便聚类计算
        clients_weight_ = []
        clients_weight_total = []
        for data in clients_weight:
            client_weight = torch.tensor([])
            client_weight_total = torch.tensor([])

            for name, params in data.items():
                client_weight = torch.cat((client_weight, params.reshape(-1).cpu()))
                if name == 'fc.weight' or name == 'fc.bias':
                    client_weight_total = torch.cat((client_weight_total, (params + self.global_model.state_dict()[name]).reshape(-1).cpu()))

            clients_weight_.append(client_weight)
            clients_weight_total.append(client_weight_total)

        # 获得了每个客户端模型的参数，矩阵大小为(客户端数, 参数个数)
        clients_weight_ = torch.stack(clients_weight_)
        clients_weight_total = torch.stack(clients_weight_total)
        num_clients = clients_weight_total.shape[0]
        euclidean = (clients_weight_ ** 2).sum(1).sqrt()
        med = euclidean.median()
        tpr, tnr = 0, 0

        if self.conf['defense'] == 'flame':

            # # 1. HDBSCAN余弦相似度聚类
            clients_weight_total = clients_weight_total.double()
            cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=num_clients//2+1, min_samples=1,allow_single_cluster=True)

            # L2 = torch.norm(clients_weight_total, p=2, dim=1, keepdim=True)
            # clients_weight_total = clients_weight_total.div(L2)
            # cluster = hdbscan.HDBSCAN(min_cluster_size=num_clients//2+1, min_samples=1, allow_single_cluster=True)

            cluster.fit(clients_weight_total)
            predict_good = []
            predict_bad = []
            for i, j in enumerate(cluster.labels_):
                if j == 0:
                    predict_good.append(all_candidates[i])
                else:
                    predict_bad.append(all_candidates[i])

            print(cluster.labels_)
            predict_good = set(predict_good)
            predict_bad = set(predict_bad)
            true_bad = set(true_bad)
            true_good = set(true_good)
            if len(true_good) == 0 and len(predict_good) == 0:
                tnr = 1
            elif len(predict_good) == 0 and len(true_good)!=0:
                tnr = 0
            else:
                tnr = len(true_good & predict_good) / len(predict_good)

            if len(true_bad) == 0 and len(predict_bad) == 0:
                tpr = 1
            elif len(predict_bad) == 0 and len(true_bad)!=0:
                tpr = 0
            else:
                tpr = len(true_bad & predict_bad) / len(predict_bad)

            # 2. 范数中值裁剪
            for i, data in enumerate(clients_weight):
                gama = med.div(euclidean[i])
                if gama > 1:
                    gama = 1

                for name, params in data.items():
                    params.data = (params.data * gama).to(params.data.dtype)

        elif self.conf['defense'] == 'krum':
            # 记录距离与得分
            number = 6
            if round == 4:
                number = 7
            dis = torch.zeros(num_clients, num_clients)
            score = torch.zeros(num_clients)
            for i in range(num_clients):
                for j in range(i+1, num_clients):
                    dis[i][j] = torch.norm(clients_weight_total[i] - clients_weight_total[j], p=2)
                    dis[j][i] = dis[i][j]

            # 获取最近的6个模型参数，包括自己
            for i, di in enumerate(dis):
                values, _ = torch.topk(di, k=number, largest=False)
                score[i] = values.sum()

            # 获得得分最低的6个模型参数
            _, indices = torch.topk(score, k=number, largest=False)
            print(indices)

        else:
            for i, data in enumerate(clients_weight):
                gama = med.div(euclidean[i])
                if gama > 1:
                    gama = 1

                for name, params in data.items():
                    params.data = (params.data * gama).to(params.data.dtype)
        # 3. 聚合
        num_in = 0
        for i, data in enumerate(clients_weight):
            if self.conf['defense'] == "flame":
                if cluster.labels_[i] == 0:
                    num_in += 1
                    for name, params in data.items():
                        weight_accumulator[name].add_(params)

            elif self.conf['defense'] == "krum":
                if i in indices:
                    num_in += 1
                    for name, params in data.items():
                        weight_accumulator[name].add_(params)

            else:
                num_in += 1
                for name, params in data.items():
                    weight_accumulator[name].add_(params)

        temp = torch.tensor([])
        for name, data in self.global_model.named_parameters():
            temp = torch.cat((temp, weight_accumulator[name].reshape(-1).cpu()))

        print(temp.norm(2))

        self.model_aggregate(weight_accumulator, num_in)

        # 4. 聚合模型添加噪声

        if self.conf['defense'] == 'flame' or self.conf['defense'] == 'canyou':
            lamda = 0.000012
            for name, param in self.global_model.named_parameters():
                if 'bias' in name or 'bn' in name:
                    # 不对偏置和BatchNorm的参数添加噪声
                    continue
                std = lamda * med * param.data.std()
                noise = torch.normal(0, std, size=param.size()).cuda()
                param.data.add_(noise)

        return tpr, tnr

    def detect(self, features: Dict[int, Dict[str, float]], scores: Dict[int, float]) -> Dict[int, bool]:
        ids, X = _flattened(features)
        if X is None or X.shape[0] < self.min_points:
            keys = set(list(scores.keys()) + list(features.keys()))
            return {int(n): (float(scores.get(n, 0.0)) < self.detect_score_thresh) for n in keys}

        norms = np.linalg.norm(X, axis=1)

        med = float(np.median(norms))
        iqr = np.subtract(*np.percentile(norms, [75, 25]))
        thresh = med + 1.5 * max(1e-8, float(iqr))
        norm_flags = norms > thresh
        if np.any(norm_flags):
            flagged = {ids[i]: bool(norm_flags[i]) for i in range(len(ids))}

            for nid in (set(scores.keys()) - set(ids)):
                flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
            return flagged

        Xn = _l2_normalize(X)

        labels = None
        try:
            
            min_cluster_size = max(self.min_points, int(self.min_cluster_frac * Xn.shape[0]))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
            labels = clusterer.fit_predict(Xn)
        except Exception:
            try:
                
                min_samples = max(2, int(self.min_cluster_frac * Xn.shape[0]))
                clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=min_samples, metric='cosine')
                labels = clusterer.fit_predict(Xn)
            except Exception:
                labels = None

        if labels is not None and np.any(labels != -1):
            keep_label = _largest_label(labels)
            admitted = {ids[i] for i, lb in enumerate(labels) if lb == keep_label}
            flagged = {nid: (nid not in admitted) for nid in ids}
            for nid in (set(scores.keys()) - set(ids)):
                flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
            return flagged

        # Fallback: medoid + IQR
        D = _pairwise_euclidean(Xn)
        # D = np.sqrt(np.maximum(0.0, (Xn @ Xn.T * -2.0) + (np.sum(Xn*Xn,1,keepdims=True) + np.sum(Xn*Xn,1,keepdims=True).T))) + 1e-12
        s = np.sum(D, axis=1)
        med = int(np.argmin(s))
        d_med = D[med]
        q1, q3 = np.percentile(d_med, [25, 75])
        iqr = max(1e-8, float(q3 - q1))
        tau = float(q3 + 1.5 * iqr)
        flags = (d_med > tau)
        flagged = {ids[i]: bool(flags[i]) for i in range(len(ids))}
        for nid in (set(scores.keys()) - set(ids)):
            flagged[int(nid)] = bool(float(scores.get(nid, 0.0)) < self.detect_score_thresh)
        print("flagged: ", flagged)
        return flagged
