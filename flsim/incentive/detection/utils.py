import numpy as np
import torch
from typing import Dict, List, Tuple, Any


# 模型聚合
def model_aggregate(self, weight_accumulator, num):
    for name, data in self.global_model.state_dict().items():
        
        update_per_layer = weight_accumulator[name] / num

        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)


# 模型评估
def model_eval(self):

    self.global_model.eval()
    
    total_loss = 0.0
    correct = 0
    correct_poison = 0
    dataset_size = 0
    total_poison_count = 0

    for batch_id, batch in enumerate(self.eval_loader):
        data, target = batch 
        dataset_size += data.size()[0]

        poison_data = data.clone()

        for i, image in enumerate(poison_data):
            if self.conf["type"] == 'mnist':
                image[0][3:5, 3:5] = 2.821
            elif self.conf["type"] == "fmnist":
                image[0][3:5, 3:5] = 2.028
            else:
                image[0][3:7, 3:7] = 2.514
                image[1][3:7, 3:7] = 2.597
                image[2][3:7, 3:7] = 2.754
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            poison_data = poison_data.cuda()
            
        output = self.global_model(data)
        output_poison = self.global_model(poison_data)
        
        total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        pred_poison = output_poison.data.max(1)[1]  # 在后门图片上的预测值

        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        # correct_poison += pred_poison.eq(target.data.view_as(pred_poison)).cpu().sum().item()

        # 要算在后门图片上的正确率，那原来就是后门标签的数据肯定不能算进来，需要去掉
        # 就比如图片标签本来就是8，而后门攻击目标也是8，如果预测出来是8这个肯定不能算入后门攻击成功
        for i in range(data.size()[0]):
            if pred_poison[i] == self.conf['poison_num'] and target[i] != self.conf['poison_num']:
                total_poison_count += 1

    # correct_poison -= total_poison_count
    correct_poison = total_poison_count
    acc = 100.0 * (float(correct) / float(dataset_size))
    acc_poison = 100.0 * (float(correct_poison) / float(dataset_size))

    total_l = total_loss / dataset_size

    return acc, acc_poison, total_l

def to_1d_float(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        parts = [to_1d_float(x[k]) for k in sorted(x.keys())]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    elif isinstance(x, (list, tuple)):
        parts = [to_1d_float(e) for e in x]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    else:
        return np.asarray(x, dtype=float).ravel()

def flatten_features(features: Dict[int, Dict[str, Any]]) -> Tuple[List[int], np.ndarray | None]:
    ids, vecs = [], []
    for nid, f in features.items():
        if "flat_update" not in f:
            continue
        v = to_1d_float(f["flat_update"])
        if v.size > 0 and np.isfinite(v).all():
            ids.append(nid)
            vecs.append(v)
    if not vecs:
        return ids, None
    max_len = max(v.size for v in vecs)
    vecs = [np.pad(v, (0, max_len - v.size), constant_values=0.0) if v.size < max_len else v for v in vecs]
    return ids, np.stack(vecs)

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def aggregate_models(model_list: List[Dict[str, np.ndarray]], base_model) -> Any:
    # Assume model_list is a list of state_dicts
    import copy
    agg = copy.deepcopy(base_model.state_dict())
    for k in agg:
        agg[k] = sum(m[k] for m in model_list) / len(model_list)
    base_model.load_state_dict(agg)
    return base_model

def evaluate_on_trigger_data(model, dataloader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dataloader:
        with torch.no_grad():
            preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total

def evaluate_on_clean_data(model, dataloader) -> float:
    return evaluate_on_trigger_data(model, dataloader)  # same logic
