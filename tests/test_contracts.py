import numpy as np
from flsim.contracts.composed import ComposedContract, ContractConfig
from flsim.core.types import ModelUpdate

def test_selection_size_and_cooldown():
    c = ComposedContract(ContractConfig(committee_size=3, selection="stratified_softmax", settlement="plans_engine"))
    for nid in range(1, 8):
        c.register_node(nid, stake=100.0, reputation=50.0+nid)
    sel = c.select_committee()
    assert len(sel) == 3
    for nid in sel:
        assert c.cooldowns[nid] >= 1

def test_settlement_penalty_applied():
    c = ComposedContract(ContractConfig(committee_size=3, settlement="plans_engine"))
    for nid in range(1, 5):
        c.register_node(nid, stake=100.0, reputation=50.0)
    for nid in range(1, 5):
        vec = np.ones(10) * (100.0 if nid == 1 else 1.0)
        c.set_features(nid, flat_update=vec, claimed_acc=0.9, eval_acc=0.85)
        c.set_contribution(nid, 0.0 if nid == 1 else 0.9)
        c.credit_reward(nid, 10.0)
    # Mark node 1 as malicious via detected_ids
    res = c.run_round(1, detected_ids={1}, updates=None, true_malicious={1})
    assert isinstance(res, dict)
    assert c.nodes[1].stake < 100.0

def test_flame_aggregation_shape():
    from flsim.aggregation.flame import FlameAggregation
    agg = FlameAggregation(percentile=0.9, use_noise=False)

    ups = [
        ModelUpdate(
            node_id=i,
            params=np.ones(5) * i,
            weight=1.0,
            update_type="weights",
        )
        for i in range(1, 6)
    ]
    out = agg.aggregate(ups, prev_global=np.zeros(5), admitted_ids=[1,2,3,4,5])

    if isinstance(out, dict):
        out_vec = np.concatenate([v.ravel() for v in out.values()], axis=0)
    else:
        out_vec = out.ravel()
    assert out_vec.shape == (5,)


def test_run_round_no_updates_returns_prev_global():
    c = ComposedContract(ContractConfig(committee_size=2, settlement="plans_engine"))
    for nid in range(1, 3):
        c.register_node(nid, stake=100.0, reputation=50.0)
    updates = [ModelUpdate(node_id=1, params=np.ones(3), weight=1.0)]
    first = c.run_round(0, detected_ids=set(), updates=updates)
    second = c.run_round(1, detected_ids=set(), updates=None)
    assert np.array_equal(second["global_params"], first["global_params"])


def test_nodestate_updates():
    c = ComposedContract(ContractConfig(committee_size=2, settlement="plans_engine"))
    for nid in range(1, 4):
        c.register_node(nid, stake=100.0, reputation=50.0)
        c.set_contribution(nid, 0.5)
    updates = [ModelUpdate(node_id=i, params=np.ones(3), weight=1.0) for i in range(1, 4)]
    out = c.run_round(0, detected_ids=set(), updates=updates)
    # participation and contribution history recorded
    for nid in range(1, 4):
        assert c.nodes[nid].participation == 1
        assert len(c.nodes[nid].contrib_history) == 1
        assert c.nodes[nid].cooldown == c.cooldowns[nid]
    # committee history for selected nodes
    for nid in out["committee"]:
        assert 0 in c.nodes[nid].committee_history


def test_committee_bonus_applied_in_settlement():
    c = ComposedContract(ContractConfig(committee_size=1, settlement="plans_engine"))
    for nid in (1, 2):
        c.register_node(nid, stake=100.0, reputation=50.0)
        c.nodes[nid].contrib_history.append(1.0)
    committee = [1]
    c.committee = committee
    plans = c.settlement.run(
        0,
        c.nodes,
        {},
        {},
        {},
        set(),
        committee,
        c.reward,
        c.penalty,
        c.reputation,
    )
    r1 = plans["computed_rewards_next"][1]
    r2 = plans["computed_rewards_next"][2]
    assert r1 > r2
