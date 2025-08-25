import numpy as np

from flsim.contracts.composed import ComposedContract, ContractConfig


def test_node_zero_incentive_flow():
    """Ensure node 0 participates fully in incentive mechanisms."""
    c = ComposedContract(ContractConfig(committee_size=1, settlement="plans_engine"))

    # Register nodes including node 0
    c.register_node(0, stake=100.0, reputation=50.0)
    c.register_node(1, stake=100.0, reputation=50.0)

    # Contribution tracking should record node 0
    c.set_contribution(0, 0.8)
    assert c.contributions.get(0) == 8

    # Reward accounting should credit node 0
    c.credit_reward(0, amount=10.0)
    assert c.rewards.get(0) == 10.0

    # Reputation updates should modify node 0's reputation
    prev_rep = c.nodes[0].reputation
    c.update_reputation(0, contribution=0.8, current_round=1)
    assert c.nodes[0].reputation != prev_rep

    # Penalty application should reduce stake and reputation of node 0
    prev_stake = c.nodes[0].stake
    prev_rep = c.nodes[0].reputation
    c.apply_penalty(0, stake_mul=0.9, rep_mul=0.9)
    assert np.isclose(c.nodes[0].stake, prev_stake * 0.9)
    assert np.isclose(c.nodes[0].reputation, prev_rep * 0.9)

