import pytest
from flsim.aggregation.base import AggregationStrategy


def test_aggregate_empty_input_raises():
    agg = AggregationStrategy()
    with pytest.raises(ValueError, match="no updates"):
        agg.aggregate([])
