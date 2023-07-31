from fold.events.weights.max import WeightByMax, WeightByMaxWithLookahead
from fold.utils.tests import generate_sine_wave_data


def test_weighting_max_with_lookahead():
    strategy = WeightByMaxWithLookahead()
    X, y = generate_sine_wave_data(length=1000)
    weights = strategy.calculate(y)
    assert (weights == y.abs()).all()


def test_weighting_max_without_lookahead():
    strategy = WeightByMax(window_size=0.01)
    X, y = generate_sine_wave_data(length=1000)
    weights = strategy.calculate(y)
    assert not (weights == y.abs()).all()
