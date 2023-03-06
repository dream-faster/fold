from fold.splitters import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from fold.utils.tests import generate_sine_wave_data


def test_expanding_window_splitter():

    X = generate_sine_wave_data()
    splitter = ExpandingWindowSplitter(train_window_size=400, step=400)

    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)


def test_expanding_window_splitter_embargo():

    X = generate_sine_wave_data()
    splitter = ExpandingWindowSplitter(train_window_size=400, step=400, embargo=10)

    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400


def test_sliding_window_splitter():

    X = generate_sine_wave_data()
    splitter = SlidingWindowSplitter(train_window_size=400, step=400)

    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)


def test_sliding_window_splitter_embargo():

    X = generate_sine_wave_data()
    splitter = SlidingWindowSplitter(train_window_size=400, step=400, embargo=10)

    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400


def test_single_window_splitter():

    X = generate_sine_wave_data()
    splitter = SingleWindowSplitter(train_window_size=0.4)

    splits = splitter.splits(len(X))
    assert len(splits) == 1
    assert splits[0].test_window_end == 1000


def test_single_window_splitter_embargo():

    X = generate_sine_wave_data()
    splitter = SingleWindowSplitter(train_window_size=0.4, embargo=10)

    splits = splitter.splits(len(X))
    assert len(splits) == 1
    assert splits[0].test_window_end == 1000
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400
