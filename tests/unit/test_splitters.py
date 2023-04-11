from fold.splitters import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from fold.utils.tests import generate_sine_wave_data


def test_expanding_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)

    splitter = ExpandingWindowSplitter(initial_train_window=0.4, step=0.4)
    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)


def test_expanding_window_splitter_embargo():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400, embargo=10)

    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400

    splitter = ExpandingWindowSplitter(initial_train_window=0.4, step=0.4, embargo=10)
    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400


def test_sliding_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SlidingWindowSplitter(initial_train_window=400, step=400)

    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)

    splitter = SlidingWindowSplitter(initial_train_window=0.4, step=0.4)

    splits = splitter.splits(len(X))
    assert len(splits) == len(X) // 400
    assert splits[-1].test_window_end == len(X)


def test_sliding_window_splitter_embargo():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SlidingWindowSplitter(initial_train_window=400, step=400, embargo=10)

    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400

    splitter = SlidingWindowSplitter(initial_train_window=0.4, step=0.4, embargo=10)
    splits = splitter.splits(len(X))
    assert splits[-1].test_window_end == len(X)
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400


def test_single_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SingleWindowSplitter(train_window=0.4)

    splits = splitter.splits(len(X))
    assert len(splits) == 1
    assert splits[0].test_window_end == 1000

    splitter = SingleWindowSplitter(train_window=400)

    splits = splitter.splits(len(X))
    assert len(splits) == 1
    assert splits[0].test_window_end == 1000


def test_single_window_splitter_embargo():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SingleWindowSplitter(train_window=0.4, embargo=10)

    splits = splitter.splits(len(X))
    assert len(splits) == 1
    assert splits[0].test_window_end == 1000
    assert splits[0].train_window_end == 390
    assert splits[0].test_window_start == 400
