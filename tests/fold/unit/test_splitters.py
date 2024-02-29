from fold.splitters import (
    Bounds,
    ExpandingWindowSplitter,
    ForwardSingleWindowSplitter,
    SlidingWindowSplitter,
    _bounds_exists_after,
    _bounds_exists_before,
    get_splits,
)
from fold.utils.tests import generate_sine_wave_data


def test_bounds_exists_after():
    # Test case 1: bounds exists after the index
    bounds = Bounds(0, 1)
    after_bounds = [Bounds(1, 2)]
    assert _bounds_exists_after(bounds, after_bounds) is True

    # Test case 2: bounds does not exist after the index
    bounds = Bounds(5, 6)
    before_bounds = [Bounds(1, 2)]
    assert _bounds_exists_after(bounds, before_bounds) is False

    # Test case 3: bounds does not exist after the index
    bounds = Bounds(0, 1)
    before_bounds = [Bounds(0, 1)]
    assert _bounds_exists_after(bounds, before_bounds) is False

    # Test case 3: bounds is empty
    bounds = Bounds(5, 6)
    assert _bounds_exists_after(bounds, []) is False


def test_bounds_exists_before():
    # Test case 1: bounds exists after the index
    bounds = Bounds(0, 1)
    after_bounds = [Bounds(1, 2)]
    assert _bounds_exists_before(bounds, after_bounds) is False

    # Test case 2: bounds does not exist after the index
    bounds = Bounds(5, 6)
    before_bounds = [Bounds(1, 2)]
    assert _bounds_exists_before(bounds, before_bounds) is True

    # Test case 3: bounds does not exist after the index
    bounds = Bounds(0, 1)
    before_bounds = [Bounds(0, 1)]
    assert _bounds_exists_before(bounds, before_bounds) is False

    # Test case 3: bounds is empty
    bounds = Bounds(5, 6)
    assert _bounds_exists_before(bounds, []) is False


def test_expanding_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)

    splits = get_splits(
        splitter=splitter, index=X.index, gap_after=0, gap_before=0, merge_threshold=0.0
    )
    assert len(splits) == len(X) // 400
    assert splits[-1].test_bounds[0].end == len(X)

    splitter = ExpandingWindowSplitter(initial_train_window=0.4, step=0.4)
    splits = get_splits(
        splitter=splitter, index=X.index, gap_after=0, gap_before=0, merge_threshold=0.0
    )
    assert len(splits) == len(X) // 400
    assert splits[-1].test_bounds[0].end == len(X)


def test_gap_before():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=0,
        gap_before=10,
        merge_threshold=0.0,
    )
    assert splits[-1].test_bounds[0].end == len(X)
    assert splits[0].train_bounds[0].end == 390
    assert splits[0].test_bounds[0].start == 400


def test_gap_after():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=10,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert splits[-1].test_bounds[0].end == len(X)
    assert splits[0].train_bounds[0].end == 400
    assert splits[0].test_bounds[0].start == 400

    # CPCV!


def test_splitter_merge():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ExpandingWindowSplitter(initial_train_window=0.1, step=0.0999)

    splits = get_splits(
        splitter, index=X.index, gap_before=0, gap_after=0, merge_threshold=0.02
    )
    assert len(splits) == (len(X) // int(1000 * 0.0999) - 1)
    assert splits[-1].test_bounds[0].end == len(X)


def test_sliding_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SlidingWindowSplitter(train_window=400, step=400)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=0,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert len(splits) == len(X) // 400
    assert splits[-1].test_bounds[0].end == len(X)

    splitter = SlidingWindowSplitter(train_window=0.4, step=0.4)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=10,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert len(splits) == len(X) // 400
    assert splits[-1].test_bounds[0].end == len(X)


def test_sliding_window_splitter_initial_window():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = SlidingWindowSplitter(train_window=0.4, step=0.1, initial_window=0.2)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=00,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert splits[0].train_bounds[0].end == 200
    assert len(splits) == (len(X) // 100) - 2
    assert splits[-1].test_bounds[0].end == len(X)


def test_single_window_splitter():
    X, _ = generate_sine_wave_data(length=1000)
    splitter = ForwardSingleWindowSplitter(train_window=0.4)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=0,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert len(splits) == 1
    assert splits[0].test_bounds[0].end == 1000

    splitter = ForwardSingleWindowSplitter(train_window=400)

    splits = get_splits(
        splitter=splitter,
        index=X.index,
        gap_after=0,
        gap_before=0,
        merge_threshold=0.0,
    )
    assert len(splits) == 1
    assert splits[0].test_bounds[0].end == 1000
