from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from finml_utils.list import flatten_lists


@dataclass
class Bounds:
    start: int
    end: int

    def union(self: Bounds, other: Bounds) -> Bounds:
        return Bounds(start=min(self.start, other.start), end=max(self.end, other.end))

    def intersection(self: Bounds, other: Bounds) -> Bounds:
        return Bounds(max(self.start, other.start), min(self.end, other.end))

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bounds):
            return NotImplemented
        return self.start == other.start and self.end == other.end


@dataclass
class Fold:
    index: int
    train_bounds: list[Bounds]
    test_bounds: list[Bounds]

    def train_indices(self) -> np.ndarray:
        return np.hstack(
            [np.arange(bound.start, bound.end) for bound in self.train_bounds]
        )

    def test_indices(self) -> np.ndarray:
        return np.hstack(
            [np.arange(bound.start, bound.end) for bound in self.test_bounds]
        )


class Splitter:
    def splits(self, index: pd.Index) -> list[Fold]:
        raise NotImplementedError


def get_splits(
    splitter: Splitter,
    index: pd.Index,
    gap_before: int,
    gap_after: int,
    merge_threshold: float = 0.1,
) -> list[Fold]:
    return _merge_last_fold_if_too_small(
        [
            _apply_gap_before_test(_apply_gap_after_test(fold, gap_after), gap_before)
            for fold in splitter.splits(index)
        ],
        int(len(index) * merge_threshold),
    )


@dataclass
class SlidingWindowSplitter(Splitter):
    """
    Creates folds with a sliding window.
    The folds are created by moving the train and test windows forward by a fixed step size.
    See [the documentation](https://dream-faster.github.io/fold/concepts/splitters/) for more details.

    Parameters
    ----------

    train_window : int, float
        The training window size. If a float, it is interpreted as a fraction of the total length of the data.
    step : int, float
        The step size of the sliding window. If a float, it is interpreted as a fraction of the total length of the data.
    embargo : int, optional
        The gap between the train and the test window, by default 0.
    start : int, optional
        The start index of the first fold, by default 0.
    end : int, optional
        The end index of the last fold, by default None.
    """

    train_window: int | float  # this is what you don't get out of sample get predictions for
    step: int | float
    initial_window: int | float | None = None
    start: int | pd.Timestamp = 0
    subtract_initial_train_window_from_start: bool = False
    end: int | pd.Timestamp | None = None
    overlapping_test_sets: bool = False

    def splits(self, index: pd.Index) -> list[Fold]:
        length = len(index)
        window_size = _translate_float_if_needed(self.train_window, length)
        initial_window = (
            _translate_float_if_needed(self.initial_window, length)
            if self.initial_window is not None
            else window_size
        )
        start = _get_start(
            self.start,
            index,
            self.subtract_initial_train_window_from_start,
            initial_window,
        )
        end = _get_end(self.end, index)
        step = _translate_float_if_needed(self.step, length)

        first_window_start = (
            start + initial_window
            if initial_window is not None
            else start + window_size
        )

        return [
            Fold(
                index=order,
                train_bounds=[Bounds(start=max(index - window_size, start), end=index)],
                test_bounds=[
                    Bounds(
                        start=index,
                        end=end
                        if self.overlapping_test_sets
                        else min(end, index + step),
                    )
                ],
            )
            for order, index in enumerate(range(first_window_start, end, step))
        ]


@dataclass
class ExpandingWindowSplitter(Splitter):
    """
    Creates folds with an expanding window.
    The folds are created by moving the end of the train and test windows forward by a fixed step size,
    while keeping the training window's start fixed.
    See [the documentation](https://dream-faster.github.io/fold/concepts/splitters/) for more details.

    Parameters
    ----------

    initial_train_window : int, float
        The initial training window size. If a float, it is interpreted as a fraction of the total length of the data.
    step : int, float
        The step size of the sliding window. If a float, it is interpreted as a fraction of the total length of the data.
    start : int, optional
        The start index of the first fold, by default 0.
    end : int, optional
        The end index of the last fold, by default None.
    """

    initial_train_window: int | float  # this is what you don't out of sample get predictions for
    step: int | float
    start: int | pd.Timestamp = 0
    subtract_initial_train_window_from_start: bool = False
    end: int | pd.Timestamp | None = None
    overlapping_test_sets: bool = False

    def splits(self, index: pd.Index) -> list[Fold]:
        length = len(index)
        end = _get_end(self.end, index)
        window_size = _translate_float_if_needed(self.initial_train_window, length)
        start = _get_start(
            self.start,
            index,
            self.subtract_initial_train_window_from_start,
            window_size,
        )
        step = _translate_float_if_needed(self.step, length)

        return [
            Fold(
                index=order,
                train_bounds=[Bounds(start=start, end=index)],
                test_bounds=[
                    Bounds(
                        start=index,
                        end=end
                        if self.overlapping_test_sets
                        else min(end, index + step),
                    )
                ],
            )
            for order, index in enumerate(range(start + window_size, end, step))
        ]


@dataclass
class ForwardSingleWindowSplitter(Splitter):
    """
    Creates a single fold with a fixed train and test window.
    See [the documentation](https://dream-faster.github.io/fold/concepts/splitters/) for more details.

    Parameters
    ----------
    train_window : int, float
        The training window size. If a float, it is interpreted as a fraction of the total length of the data.
    """

    train_window: int | float  # this is what you don't out of sample get predictions for

    def splits(self, index: pd.Index) -> list[Fold]:
        length = len(index)
        window_size = _translate_float_if_needed(self.train_window, length)

        return [
            Fold(
                index=0,
                train_bounds=[Bounds(start=0, end=window_size)],
                test_bounds=[Bounds(start=window_size, end=length)],
            ),
        ]


@dataclass
class WholeWindowSplitter(Splitter):
    """
    Creates a single fold with a fixed train and test window, the train window is always the full series.

    Parameters
    ----------
    train_window : int, float
        The training window size. If a float, it is interpreted as a fraction of the total length of the data.
    """

    test_window: int | float

    def splits(self, index: pd.Index) -> list[Fold]:
        length = len(index)
        window_size = _translate_float_if_needed(self.test_window, length)

        return [
            Fold(
                index=0,
                train_bounds=[Bounds(start=0, end=length)],
                test_bounds=[Bounds(start=length - window_size, end=length)],
            ),
        ]


@dataclass
class CombinedSplitter(Splitter):
    train_val_splitter: list[Splitter]
    test_splitter: Splitter

    def all_splitters(self) -> Sequence[Splitter]:
        return [*self.train_val_splitter, self.test_splitter]

    def splits(self, index: pd.Index) -> Sequence[Fold]:
        return flatten_lists(
            [splitter.splits(index) for splitter in self.all_splitters()]
        )


def _merge_last_fold_if_too_small(splits: list[Fold], threshold: int) -> list[Fold]:
    if len(splits) == 1:
        return splits
    last_fold = splits[-1]
    if last_fold.test_bounds[0].end - last_fold.test_bounds[0].start > threshold:
        return splits

    previous_fold = splits[-2]
    merged_fold = Fold(
        index=previous_fold.index,
        train_bounds=[
            previous_fold.train_bounds[0].union(previous_fold.train_bounds[0])
        ],
        test_bounds=[previous_fold.test_bounds[0].union(last_fold.test_bounds[0])],
    )
    return splits[:-2] + [merged_fold]


def _get_end(end: int | pd.Timestamp | None, index: pd.Index) -> int:
    if isinstance(end, pd.Timestamp):
        return index.get_loc(end)
    return end if end is not None else len(index)


def _get_start(
    start: int | pd.Timestamp,
    index: pd.Index,
    subtract_initial_train_window_from_start: bool,
    initial_train_window: int,
) -> int:
    if isinstance(start, pd.Timestamp):
        start = index.get_loc(start)
    if subtract_initial_train_window_from_start:
        return start - initial_train_window
    return start


def _translate_float_if_needed(window_size: int | float, length: int) -> int:
    if window_size >= 1 and isinstance(window_size, int):
        return window_size
    if window_size < 1 and isinstance(window_size, float):
        return int(window_size * length)
    raise ValueError(
        "Invalid window size, should be either a float less than 1 or an int"
        " greater than 1"
    )


# We always apply the gaps to the train bounds.
# But we shall only apply gaps if there are test bounds after/before the train bounds (depending on the type of gap we're applying).
# Otherwise, we're reducing the size of the train set without a reason.


def _apply_gap_before_test(fold: Fold, gap_before: int) -> Fold:
    return Fold(
        index=fold.index,
        train_bounds=[
            Bounds(bound.start, bound.end - gap_before)
            if _bounds_exists_after(bound, fold.test_bounds)
            else bound
            for bound in fold.train_bounds
        ],
        test_bounds=fold.test_bounds,
    )


def _apply_gap_after_test(fold: Fold, gap_after: int) -> Fold:
    return Fold(
        index=fold.index,
        train_bounds=[
            Bounds(bound.start + gap_after, bound.end)
            if _bounds_exists_before(bound, fold.test_bounds)
            else bound
            for bound in fold.train_bounds
        ],
        test_bounds=fold.test_bounds,
    )


def _bounds_exists_after(bounds: Bounds, other_bounds: list[Bounds]) -> bool:
    return any(bounds.end <= other_bound.start for other_bound in other_bounds)


def _bounds_exists_before(bounds: Bounds, other_bounds: list[Bounds]) -> bool:
    return any(bounds.start >= other_bound.end for other_bound in other_bounds)


def _merge_neighbouring_bounds(bounds: list[Bounds]) -> list[Bounds]:
    """
    Important assumption is that the bounds are sorted beforehand
    """
    merged_bounds: list[Bounds] = []
    for t in bounds:
        if merged_bounds and merged_bounds[-1].end == t.start:
            merged_bounds[-1] = Bounds(merged_bounds[-1].start, t.end)
        else:
            merged_bounds.append(t)
    return merged_bounds
