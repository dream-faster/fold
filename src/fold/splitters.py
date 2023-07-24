# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd


@dataclass
class Fold:
    order: int
    model_index: int
    train_window_start: int
    train_window_end: int
    update_window_start: int
    update_window_end: int
    test_window_start: int
    test_window_end: int


class Splitter:
    from_cutoff: Optional[pd.Timestamp]

    def splits(self, index: pd.Index) -> List[Fold]:
        raise NotImplementedError


def translate_float_if_needed(window_size: Union[int, float], length: int) -> int:
    if window_size >= 1 and isinstance(window_size, int):
        return window_size
    elif window_size < 1 and isinstance(window_size, float):
        return int(window_size * length)
    else:
        raise ValueError(
            "Invalid window size, should be either a float less than 1 or an int"
            " greater than 1"
        )


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
    merge_threshold: float, optional, default 0.01
        The percentage threshold for merging the last fold with the previous one if it is too small.
    """

    def __init__(
        self,
        train_window: Union[
            int, float
        ],  # this is what you don't get out of sample get predictions for
        step: Union[int, float],
        initial_window: Optional[Union[int, float]] = None,
        embargo: int = 0,
        start: int = 0,
        end: Optional[int] = None,
        merge_threshold: float = 0.01,
        from_cutoff: Optional[pd.Timestamp] = None,
    ) -> None:
        self.window_size = train_window
        self.initial_window = initial_window
        self.step = step
        self.embargo = embargo
        self.start = start
        self.end = end
        self.merge_threshold = merge_threshold
        self.from_cutoff = from_cutoff

    def splits(self, index: pd.Index) -> List[Fold]:
        length = len(index)
        merge_threshold = int(length * self.merge_threshold)
        end = self.end if self.end is not None else length
        window_size = translate_float_if_needed(self.window_size, length)
        step = translate_float_if_needed(self.step, length)
        initial_window = (
            translate_float_if_needed(self.initial_window, length)
            if self.initial_window is not None
            else window_size
        )
        first_window_start = (
            self.start + initial_window
            if initial_window is not None
            else self.start + window_size
        )

        folds = [
            Fold(
                order=order,
                model_index=index,
                train_window_start=max(index - window_size, self.start),
                train_window_end=index - self.embargo,
                update_window_start=0,  # SlidingWindowSplitter is incompatible with sequential updates
                update_window_end=0,
                test_window_start=index,
                test_window_end=min(end, index + step),
            )
            for order, index in enumerate(range(first_window_start, end, step))
        ]
        return filter_folds(
            merge_last_fold_if_too_small(folds, merge_threshold),
            index,
            self.from_cutoff,
        )


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
    embargo : int, optional
        The gap between the train and the test window, by default 0.
    start : int, optional
        The start index of the first fold, by default 0.
    end : int, optional
        The end index of the last fold, by default None.
    merge_threshold: float, optional, default 0.01
        The percentage threshold for merging the last fold with the previous one if it is too small.
    """

    def __init__(
        self,
        initial_train_window: Union[
            int, float
        ],  # this is what you don't out of sample get predictions for
        step: Union[int, float],
        embargo: int = 0,
        start: int = 0,
        end: Optional[int] = None,
        merge_threshold: float = 0.01,
        from_cutoff: Optional[pd.Timestamp] = None,
    ) -> None:
        self.window_size = initial_train_window
        self.step = step
        self.embargo = embargo
        self.start = start
        self.end = end
        self.merge_threshold = merge_threshold
        self.from_cutoff = from_cutoff

    def splits(self, index: pd.Index) -> List[Fold]:
        length = len(index)
        merge_threshold = int(length * self.merge_threshold)
        end = self.end if self.end is not None else length
        window_size = translate_float_if_needed(self.window_size, length)
        step = translate_float_if_needed(self.step, length)

        folds = [
            Fold(
                order=order,
                model_index=index,
                train_window_start=self.start,
                train_window_end=index - self.embargo,
                update_window_start=index
                - step,  # the length of the update window is the step size, see documentation
                update_window_end=index - self.embargo,
                test_window_start=index,
                test_window_end=min(end, index + step),
            )
            for order, index in enumerate(range(self.start + window_size, end, step))
        ]
        return filter_folds(
            merge_last_fold_if_too_small(folds, merge_threshold),
            index,
            self.from_cutoff,
        )


class SingleWindowSplitter(Splitter):
    """
    Creates a single fold with a fixed train and test window.
    See [the documentation](https://dream-faster.github.io/fold/concepts/splitters/) for more details.

    Parameters
    ----------
    train_window : int, float
        The training window size. If a float, it is interpreted as a fraction of the total length of the data.
    embargo : int, optional
        The gap between the train and the test window, by default 0.
    """

    def __init__(
        self,
        train_window: Union[
            int, float
        ],  # this is what you don't out of sample get predictions for
        embargo: int = 0,
    ) -> None:
        self.window_size = train_window
        self.embargo = embargo
        self.from_cutoff = None

    def splits(self, index: pd.Index) -> List[Fold]:
        length = len(index)
        window_size = translate_float_if_needed(self.window_size, length)

        return [
            Fold(
                order=0,
                model_index=0,
                train_window_start=0,
                train_window_end=window_size - self.embargo,
                update_window_start=0,
                update_window_end=0,  # SingleWindowSplitter is incompatible with sequantial updating
                test_window_start=window_size,
                test_window_end=length,
            ),
        ]


def merge_last_fold_if_too_small(splits: List[Fold], threshold: int) -> List[Fold]:
    last_fold = splits[-1]
    if last_fold.test_window_end - last_fold.test_window_start > threshold:
        return splits

    previous_fold = splits[-2]
    merged_fold = Fold(
        order=previous_fold.order,
        model_index=previous_fold.model_index,
        train_window_start=previous_fold.train_window_start,
        train_window_end=previous_fold.train_window_end,
        update_window_start=previous_fold.update_window_start,
        update_window_end=previous_fold.update_window_end,
        test_window_start=previous_fold.test_window_start,
        test_window_end=last_fold.test_window_end,
    )
    return splits[:-2] + [merged_fold]


def filter_folds(
    splits: List[Fold],
    index: pd.Index,
    from_cutoff: Optional[pd.Timestamp],
) -> List[Fold]:
    if from_cutoff is not None:
        assert isinstance(
            from_cutoff, pd.Timestamp
        ), "from_cutoff must be a Timestamp, otherwise comparison doesn't work reliably"
        return [
            split for split in splits if index[split.test_window_end - 1] >= from_cutoff
        ]
    else:
        return splits
