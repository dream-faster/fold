# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from dataclasses import dataclass
from typing import List, Optional, Union


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
    def splits(self, length: int) -> List[Fold]:
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
    """

    def __init__(
        self,
        train_window: Union[
            int, float
        ],  # this is what you don't out of sample get predictions for
        step: Union[int, float],
        embargo: int = 0,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.window_size = train_window
        self.step = step
        self.embargo = embargo
        self.start = start
        self.end = end
        super().__init__()

    def splits(self, length: int) -> List[Fold]:
        end = self.end if self.end is not None else length
        window_size = translate_float_if_needed(self.window_size, length)
        step = translate_float_if_needed(self.step, length)
        return [
            Fold(
                order=order,
                model_index=index,
                train_window_start=index - window_size,
                train_window_end=index - self.embargo,
                update_window_start=0,  # SlidingWindowSplitter is incompatible with sequential updates
                update_window_end=0,
                test_window_start=index,
                test_window_end=min(end, index + step),
            )
            for order, index in enumerate(range(self.start + window_size, end, step))
        ]


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
    ) -> None:
        self.window_size = initial_train_window
        self.step = step
        self.embargo = embargo
        self.start = start
        self.end = end
        super().__init__()

    def splits(self, length: int) -> List[Fold]:
        end = self.end if self.end is not None else length
        window_size = translate_float_if_needed(self.window_size, length)
        step = translate_float_if_needed(self.step, length)

        return [
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
        super().__init__()

    def splits(self, length: int) -> List[Fold]:
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
