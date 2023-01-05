from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Split:
    model_index: int
    train_window_start: int
    train_window_end: int
    test_window_start: int
    test_window_end: int


class Splitter:
    def splits(self, length: int) -> List[Split]:
        raise NotImplementedError


class SlidingWindowSplitter(Splitter):
    def __init__(
        self,
        train_window_size: int,
        step: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.window_size = train_window_size
        self.step = step
        self.start = start
        self.end = end

    def splits(self, length: int) -> List[Split]:
        end = self.end if self.end is not None else length
        return [
            Split(
                model_index=index,
                train_window_start=index - self.window_size,
                train_window_end=index - 1,
                test_window_start=index,
                test_window_end=min(end, index + self.step),
            )
            for index in range(self.start + self.window_size, end, self.step)
        ]


class ExpandingWindowSplitter(Splitter):
    def __init__(
        self,
        train_window_size: int,
        step: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> None:
        self.window_size = train_window_size
        self.step = step
        self.start = start
        self.end = end

    def splits(self, length: int) -> List[Split]:
        end = self.end if self.end is not None else length
        return [
            Split(
                model_index=index,
                train_window_start=self.start,
                train_window_end=index - 1,
                test_window_start=index,
                test_window_end=min(end, index + self.step),
            )
            for index in range(self.start + self.window_size, end, self.step)
        ]
