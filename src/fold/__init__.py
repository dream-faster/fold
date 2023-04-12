from .loop.backtesting import backtest
from .loop.encase import evaluate, train_backtest, train_evaluate
from .loop.training import train
from .loop.types import Backend, TrainMethod
from .splitters import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
