from .loop.backtesting import backtest
from .loop.encase import backtest_score as evaluate
from .loop.encase import train_backtest_score as train_evaluate
from .loop.training import train
from .loop.types import Backend, TrainMethod
from .splitters import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
