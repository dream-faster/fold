from .loop.backtesting import backtest
from .loop.encase import backtest_score as evaluate
from .loop.encase import train_backtest_score as train_evaluate
from .loop.inference import infer, update
from .loop.training import train, train_for_deployment
from .loop.types import Backend, TrainMethod
from .splitters import (
    ExpandingWindowSplitter,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
