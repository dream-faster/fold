from .loop.backtesting import backtest
from .loop.encase import backtest_evaluate as evaluate
from .loop.encase import train_backtest_evaluate as train_evuate
from .loop.inference import infer, update
from .loop.training import train, train_for_deployment
from .loop.types import Backend, TrainMethod
