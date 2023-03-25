from .backtesting import backtest
from .encase import backtest_score
from .encase import backtest_score as evaluate
from .encase import train_backtest_score
from .encase import train_backtest_score as train_evaluate
from .inference import infer, update
from .training import train, train_for_deployment
from .types import Backend, TrainMethod
