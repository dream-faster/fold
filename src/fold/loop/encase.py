import importlib.util
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import mean_squared_error

from ..base import BlocksOrWrappable, OutOfSamplePredictions, TrainedPipelines
from ..splitters import ExpandingWindowSplitter, Splitter
from .backtesting import backtest
from .training import train
from .types import Backend, TrainMethod

if TYPE_CHECKING:
    from krisi import ScoreCard


def backtest_score(
    trained_pipelines: TrainedPipelines,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[str, Backend] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    silent: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[Union["ScoreCard", Dict[str, float]], OutOfSamplePredictions]:
    """
    Convenience wrapper to run backtest then scoring.
    If [`krisi`](https://github.com/dream-faster/krisi) is installed it will use it to generate a ScoreCard,
    otherwise it will run the `evaluation_func` passed in.

    Parameters
    ----------
    trained_pipelines: TrainedPipelines
        The pipeline that was already fitted
    X: Optional[pd.DataFrame]
        Exogenous Data
    y: pd.Series
        Endogenous Data (Target)
    splitter: Splitter
        A Splitter that cuts the data into folds.
    backend: Union[str, Backend] = Backend.no
        For parallelization purposes.
        -   "ray"
        -   "no"
    sample_weights: Optional[pd.Series] = None

    silent: bool = False
        Wether the pipeline should print to the console.
    krisi_args: Optional[Dict[str, Any]] = None
        Arguments that will be passed into `krisi` score function.
    evaluation_func: Callable = mean_squared_error
        Function to evaluate with if `krisi` is not available.



    Returns
    -------
    _type_
        _description_
    """
    backend = Backend.from_str(backend)

    pred = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        mutate=False,
    )

    if importlib.util.find_spec("krisi") is not None:
        from krisi import score

        scorecard = score(
            y[pred.index],
            pred.squeeze(),
            **(krisi_args if krisi_args is not None else {}),
        )
    else:
        scorecard = {
            "mean_squared_error": evaluation_func(y[pred.index], pred.squeeze())
        }
    return scorecard, pred


def train_backtest(
    transformations: BlocksOrWrappable,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Backend = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    train_method: TrainMethod = TrainMethod.parallel,
    silent: bool = False,
) -> Tuple[OutOfSamplePredictions, TrainedPipelines]:
    trained_pipelines = train(
        transformations, X, y, splitter, sample_weights, train_method, backend, silent
    )

    pred = backtest(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
    )

    return pred, trained_pipelines


def train_backtest_score(
    transformations: BlocksOrWrappable,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter = ExpandingWindowSplitter(initial_train_window=0.2, step=0.2),
    backend: Backend = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    train_method: TrainMethod = TrainMethod.parallel,
    silent: bool = False,
    krisi_args: Optional[Dict[str, Any]] = None,
    evaluation_func: Callable = mean_squared_error,
) -> Tuple[
    Union["ScoreCard", Dict[str, float]],
    OutOfSamplePredictions,
    TrainedPipelines,
]:
    trained_pipelines = train(
        transformations, X, y, splitter, sample_weights, train_method, backend, silent
    )

    scorecard, pred = backtest_score(
        trained_pipelines,
        X,
        y,
        splitter,
        backend,
        sample_weights,
        silent,
        krisi_args,
        evaluation_func,
    )

    return scorecard, pred, trained_pipelines
