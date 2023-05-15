# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from ..base import Artifact, OutOfSamplePredictions, TrainedPipelines
from ..splitters import Splitter
from ..utils.dataframe import concat_on_index
from ..utils.trim import trim_initial_nans_single
from .checks import check_types
from .common import _backtest_on_window
from .types import Backend


def backtest(
    trained_pipelines: TrainedPipelines,
    X: Optional[pd.DataFrame],
    y: pd.Series,
    splitter: Splitter,
    backend: Union[Backend, str] = Backend.no,
    sample_weights: Optional[pd.Series] = None,
    silent: bool = False,
    mutate: bool = False,
    return_artifacts: bool = False,
) -> Union[OutOfSamplePredictions, Tuple[OutOfSamplePredictions, Artifact]]:
    """
    Run backtest on TrainedPipelines and given data.

    Parameters
    ----------

    trained_pipelines: TrainedPipelines
        The fitted pipelines, for all folds.
    X: pd.DataFrame, optional
        Exogenous Data.
    y: pd.Series
        Endogenous Data (Target).
    splitter: Splitter
        Defines how the folds should be constructed.
    backend: str, Backend = Backend.no
        The library/service to use for parallelization / distributed computing, by default `no`.
    sample_weights: pd.Series, optional = None
        Weights assigned to each sample/timestamp, that are passed into models that support it, by default None.
    silent: bool = False
        Wether the pipeline should print to the console, by default False.
    mutate: bool = False
        Whether `trained_pipelines` should be mutated, by default False. This is discouraged.

    Returns
    -------
    OutOfSamplePredictions
        Predictions for all folds, concatenated.
    """
    backend = Backend.from_str(backend)
    X, y = check_types(X, y)

    results, artifacts = zip(
        *[
            _backtest_on_window(
                trained_pipelines,
                split,
                X,
                y,
                sample_weights,
                backend,
                mutate=mutate,
            )
            for split in tqdm(splitter.splits(length=len(X)), disable=silent)
        ]
    )
    results = trim_initial_nans_single(concat_on_index(results))
    if return_artifacts:
        return results, concat_on_index(artifacts)
    else:
        return results
