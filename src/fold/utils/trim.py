# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.

import logging
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from ..base import Artifact

T = TypeVar("T", pd.Series, Optional[pd.Series])
logger = logging.getLogger("fold:utils")


def trim_initial_nans(
    X: pd.DataFrame, y: pd.Series, artifact: Artifact
) -> Tuple[pd.DataFrame, pd.Series, Artifact]:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any() and (y is None or not np.isnan(y.iloc[0])):
        return X, y, artifact
    first_valid_index_X = get_first_valid_index(X)
    first_valid_index_y = get_first_valid_index(y)
    if first_valid_index_X is None or first_valid_index_y is None:
        return (
            pd.DataFrame(),
            pd.Series(dtype="float64"),
            Artifact.empty(X.index),
        )
    first_valid_index = max(first_valid_index_X, first_valid_index_y)
    if first_valid_index == 0:
        return X, y, artifact
    else:
        logger.warn(
            f"The first {first_valid_index} rows of the dataset were removed because they contained NaN values."
        )
        return (
            X.iloc[first_valid_index:],
            y.iloc[first_valid_index:],
            artifact.iloc[first_valid_index:],
        )


def trim_initial_nans_single(X: pd.DataFrame) -> pd.DataFrame:
    # Optimize for speed, if the first value is not NaN, we can save all the subsequent computation
    if not X.iloc[0].isna().any():
        return X
    first_valid_index = get_first_valid_index(X)
    return X.iloc[first_valid_index:]


def get_first_valid_index(series: Union[pd.Series, pd.DataFrame]) -> int:
    if series.empty:
        return 0
    if isinstance(series, pd.DataFrame):
        return next(
            (
                idx
                for idx, (_, x) in enumerate(series.iterrows())
                if not pd.isna(x).any()
            ),
            None,
        )
    elif isinstance(series, pd.Series):
        return next(
            (idx for idx, (_, x) in enumerate(series.items()) if not pd.isna(x)),
            None,
        )
