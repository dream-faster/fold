# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional, Tuple, TypeVar

import pandas as pd

from ..base import Transformation
from ..utils.dataframe import concat_on_index

T = TypeVar("T", pd.Series, Optional[pd.Series])
S = TypeVar("S", pd.Series, Optional[pd.Series])


def preprocess_X_y_with_memory(
    transformation: Transformation,
    X: pd.DataFrame,
    y: T,
    sample_weights: S,
    in_sample: bool,
) -> Tuple[pd.DataFrame, T, S]:
    if transformation.properties.disable_memory:
        return X, y, sample_weights
    memory_size = transformation.properties.memory_size
    if (
        not hasattr(transformation, "_state")
        or transformation._state is None
        or memory_size is None
    ):
        return X, y, sample_weights
    memory_X, memory_y, memory_sample_weights = (
        transformation._state.memory_X,
        transformation._state.memory_y,
        transformation._state.memory_sample_weights,
    )
    non_overlapping_indices = ~memory_X.index.isin(X.index)
    memory_X, memory_y, memory_sample_weights = (
        memory_X[non_overlapping_indices],
        memory_y[non_overlapping_indices],
        memory_sample_weights[non_overlapping_indices]
        if memory_sample_weights is not None
        else None,
    )
    if len(memory_X) == 0:
        return X, y, sample_weights
    assert len(memory_X) == len(memory_y)
    if memory_sample_weights is not None:
        assert len(memory_X) == len(memory_sample_weights)
    if y is None:
        assert sample_weights is None
        return (
            concat_on_index([memory_X.iloc[-memory_size:None], X], copy=True),
            y,
            sample_weights,
        )
    elif in_sample is True:
        memory_y.name = y.name
        return (
            concat_on_index([memory_X, X], copy=True),
            concat_on_index([memory_y, y], copy=True),
            concat_on_index([memory_sample_weights, sample_weights], copy=True),
        )
    else:
        memory_y.name = y.name
        return (
            concat_on_index([memory_X.iloc[-memory_size:None], X], copy=True),
            concat_on_index(
                [memory_y.iloc[-memory_size:None], y],
                copy=True,
            ),
            concat_on_index(
                [memory_sample_weights.iloc[-memory_size:None], sample_weights],
                copy=True,
            )
            if sample_weights is not None
            else None,
        )


def postprocess_X_y_into_memory_(
    transformation: Transformation,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    in_sample: bool,
) -> None:
    if transformation.properties.disable_memory:
        return
    # This function mutates the transformation's state property
    # don't update the transformation if we're in inference mode (y is None)
    memory_size = transformation.properties.memory_size
    if memory_size is None or y is None:
        return

    window_size = memory_size
    if in_sample:
        # store the whole training X and y
        transformation._state = Transformation.State(
            memory_X=X, memory_y=y, memory_sample_weights=sample_weights
        )
    elif memory_size < len(X):
        memory_X, memory_y, memory_sample_weights = (
            transformation._state.memory_X,
            transformation._state.memory_y,
            transformation._state.memory_sample_weights,
        )
        memory_y.name = y.name
        #  memory requirement is greater than the current batch, so we use the previous memory as well
        transformation._state = Transformation.State(
            memory_X=concat_on_index([memory_X, X], copy=True).iloc[-window_size:None],
            memory_y=concat_on_index([memory_y, y], copy=True).iloc[-window_size:None],
            memory_sample_weights=concat_on_index(
                [memory_sample_weights, sample_weights], copy=True
            ).iloc[-window_size:None]
            if sample_weights is not None
            else None,
        )
    else:
        transformation._state = Transformation.State(
            memory_X=X.iloc[-window_size:None],
            memory_y=y.iloc[-window_size:None],
            memory_sample_weights=sample_weights.iloc[-window_size:None]
            if sample_weights is not None
            else None,
        )
