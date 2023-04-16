from typing import Optional, Tuple

import pandas as pd

from ..base import Transformation


def preprocess_X_y_with_memory(
    transformation: Transformation, X: pd.DataFrame, y: Optional[pd.Series]
) -> Tuple[pd.DataFrame, pd.Series]:
    if transformation._state is None or transformation.properties.memory_size is None:
        return X, y
    memory_X, memory_y = transformation._state.memory_X, transformation._state.memory_y
    non_overlapping_indices = ~memory_X.index.isin(X.index)
    memory_X, memory_y = (
        memory_X[non_overlapping_indices],
        memory_y[non_overlapping_indices],
    )
    if len(memory_X) == 0:
        return X, y
    assert len(memory_X) == len(memory_y)
    if y is None:
        return (
            pd.concat(
                [memory_X[-transformation.properties.memory_size : None], X],
                axis="index",
            ),
            y,
        )
    else:
        memory_y.name = y.name
        return pd.concat(
            [memory_X[-transformation.properties.memory_size : None], X], axis="index"
        ), pd.concat(
            [memory_y[-transformation.properties.memory_size : None], y], axis="index"
        )


def postprocess_X_y_into_memory(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    in_sample: bool,
) -> None:
    # don't update the transformation if we're in inference mode (y is None)
    if transformation.properties.memory_size is None or y is None:
        return

    window_size = (
        len(X)
        if transformation.properties.memory_size == 0
        else transformation.properties.memory_size
    )
    if in_sample:
        # store the whole training X and y
        transformation._state = Transformation.State(
            memory_X=X,
            memory_y=y,
        )
    elif transformation.properties.memory_size < len(X):
        memory_X, memory_y = (
            transformation._state.memory_X,
            transformation._state.memory_y,
        )
        memory_y.name = y.name
        #  memory requirement is greater than the current batch, so we use the previous memory as well
        transformation._state = Transformation.State(
            memory_X=pd.concat([memory_X, X], axis="index").iloc[-window_size:],
            memory_y=pd.concat([memory_y, y], axis="index").iloc[-window_size:],
        )
    else:
        transformation._state = Transformation.State(
            memory_X=X.iloc[-window_size:],
            memory_y=y.iloc[-window_size:],
        )
