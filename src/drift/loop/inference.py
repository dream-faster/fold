from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..all_types import (
    InSamplePredictions,
    OutSamplePredictions,
    TransformationsOverTime,
)
from ..models.base import Model
from ..transformations.base import Transformation
from ..utils.splitters import Split, Splitter


def walk_forward_inference(
    transformations_over_time: TransformationsOverTime,
    X: pd.DataFrame,
    splitter: Splitter,
    name: Optional[str] = None,
) -> tuple[InSamplePredictions, OutSamplePredictions]:

    results = [
        __inference_from_window(
            split,
            X,
            transformations_over_time,
        )
        for split in tqdm(splitter.splits(length=len(X)))
    ]
    insample_idx, insample_values, outofsample_idx, outofsample_values = zip(*results)

    insample_idx = insample_idx[-1]
    insample_values = insample_values[-1]
    outofsample_idx = np.concatenate(outofsample_idx)
    outofsample_values = np.concatenate(outofsample_values)

    insample_predictions = pd.Series(
        insample_values, X.index[insample_idx[0] : insample_idx[-1] + 1]
    ).rename("insample_predictions")
    outofsample_predictions = pd.Series(
        outofsample_values, X.index[outofsample_idx[0] : outofsample_idx[-1] + 1]
    ).rename("outofsample_predictions")
    return insample_predictions, outofsample_predictions


def __inference_from_window(
    split: Split,
    X: pd.DataFrame,
    transformations_over_time: TransformationsOverTime,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model: Model = transformations_over_time[-1][split.model_index]
    X_train = X.iloc[split.train_window_start : split.train_window_end]

    current_transformations = [
        transformation_over_time.loc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]
    for transformation in current_transformations:
        if isinstance(transformation, Transformation):
            X_train = transformation.transform(X_train)
        elif isinstance(transformation, Model):
            X_train = transformation.predict(X_train)
        else:
            raise ValueError(
                f"{type(transformation)} is not a Drift Model or Transformation."
            )

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    for transformation in current_transformations:
        if isinstance(transformation, Transformation):
            X_test = transformation.transform(X_test)
        elif isinstance(transformation, Model):
            X_test = transformation.predict(X_test)
        else:
            raise ValueError(
                f"{type(transformation)} is not a Drift Model or Transformation."
            )

    insample_predictions = model.predict(X_train)
    insample_idx = np.arange(
        split.train_window_start,
        split.train_window_end,
        1,
    )

    outofsample_predictions = model.predict(X_test)
    outofsample_idx = np.arange(
        split.test_window_start,
        split.test_window_end,
        1,
    )

    return insample_idx, insample_predictions, outofsample_idx, outofsample_predictions
