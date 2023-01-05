from copy import deepcopy
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..all_types import ModelOverTime, TransformationsOverTime
from ..models.base import Model, ModelType
from ..utils.pandas import shift_and_duplicate_first_value
from ..utils.splitters import Split, Splitter


def walk_forward_train(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
    transformations_over_time: Optional[TransformationsOverTime],
) -> ModelOverTime:

    if model.type == ModelType.Univariate:
        X = shift_and_duplicate_first_value(y, 1)

    # easy to parallize this with ray
    models = [
        __train_on_window(split, X, y, model, transformations_over_time)
        for split in tqdm(splitter.splits(length=len(y)))
    ]

    idx, values = zip(*models)
    return pd.Series(values, idx).rename(model.name)


def __train_on_window(
    split: Split,
    X: pd.DataFrame,
    y: pd.Series,
    model: Model,
    transformations_over_time: Optional[TransformationsOverTime],
) -> tuple[int, Model]:
    X_train = X.iloc[split.train_window_start : split.train_window_end].to_numpy()
    y_train = y.iloc[split.train_window_start : split.train_window_end].to_numpy()

    if transformations_over_time is not None:
        current_transformations = [
            transformation_over_time[split.model_index]
            for transformation_over_time in transformations_over_time
        ]

        for transformation in current_transformations:
            X_train = transformation.transform(X_train)

    current_model = deepcopy(model)
    current_model.fit(X_train, y_train)
    return split.model_index, current_model
