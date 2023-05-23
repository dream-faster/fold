# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from ...base import Artifact, Extras, Transformation, X
from ...utils.checks import is_X_available
from ...utils.dataframe import concat_on_columns
from ...utils.trim import trim_initial_nans
from ..memory import postprocess_X_y_into_memory_, preprocess_X_y_with_memory
from ..types import Stage


def _process_minibatch_transformation(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
    stage: Stage,
) -> Tuple[Transformation, X, Artifact]:
    X, y, extras = trim_initial_nans(X, y, extras)

    if not is_X_available(X) and transformation.properties.requires_X:
        raise ValueError(
            "X is None, but transformation"
            f" {transformation.__class__.__name__} requires it."
        )

    in_sample = stage == Stage.inital_fit
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(
        transformation, X, y, extras.sample_weights, in_sample=in_sample
    )
    # The order is:
    # 1. fit (if we're in the initial_fit stage)
    artifact = None
    if stage == Stage.inital_fit:
        artifact = transformation.fit(
            X_with_memory, y_with_memory, sample_weights_with_memory
        )
        postprocess_X_y_into_memory_(
            transformation,
            X_with_memory,
            y_with_memory,
            sample_weights_with_memory,
            in_sample=stage == Stage.inital_fit,
        )
        artifacts = concat_on_columns([artifact, artifacts])
    # 2. transform (inference)
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(
        transformation, X, y, extras.sample_weights, in_sample=False
    )
    return_value, artifact = transformation.transform(
        X_with_memory, in_sample=in_sample
    )
    artifacts = concat_on_columns([artifact, artifacts])
    # 3. update (if we're in the update stage)
    if stage == Stage.update:
        artifact = transformation.update(
            X_with_memory, y_with_memory, extras.sample_weights
        )
        artifacts = concat_on_columns([artifact, artifacts])
        postprocess_X_y_into_memory_(transformation, X, y, extras.sample_weights, False)
    return transformation, return_value.loc[X.index], artifacts


def _process_internal_online_model_minibatch_inference_and_update(
    transformation: Transformation,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    extras: Extras,
    artifacts: Artifact,
) -> Tuple[Transformation, X, Artifact]:
    X, y, extras = trim_initial_nans(X, y, extras)
    (
        X_with_memory,
        y_with_memory,
        sample_weights_with_memory,
    ) = preprocess_X_y_with_memory(
        transformation, X, y, extras.sample_weights, in_sample=True
    )
    postprocess_X_y_into_memory_(
        transformation, X_with_memory, y_with_memory, sample_weights_with_memory, True
    )
    return_value, artifact = transformation.transform(X_with_memory, in_sample=True)
    artifacts = concat_on_columns([artifact, artifacts])

    artifact = transformation.update(
        X_with_memory, y_with_memory, extras.sample_weights
    )
    postprocess_X_y_into_memory_(transformation, X, y, extras.sample_weights, False)
    return (
        transformation,
        return_value.loc[X.index],
        concat_on_columns([artifact, artifacts]),
    )
