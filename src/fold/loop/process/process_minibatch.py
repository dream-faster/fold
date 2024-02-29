# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from __future__ import annotations

import pandas as pd
from finml_utils.dataframes import concat_on_columns

from ...base import Artifact, Transformation, X
from ...utils.checks import is_X_available
from ..types import Stage


def _process_minibatch_transformation(
    transformation: Transformation,
    X: pd.DataFrame,
    y: pd.Series | None,
    artifacts: Artifact,
    stage: Stage,
) -> tuple[Transformation, X, Artifact]:
    if not is_X_available(X) and transformation.properties.requires_X:
        raise ValueError(
            "X is None, but transformation"
            f" {transformation.__class__.__name__} requires it."
        )

    in_sample = stage == Stage.inital_fit
    sample_weights = Artifact.get_sample_weights(artifacts)
    raw_y = Artifact.get_raw_y(artifacts)

    # The order is:
    # 1. fit (if we're in the initial_fit stage)
    artifact = None
    if stage == Stage.inital_fit:
        artifact = transformation.fit(X, y, sample_weights, raw_y)
        artifacts = concat_on_columns([artifact, artifacts])
    # 2. transform (inference)
    return_value = transformation.transform(X, in_sample=in_sample)

    return transformation, return_value.loc[X.index], artifacts.loc[X.index]
