# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, Optional

import pandas as pd

from ..base import Artifact, Transformation, Tunable, fit_noop


class DontUpdate(Transformation, Tunable):
    def __init__(
        self,
        transformation: Transformation,
        name: Optional[str] = None,
    ) -> None:
        self.transformation = transformation
        self.name = name or f"DontUpdate-{transformation.name}"
        self.properties = Transformation.Properties(requires_X=False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        self.transformation.fit(X, y, sample_weights)

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return self.transformation.transform(X, in_sample)

    update = fit_noop

    def get_params(self) -> dict:
        if hasattr(self.transformation, "get_params"):
            return self.transformation.get_params()
        else:
            return {}

    def get_params_to_try(self) -> Optional[dict]:
        if hasattr(self.transformation, "get_params_to_try"):
            return self.transformation.get_params_to_try()
        else:
            return None

    def clone_with_params(
        self, parameters: dict, clone_children: Optional[Callable] = None
    ) -> Tunable:
        if hasattr(self.transformation, "clone_with_params"):
            return DontUpdate(
                transformation=self.transformation.clone_with_params(parameters)
            )
        else:
            return DontUpdate(transformation=clone_children(self.transformation))
