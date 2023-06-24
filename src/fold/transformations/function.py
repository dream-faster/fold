# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, Optional, Tuple

import pandas as pd

from fold.base.classes import Tunable

from ..base import Artifact, Transformation, fit_noop


class ApplyFunction(Transformation, Tunable):
    """
    Wraps and arbitrary function that will run at inference.
    """

    def __init__(
        self,
        func: Callable,
        past_window_size: Optional[int],
        name: Optional[str] = None,
        params_to_try: Optional[dict] = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.params_to_try = params_to_try
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size
        )

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        return self.func(X), None

    fit = fit_noop
    update = fit
