# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, Optional, Tuple

import pandas as pd

from ..base import Artifact, Transformation, fit_noop


class WrapFunction(Transformation):
    """
    Wraps and arbitrary function that will run at inference.
    """

    def __init__(self, func: Callable, past_window_size: Optional[int]) -> None:
        self.func = func
        self.name = func.__name__
        self.properties = Transformation.Properties(
            requires_X=True, memory_size=past_window_size
        )
        super().__init__()

    def transform(
        self, X: pd.DataFrame, in_sample: bool
    ) -> Tuple[pd.DataFrame, Optional[Artifact]]:
        return self.func(X), None

    fit = fit_noop
    update = fit
