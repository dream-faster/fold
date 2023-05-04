# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Optional

import pandas as pd

from ...base import Artifact, Transformation, Tunable
from .base import EventFilter, EventLabeler
from .filters import EveryNFilter, NoFilter
from .labeling import BinarizeFixedForwardHorizon


class CreateEvents(Transformation, Tunable):

    name = "CreateEvents"
    properties = Transformation.Properties(requires_X=True)

    def __init__(
        self,
        labeler: EventLabeler,
        event_filter: EventFilter,
    ) -> None:
        self.labeler = labeler
        self.filter = event_filter

    def transform(self, X: pd.DataFrame, in_sample: bool) -> pd.DataFrame:
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        pass

    def update(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
    ) -> Optional[Artifact]:
        pass

    def get_params(self) -> dict:
        return {
            "labeler": self.labeler,
            "filter": self.filter,
        }
