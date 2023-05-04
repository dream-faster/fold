# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import List, Optional, Tuple

import pandas as pd

from ..base import Artifact, Composite, T
from .base import EventFilter, EventLabeler
from .filters import EveryNFilter, NoFilter
from .labeling import BinarizeFixedForwardHorizon


class CreateEvents(Composite):
    def __init__(
        self,
        labeler: EventLabeler,
        event_filter: EventFilter,
    ) -> None:
        self.labeler = labeler
        self.filter = event_filter

    def preprocess_primary(
        self, X: pd.DataFrame, index: int, y: T, fit: bool
    ) -> Tuple[pd.DataFrame, T]:
        start_times = self.filter.get_event_start_times(y)
        events = self.labeler.label_events(start_times, y)
        return X.loc[start_times], events["label"]

   def postprocess_result_primary(
        self, results: List[pd.DataFrame], y: Optional[pd.Series]
    ) -> pd.DataFrame:
        
        

