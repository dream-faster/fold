from copy import deepcopy
from typing import Callable, List

import pandas as pd

from ..transformations.base import Composite, Transformations
from ..utils.list import unique, wrap_in_list


class MetaLabelling(Composite):
    def __init__(self, primary: Transformations, meta: Transformations, primary_output_included: bool = False) -> None:
        self.primary = wrap_in_list(primary)
        self.meta = wrap_in_list(meta)
        self.primary_output_included = primary_output_included
        # fix name
        # self.name = "MetaLabelling-" + "-".join(
        #     [
        #         transformation.name if hasattr(transformation, "name") else ""
        #         for transformation in self.models
        #     ]
        # )

    def preprocess_X_secondary(self, X: pd.DataFrame, results_primary: List[pd.DataFrame], index: int) -> pd.DataFrame:
        if self.primary_output_included:
            return pd.concat([X] + results_primary, axis=1)
        else:
            return X

    def preprocess_y_secondary(self, y: pd.Series, results_primary: List[pd.DataFrame]) -> pd.Series:

        meta_y: pd.Series = pd.concat([discretized_predictions, y], axis=1).apply(
            equal_except_nan, axis=1
        )
        return meta_y
    
    def postprocess_result_secondary(
        self,
        primary_results: List[pd.DataFrame],
        secondary_results: List[pd.DataFrame],
    ) -> pd.DataFrame:
        # multiply probabilities of secondary results with predictions of primary results


    def get_child_transformations_primary(self) -> TransformationsAlwaysList:
        return self.models

    def clone(self, clone_child_transformations: Callable) -> Composite:
        return MetaLabelling(
            primary=clone_child_transformations(self.primary),
            meta=clone_child_transformations(self.meta),
            primary_output_included=self.primary_output_included,
        )
