# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from importlib.util import find_spec
from typing import Callable, List, Tuple

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from ..base import Clonable, Pipeline, Transformation
from ..models.sklearn import WrapSKLearnClassifier, WrapSKLearnRegressor
from ..transformations.function import ApplyFunction
from ..transformations.sklearn import (
    WrapSKLearnFeatureSelector,
    WrapSKLearnTransformation,
)


def wrap_transformation_if_needed(
    transformation: Pipeline,
) -> Pipeline:
    if isinstance(transformation, List) or isinstance(transformation, Tuple):
        return [wrap_transformation_if_needed(t) for t in transformation]
    elif isinstance(transformation, RegressorMixin):
        return WrapSKLearnRegressor.from_model(transformation)
    elif isinstance(transformation, ClassifierMixin):
        return WrapSKLearnClassifier.from_model(transformation)
    elif isinstance(transformation, Callable):
        return ApplyFunction(transformation, None)
    elif isinstance(transformation, SelectorMixin):
        return WrapSKLearnFeatureSelector.from_model(transformation)
    elif isinstance(transformation, TransformerMixin):
        return WrapSKLearnTransformation.from_model(transformation)
    elif isinstance(transformation, Clonable):
        return transformation.clone(wrap_transformation_if_needed)
    elif isinstance(transformation, Transformation):
        return transformation
    elif find_spec("fold_wrappers") is not None:
        from fold_wrappers.convenience import wrap_transformation_if_possible

        return wrap_transformation_if_possible(transformation)
    else:
        raise ValueError(f"Transformation {transformation} is not supported")
