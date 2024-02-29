# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from collections.abc import Callable
from importlib.util import find_spec

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from ..base import Clonable, Pipeline, Transformation
from ..models.sklearn import WrapSKLearnClassifier, WrapSKLearnRegressor
from ..models.wrappers.convenience import (
    _wrap_lightgbm,
    _wrap_xgboost,
)
from ..transformations.function import ApplyFunction
from ..transformations.sklearn import (
    WrapSKLearnFeatureSelector,
    WrapSKLearnTransformation,
)


def wrap_transformation_if_needed(
    transformation: Pipeline,
) -> Pipeline:
    if isinstance(transformation, list | tuple):
        return [wrap_transformation_if_needed(t) for t in transformation]
    if find_spec("xgboost") is not None and _wrap_xgboost(transformation) is not None:
        return _wrap_xgboost(transformation)  # type: ignore (we already check if it's not None)
    if find_spec("lightgbm") is not None and _wrap_lightgbm(transformation) is not None:
        return _wrap_lightgbm(transformation)  # type: ignore
    if isinstance(transformation, RegressorMixin):
        return WrapSKLearnRegressor.from_model(transformation)
    if isinstance(transformation, ClassifierMixin):
        return WrapSKLearnClassifier.from_model(transformation)
    if isinstance(transformation, Callable):
        return ApplyFunction(transformation, None)
    if isinstance(transformation, SelectorMixin):
        return WrapSKLearnFeatureSelector.from_model(transformation)
    if isinstance(transformation, TransformerMixin):
        return WrapSKLearnTransformation.from_model(transformation)
    if isinstance(transformation, Clonable):
        return transformation.clone(wrap_transformation_if_needed)
    if isinstance(transformation, Transformation):
        return transformation
    raise ValueError(f"Transformation {transformation} is not supported")
