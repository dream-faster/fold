# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from importlib.util import find_spec
from typing import Callable, List, Tuple

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from ..base import Clonable, Pipeline, Transformation
from ..models.sklearn import WrapSKLearnClassifier, WrapSKLearnRegressor
from ..models.wrappers.convenience import (
    _wrap_lightgbm,
    _wrap_prophet,
    _wrap_sktime,
    _wrap_statsforecast,
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
    if isinstance(transformation, List) or isinstance(transformation, Tuple):
        return [wrap_transformation_if_needed(t) for t in transformation]
    elif find_spec("xgboost") is not None and _wrap_xgboost(transformation) is not None:
        return _wrap_xgboost(transformation)  # type: ignore (we already check if it's not None)
    elif (
        find_spec("lightgbm") is not None and _wrap_lightgbm(transformation) is not None
    ):
        return _wrap_lightgbm(transformation)  # type: ignore
    elif find_spec("prophet") is not None and _wrap_prophet(transformation) is not None:
        return _wrap_prophet(transformation)  # type: ignore
    elif find_spec("sktime") is not None and _wrap_sktime(transformation) is not None:
        return _wrap_sktime(transformation)  # type: ignore
    elif (
        find_spec("statsforecast") is not None
        and _wrap_statsforecast(transformation) is not None
    ):
        return _wrap_statsforecast(transformation)  # type: ignore
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
    else:
        raise ValueError(f"Transformation {transformation} is not supported")
