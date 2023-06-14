# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from importlib.util import find_spec
from typing import Callable, List

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from fold.models.sklearn import WrapSKLearnClassifier, WrapSKLearnRegressor
from fold.transformations.sklearn import (
    WrapSKLearnFeatureSelector,
    WrapSKLearnTransformation,
)

from ..base import Composite, Optimizer, Pipeline, Sampler, Transformation
from ..transformations.function import WrapFunction


def wrap_transformation_if_needed(
    transformation: Pipeline,
) -> Pipeline:
    if isinstance(transformation, List):
        return [wrap_transformation_if_needed(t) for t in transformation]
    elif isinstance(transformation, RegressorMixin):
        return WrapSKLearnRegressor.from_model(transformation)
    elif isinstance(transformation, ClassifierMixin):
        return WrapSKLearnClassifier.from_model(transformation)
    elif isinstance(transformation, Callable):
        return WrapFunction(transformation, None)
    elif isinstance(transformation, SelectorMixin):
        return WrapSKLearnFeatureSelector.from_model(transformation)
    elif isinstance(transformation, TransformerMixin):
        return WrapSKLearnTransformation.from_model(transformation)
    elif isinstance(transformation, Composite):
        return transformation.clone(wrap_transformation_if_needed)
    elif isinstance(transformation, Optimizer):
        return transformation.clone(wrap_transformation_if_needed)
    elif isinstance(transformation, Sampler):
        return transformation.clone(wrap_transformation_if_needed)
    elif isinstance(transformation, Transformation):
        return transformation
    elif find_spec("fold_wrappers") is not None:
        from fold_wrappers.convenience import wrap_transformation_if_possible

        return wrap_transformation_if_possible(transformation)
    else:
        raise ValueError(f"Transformation {transformation} is not supported")
