from importlib.util import find_spec
from typing import Callable, List

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline as SKLearnPipeline

from fold.models.sklearn import (
    WrapSKLearnClassifier,
    WrapSKLearnPipeline,
    WrapSKLearnRegressor,
)
from fold.transformations.sklearn import (
    WrapSKLearnFeatureSelector,
    WrapSKLearnTransformation,
)

from ..base import Composite, Pipeline, Transformation
from ..transformations.function import WrapFunction


def replace_transformation_if_not_fold_native(
    transformation: Pipeline,
) -> Pipeline:
    if isinstance(transformation, List):
        return [replace_transformation_if_not_fold_native(t) for t in transformation]
    elif isinstance(transformation, RegressorMixin):
        return WrapSKLearnRegressor(transformation)
    elif isinstance(transformation, ClassifierMixin):
        return WrapSKLearnClassifier(transformation)
    elif isinstance(transformation, Callable):
        return WrapFunction(transformation, None)
    elif isinstance(transformation, SelectorMixin):
        return WrapSKLearnFeatureSelector(transformation)
    elif isinstance(transformation, TransformerMixin):
        return WrapSKLearnTransformation(transformation)
    elif isinstance(transformation, SKLearnPipeline):
        return WrapSKLearnPipeline(transformation)
    elif isinstance(transformation, Composite):
        return transformation.clone(replace_transformation_if_not_fold_native)
    elif isinstance(transformation, Transformation):
        return transformation
    elif find_spec("fold_wrappers") is not None:
        from fold_wrappers.convenience import wrap_transformation_if_possible

        return wrap_transformation_if_possible(transformation)
    else:
        raise ValueError(f"Transformation {transformation} is not supported")
