from importlib.util import find_spec
from typing import Callable, List

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline

from fold.models.base import Model
from fold.models.sklearn import SKLearnClassifier, SKLearnPipeline, SKLearnRegressor
from fold.transformations.sklearn import SKLearnFeatureSelector, SKLearnTransformation

from ..base import BlocksOrWrappable, Composite, Transformation, Transformations
from ..transformations.function import WrapFunction


def replace_transformation_if_not_fold_native(
    transformation: BlocksOrWrappable,
) -> Transformations:
    if isinstance(transformation, List):
        return [replace_transformation_if_not_fold_native(t) for t in transformation]
    elif isinstance(transformation, RegressorMixin):
        return SKLearnRegressor(transformation)
    elif isinstance(transformation, ClassifierMixin):
        return SKLearnClassifier(transformation)
    elif isinstance(transformation, Callable):
        return WrapFunction(transformation, None)
    elif isinstance(transformation, SelectorMixin):
        return SKLearnFeatureSelector(transformation)
    elif isinstance(transformation, TransformerMixin):
        return SKLearnTransformation(transformation)
    elif isinstance(transformation, Pipeline):
        return SKLearnPipeline(transformation)
    elif isinstance(transformation, Composite):
        return transformation.clone(replace_transformation_if_not_fold_native)
    elif find_spec("fold_models") is not None and isinstance(transformation, Model):
        from fold_models.convenience import wrap_transformation_if_possible

        return wrap_transformation_if_possible(transformation)
    elif isinstance(transformation, Transformation):
        return transformation
    else:
        raise ValueError(f"Transformation {transformation} is not supported")
