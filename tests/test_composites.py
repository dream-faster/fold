from fold.composites import PerColumnTransform
from fold.loop.common import deepcopy_transformations


def test_composite_cloning():
    instance = PerColumnTransform([lambda x: x + 1, lambda x: x + 2])
    clone = instance.clone(clone_child_transformations=deepcopy_transformations)
    assert instance is not clone
    assert instance.pipeline is not clone.pipeline
    assert len(instance.pipeline[0]) == 2
    assert len(clone.pipeline[0]) == 2
