from typing import Callable, Optional

from fold.base import Transformation, Tuneable
from fold.loop import train_backtest
from fold.splitters import SingleWindowSplitter
from fold.utils.tests import generate_sine_wave_data, generate_zeros_and_ones


def tuneability_test(
    instance: Transformation,
    different_params: dict,
    init_function: Optional[Callable] = None,
    classification: bool = False,
):
    """
    Used to test the general structure and implementation of get_params() and set_params() methods.
    the kwargs are used to set the parameters of the transformation, which should be different to the init parameters of the `instance`.
    """
    assert isinstance(instance, Tuneable)

    params = instance.get_params()
    different_instance = (
        instance.__class__(**different_params)
        if init_function is None
        else init_function(**different_params)
    )

    X, y = (
        generate_zeros_and_ones(length=500)
        if classification
        else generate_sine_wave_data(length=500)
    )
    splitter = SingleWindowSplitter(train_window=0.5)
    preds_orig, _ = train_backtest(instance, X, y, splitter)
    preds_different, _ = train_backtest(different_instance, X, y, splitter)
    assert not preds_orig.equals(preds_different)

    reconstructed_instance = different_instance
    reconstructed_instance.set_params(**params)

    preds_reconstructed, _ = train_backtest(reconstructed_instance, X, y, splitter)
    assert preds_orig.equals(preds_reconstructed)
