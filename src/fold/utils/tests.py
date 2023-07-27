# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from random import choices
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from fold.base import Transformation, Tunable
from fold.loop import train_backtest
from fold.splitters import SingleWindowSplitter


def generate_sine_wave_data(
    cycles: int = 2, length: int = 1000, freq: str = "min"
) -> Tuple[pd.DataFrame, pd.Series]:
    end_value = np.pi * 2 * cycles
    my_wave = np.sin(np.linspace(0, end_value, length + 1))
    series = pd.Series(
        my_wave,
        name="sine",
        index=pd.date_range(end="2022", periods=len(my_wave), freq=freq),
    ).round(4)
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_all_zeros(length: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    length += 1
    series = pd.Series(
        [0] * length,
        index=pd.date_range(end="2022", periods=length, freq="m"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_zeros_and_ones(
    length: int = 1000, labels=[1, 0]
) -> Tuple[pd.DataFrame, pd.Series]:
    return generate_zeros_and_ones_skewed(length, labels, weights=[0.5, 0.5])


def generate_zeros_and_ones_skewed(
    length: int = 1000, labels=[1, 0], weights: List[float] = [0.2, 0.8]
) -> Tuple[pd.DataFrame, pd.Series]:
    length += 1
    series = pd.Series(
        choices(population=labels, weights=weights, k=length),
        index=pd.date_range(end="2022", periods=length, freq="s"),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def generate_monotonous_data(
    length: int = 1000, freq: str = "m"
) -> Tuple[pd.DataFrame, pd.Series]:
    values = np.linspace(0, 1, num=length + 1)
    series = pd.Series(
        values,
        name="linear",
        index=pd.date_range(end="2022", periods=len(values), freq=freq),
    )
    X = series.to_frame()
    y = series.shift(-1)[:-1]
    X = X[:-1]
    return X, y


def check_if_transformation_mutates(
    X: pd.DataFrame, y: pd.Series, transformation: Transformation
) -> None:
    X_train = X[: len(X) // 2]
    y_train = y[: len(y) // 2]
    X_train_before = X_train.copy()
    y_train_before = y_train.copy()
    transformation.fit(X_train, y_train)
    assert X_train_before.equals(X_train)
    assert y_train_before.equals(y_train)

    X_test = X[len(X) // 2 :]
    y_test = y[len(y) // 2 :]
    X_test_before = X_test.copy()
    y_test_before = y_test.copy()
    transformation.transform(X_test, in_sample=True)
    transformation.update(X, y)
    assert X_test_before.equals(X_test)
    assert y_test_before.equals(y_test)

    X_val = X[-1:]
    y_val = y[-1:]
    X_val_before = X_val.copy()
    y_val_before = y_val.copy()
    transformation.transform(X_val, in_sample=False)
    assert X_val_before.equals(X_val)
    assert y_val_before.equals(y_val)


def tuneability_test(
    instance: Transformation,
    different_params: dict,
    init_function: Optional[Callable] = None,
    classification: bool = False,
    tolerance: Optional[float] = None,
    modify_X_func: Optional[Callable] = None,
):
    """
    Used to test the general structure and implementation of get_params() and clone_with_params() methods.
    the kwargs are used to set the parameters of the transformation, which should be different to the init parameters of the `instance`.
    """
    assert isinstance(instance, Tunable)
    assert instance.id is not None

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
    if modify_X_func is not None:
        X = modify_X_func(X)
    splitter = SingleWindowSplitter(train_window=0.5)
    preds_orig, _ = train_backtest(instance, X, y, splitter)
    preds_different, _ = train_backtest(different_instance, X, y, splitter)
    assert not preds_orig.equals(
        preds_different
    ), "The output of the two instances with different parameters should be different"

    reconstructed_instance = different_instance.clone_with_params(params)

    preds_reconstructed, _ = train_backtest(reconstructed_instance, X, y, splitter)
    if tolerance is None:
        assert preds_orig.equals(preds_reconstructed), (
            "The output of the two instances with different parameters should be the"
            " same"
        )
    else:
        assert np.allclose(preds_orig, preds_reconstructed, atol=tolerance), (
            "The output of the two instances with the same parameters should be the"
            " close"
        )
