from fold.loop import train_backtest
from fold.models.wrappers.gbd import WrapLGBM
from fold.splitters import ExpandingWindowSplitter, SingleWindowSplitter
from fold.utils.tests import (
    generate_monotonous_data,
    generate_sine_wave_data,
    generate_zeros_and_ones,
    generate_zeros_and_ones_skewed,
    tuneability_test,
)


def test_lgbm_regression() -> None:
    from lightgbm import LGBMRegressor

    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM.from_model(LGBMRegressor())

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
    tuneability_test(
        WrapLGBM.from_model(LGBMRegressor(n_estimators=1)),
        different_params=dict(n_estimators=100),
        init_function=lambda **kwargs: WrapLGBM.from_model(LGBMRegressor(**kwargs)),
    )


def test_lgbm_classification() -> None:
    from lightgbm import LGBMClassifier

    X, y = generate_zeros_and_ones()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM.from_model(LGBMClassifier())
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert "predictions_LGBMClassifier" in pred.columns
    assert "probabilities_LGBMClassifier_0.0" in pred.columns
    assert "probabilities_LGBMClassifier_1.0" in pred.columns


def test_lgbm_classification_skewed() -> None:
    from lightgbm import LGBMClassifier

    X, y = generate_zeros_and_ones_skewed()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM.from_model(
        LGBMClassifier(), set_class_weights="balanced"
    )
    pred, _ = train_backtest(transformations, X, y, splitter)
    assert "predictions_LGBMClassifier" in pred.columns
    assert "probabilities_LGBMClassifier_0.0" in pred.columns
    assert "probabilities_LGBMClassifier_1.0" in pred.columns


def test_automatic_wrapping_lgbm() -> None:
    from lightgbm import LGBMRegressor

    X, y = generate_monotonous_data()
    train_backtest(
        LGBMRegressor(),
        X,
        y,
        splitter=SingleWindowSplitter(0.5),
    )


def test_lgbm_init_with_args() -> None:
    from lightgbm import LGBMRegressor

    X, y = generate_sine_wave_data()

    splitter = ExpandingWindowSplitter(initial_train_window=500, step=100)
    transformations = WrapLGBM(LGBMRegressor, {"n_estimators": 100})

    pred, _ = train_backtest(transformations, X, y, splitter)
    assert (y.squeeze()[pred.index] - pred.squeeze()).abs().sum() < 20
