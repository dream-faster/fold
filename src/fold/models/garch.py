from fold.composites import ModelResiduals, TransformTarget
from fold.transformations.dev import Identity

from .ar import AR


class ARCH(TransformTarget):
    def __init__(self, p: int):
        super().__init__(
            wrapped_pipeline=AR(p),
            y_pipeline=lambda x: x**2,
            invert_wrapped_output=False,
        )


class GARCH(TransformTarget):
    """
    GARCH(p, q) model.
    Accepts a stationary time series (eg. returns), predicts the variance of the time series.

    CAUTION:
    The model assumes that the mean of the time series is 0.0.

    Parameters
    ----------
    p : int
        The number of lags to include in the model.
    q : int
        The number of lags to include in the residual model.
    """

    def __init__(self, p: int, q: int, ma_model_online: bool = False):
        assert p >= 0, "p must be above 0"
        assert q >= 0, "q must be above 0"

        if q > 0:
            ma_model = AR(q)
            ma_model.properties._internal_supports_minibatch_backtesting = (
                not ma_model_online
            )
        else:
            ma_model = Identity()

        garch = ModelResiduals(
            primary=AR(p),
            meta=ma_model,
        )
        super().__init__(
            wrapped_pipeline=garch,
            y_pipeline=lambda x: x**2,
            invert_wrapped_output=False,
        )
