def _wrap_xgboost(model):
    from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

    from .gbd import WrapXGB

    if (
        isinstance(model, XGBRegressor)
        or isinstance(model, XGBRFRegressor)
        or isinstance(model, XGBClassifier)
        or isinstance(model, XGBRFClassifier)
    ):
        return WrapXGB.from_model(model)
    else:
        return None


def _wrap_lightgbm(model):
    from lightgbm import LGBMClassifier, LGBMRegressor

    from .gbd import WrapLGBM

    if isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor):
        return WrapLGBM.from_model(model)
    else:
        return None


def _wrap_prophet(model):
    from prophet import Prophet

    from .prophet import WrapProphet

    if isinstance(model, Prophet):
        return WrapProphet.from_model(model)
    else:
        return None


def _wrap_sktime(model):
    from sktime.forecasting.base import BaseForecaster

    from .sktime import WrapSktime

    if isinstance(model, BaseForecaster):
        return WrapSktime.from_model(model)
    else:
        return None


def _wrap_statsforecast(model):
    from statsforecast.models import _TS

    from .statsforecast import WrapStatsForecast

    if isinstance(model, _TS):
        return WrapStatsForecast.from_model(model)
    else:
        return None
