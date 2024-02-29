def _wrap_xgboost(model):
    from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

    from .gbd import WrapXGB

    if isinstance(
        model, XGBClassifier | XGBRFClassifier | XGBRFRegressor | XGBRegressor
    ):
        return WrapXGB.from_model(model)
    return None


def _wrap_lightgbm(model):
    from lightgbm import LGBMClassifier, LGBMRegressor

    from .gbd import WrapLGBM

    if isinstance(model, LGBMClassifier | LGBMRegressor):
        return WrapLGBM.from_model(model)
    return None
