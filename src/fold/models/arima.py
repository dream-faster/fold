# from fold.base import Composite
# from fold.composites.residual import ModelResiduals
# from fold.composites.target import TransformTarget
# from fold.transformations.dev import Identity
# from fold.transformations.difference import Difference

# from .ar import AR


# class ARMA(ModelResiduals):
#     """
#     ARMA(p, q) model.

#     Parameters
#     ----------
#     p : int
#         The number of lags to include in the model.
#     q : int
#         The number of lags to include in the residual model.
#     ma_model_online : bool, optional
#         Whether to use the online version of the MA model, by default False.
#     """

#     properties = Composite.Properties(
#         primary_requires_predictions=True,
#         primary_only_single_pipeline=True,
#         secondary_requires_predictions=False,
#         secondary_only_single_pipeline=True,
#     )

#     def __init__(self, p: int, q: int, ma_model_online: bool = False):
#         assert p >= 0, "p must be above 0"
#         assert q >= 0, "q must be above 0"

#         if q > 0:
#             ma_model = AR(q)
#             ma_model.properties._internal_supports_minibatch_backtesting = (
#                 not ma_model_online
#             )
#         else:
#             ma_model = Identity()
#         super().__init__(
#             primary=AR(p),
#             meta=ma_model,
#         )


# class ARIMA(TransformTarget):
#     """
#     ARIMA(p, d, q) model.

#     Parameters
#     ----------
#     p : int
#         The number of lags to include in the model.
#     d : int
#         The number of times to difference the target.
#     q : int
#         The number of lags to include in the residual model.
#     ma_model_online : bool, optional
#         Whether to use the online version of the MA model, by default False.
#     """

#     def __init__(self, p: int, d: int, q: int, ma_model_online: bool = True):
#         assert p >= 0, "p must be above 0"
#         assert q >= 0, "q must be above 0"
#         assert d >= 0, "d must be above 0"

#         arma = ARMA(p, q, ma_model_online)
#         differencers = Identity() if d == 0 else [Difference(1) for _ in range(0, d)]

#         super().__init__(wrapped_pipeline=arma, y_pipeline=differencers)
