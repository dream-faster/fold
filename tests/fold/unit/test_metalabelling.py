# from fold.base.classes import PipelineCard
# from fold.composites.metalabeling import MetaLabeling
# from fold.events.labeling.fixed import FixedForwardHorizon
# from fold.events.weights import NoWeighting
# from fold.loop.encase import train_backtest
# from fold.models.dummy import DummyClassifier
# from fold.splitters import ExpandingWindowSplitter
# from fold.utils.checks import get_probabilities_columns
# from fold.utils.tests import generate_zeros_and_ones


# def test_metalabeling() -> None:
#     X, y = generate_zeros_and_ones(1000)

#     splitter = ExpandingWindowSplitter(initial_train_window=400, step=400)
#     transformations = PipelineCard(
#         preprocessing=None,
#         event_labeler=FixedForwardHorizon(
#             time_horizon=1,
#             labeling_strategy=IdentityLabel(),
#             weighting_strategy=NoWeighting(),
#             weighting_strategy_test=NoWeighting(),
#         ),
#         pipeline=[
#             MetaLabeling(
#                 primary=[
#                     lambda x: x,
#                     DummyClassifier(
#                         predicted_value=1,
#                         all_classes=[1, 0],
#                         predicted_probabilities=[1.0, 0.0],
#                     ),
#                 ],
#                 meta=[
#                     lambda x: x,
#                     DummyClassifier(
#                         predicted_value=0.5,
#                         all_classes=[1, 0],
#                         predicted_probabilities=[0.5, 0.5],
#                     ),
#                 ],
#             ),
#         ],
#     )

#     pred, _, _, _ = train_backtest(transformations, X, y, splitter)
#     assert (get_probabilities_columns(pred).iloc[:, 1].dropna() == 0.5).all()
