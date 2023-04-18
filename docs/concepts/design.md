# What are the design decisions that make `Fold` different?

Ergonomy

- There's no explicit "Pipeline" class. This allows us to hand back the job of fitting a collection of models to `train()`. This enables parallelization and reduces duplicate code. See section on Composites.


Bridging the gap between Online and Mini-Batch learning.

- We allow both tabular and sequence models, in the same pipeline.

- We allow both online and mini-batch models, in the same pipeline.
If a Model has `mode` property set to `online`, the main loop creates an inner "inference & fit" loop, so the Model can update its parameters on each timestamp.

- We also give our "online" models a way to access the latest values and skip the step that'd update their parameters. This enables an efficient "quasi-online" behaviour, where the model is only re-trained (or, updated) once per fold, but can "follow" the time series data - which usually comes with signifcant increase in accuracy.


Built with Distributed Computing in mind

- Deploy your research and development pipelines to a cluster with ray, and use modin to handle out-of-memory datasets (full support for modin is coming in April).



First class support for updating deployed models, easily, as new data flows in.

- Real world is not static. Let your models adapt, without the need to re-train from scratch.


Specialized in single-step ahead forecasting.

- To really cater for the right usecases, `fold` doesn't support multi-step ahead forecasts, explicitly. [See why](/concepts/forecasting-horizon)


## What is the “Composite” class?

We want to keep the “business” of fitting models to the `train` loop.

Composite acts as a “shell” for storing Transformations and combining them in different ways, primarily via the `postpocess_results_[primary|secondary]()` function.

The `primary_transformations` are fitted first, then optionally, if `secondary_transformations` are present, the output of both transformations are passed into `postprocess_results_secondary()`.

Composites can also modify `X` and `y` via `preprocess_[X|y]_[primary|secondary]()`.

Composites enable us to:

- Merge two, entirely different set of Pipelines, like ensembling.
- Use the result of the first (primary) set of Pipeline in the second Transformations/Pipeline. (like MetaLabeling, or TransformTarget)


