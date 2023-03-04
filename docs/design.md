# What are the design decisions that make `Fold` different?

1. There's no explicit "Pipeline" class. This allows us to hand back the job of fitting a collection of models to `train()`. This enables parallelization and reduces duplicate code. See section on Composites.


2. We allow both tabular and sequence models, in the same pipeline.
   If a Model has `requires_continuous_updates` property set to `True`, the main loop creates an inner "inference & fit" loop, so the Model can update its parameters on each timestamp.






### Why is “Composite” necessary?

We want to keep the “business” of fitting models to the `train` loop.

Composite acts as a “shell” for storing Transformations and combining them in different ways, primarily via the `postpocess_results_[primary|secondary]()` function.

The `primary_transformations` are fitted first, then optionally, if `secondary_transformations` are present, the output of both transformations are passed into `postprocess_results_secondary()`.

Composites can also modify `X` and `y` via `preprocess_[X|y]_[primary|secondary]()`.

Composites enable us to:

- Merge two, entirely different set of Transformations/Pipelines, like ensembling.
- Use the result of the first (primary) set of Transformations/Pipeline in the second Transformations/Pipeline. (like MetaLabeling, or TargetTransformation)