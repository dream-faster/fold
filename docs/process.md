3 step process


### 1. Information gathering

- Model Selection - "What's the best model or combination of models?" 
- Hyperparameter Optimization - "What hyperparameters to use?"
- Feature selection - "What features to use?"

**These modules are trained only on the first split's "training window", then they're static, and can't be updated.**
The predictions on the first, "training window" are always "in-sample", as the models have already "seen" (were trained on) that data.

### 2. Model training

After the initial step is done, we use the information gathered, and train models or transformations on the initial training window. So we don't lose that data.

Then, there are two methods to choose from, how to go forward:

- Sequentially updating
If training mode is sequential, then update the models for each subsequent fold.
Plus if model requires continuous updates, then update the model within the fold as well.

- Parallel, independent models
If training mode is independent, then for each fold, use as many data as possible for the initial training (till `train_window_ends` for each fold)


It's important, that the results may differ. In the "Sequentially updating" scenario, the model may be stuck in some kind of local optima, while the "Parallel, independent models" always starts model training from scratch.
"Sequentially updating" mode is conceptually incompatible with [SlidingWindowSplitter](splitters.md).


### 3. Inference

Always sequentially updating