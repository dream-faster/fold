

### Expanding Window

The recommended splitter, uses all data up until the current split date.


### Sliding Window

Useful when the Transformations should not care about "a lot of the past".


#### Single train-test split

If you are not convinced of the usefulness of [Continuous Validation](continuous-validation.md),
feel free to use the classical single train-test split.
The downside is that you throw away data that can be used as your test set.