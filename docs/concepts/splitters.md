![Splitters Overview](/images/technical_diagrams/splitters.svg)

### Expanding Window

Uuses all data up until the current split date.


### Sliding Window

Useful when you want to limit how much long models should look "into the past".


### Single train-test split

If you are not convinced of the usefulness of [Continuous Validation](continuous-validation.md),
feel free to use the classical single train-test split.
The downside is that you throw away data that can be used as your test set.