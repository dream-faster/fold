<!-- # Drift -->



<p align="center">
  <!-- <a href="https://img.shields.io/github/actions/workflow/status/dream-faster/drift/sphinx.yml"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/drift/sphinx.yml?logo=readthedocs"></a> -->
  <a href="https://codecov.io/gh/dream-faster/drift" >
<img src="https://codecov.io/gh/dream-faster/drift/branch/main/graph/badge.svg?token=Z7I2XSF188"/>
</a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
</p>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/drift/">
    <img src="docs/source/images/logo.png" alt="Logo" width="90" >
  </a>

<h3 align="center"> <i>(/drift/)</i></h3>
  <p align="center">
    Nowcasting with continuous evaluation (and soon, deployment)
    <br />
    <a href="https://github.com/dream-faster/drift">View Demo</a>  ~
    <a href="https://github.com/dream-faster/drift/tree/main/src/drift/examples">Check Examples</a> ~
    <a href="https://dream-faster.github.io/drift/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

**Drift** is a Nowcasting continuous evaluation/deployment library.
(also known as walk-forward evaluation)

It supports both univariate and (soon) multivariate time series.
It is from the ground-up extensible and lightweight.

**Avoid the mistakes people make with time series ML:**
- ignoring useful features otherwise available in production (value in T-1)
- accidentally using information that wouldn't otherwise be available at the time of training/evaluation (lookahead bias)

**It can train models without lookahead bias:**
- with expanding window
- with rolling window
- even with a single train/test split, if you really want it
  
**It can also help you with creating complex blended models:**
- Ensembling: (weighted) averaging the predictions of multiple models or pipelines
- Stacking: feed multiple model's predictions into a model
- Meta-labelling or residual-modelling: 
... or any combinations of the above.
Why? It [works really well for time series](https://linkinghub.elsevier.com/retrieve/pii/S0169207022001480).
  



  
<br/>

## Drift solves the following problems:

- Time series are often evaluated on single or multiple (finite) train-test splits. This is a source of grave mistakes.<br/>
**→ drift allows to simulate and evaluate your models like they would have performed, in reality/when deployed.**

- Complex models are hard to create and manage<br/>
**→ Drift supports easy composite model creation**

- Too many dependencies and an either you use-all-or-none-of-it approach<br/>
**→ Drift has very few hard dependencies (only pandas, numpy, tqdm and scikit-learn). It supports scikit-learn Pipelines by default (coming soon). It's main aim is to be as simple and few lines as possible.**

- Works well with industry standard libraries as well as with Myalo's other open source toolkits (eg.: [Krisi evaluation](https://github.com/dream-faster/krisi) or [Drift Models]([h](https://github.com/dream-faster/drift-models))<br/>
**→ Don't need to choose between `sktime`, `darts` or `statsforecast` models. We will or already support them all (coming soon)**

- Most of the time series libraries don't support distributed computing at all.<br/>
**→ Drift was built with distributed computing in mind. Your pipeline is automatically parallelized wherever it can be (coming soon)**

- Easy to deploy.<br/>
**→ Don't stop at training models. (coming soon)**


<br/>

## Installation


The project was entirely built in ``python``. 

Prerequisites

* ``python >= 3.7`` and ``pip``


Install from git directly

*  ``pip install https://github.com/dream-faster/drift/archive/main.zip ``

<br/>

## Quickstart

You can quickly train your chosen models and get predictions by running:

```python
from drift import ...
```


```
Outputs:
```



## Features

- Supports both Regression and Classification tasks
- Feature selection on an expanding/rolling window basis (otherwise a great source of lookahead bias)
- Use any scikit-learn model/pipeline natively
- Use any univariate or sequence models (wrappers provided in [drift-models](https://github.com/dream-faster/drift-models))
- Use any Deep Learning Time Series models (wrappers provided in [drift-models](https://github.com/dream-faster/drift-models))
- Mini-batch or Online learning
- Super easy syntax
- Probabilistic foreacasts (soon for Regression as well)
- Hyperparemeter optimization / Model selection (coming soon)


## Limitations

- No intermittent time series support
- No multi-step ahead forecasts (at least not for now)
- No hierarchical time series support


## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

The project uses ``isort`` and ``black`` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


