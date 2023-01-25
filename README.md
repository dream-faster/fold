<!-- # Drift -->



<p align="center">
  <!-- <a href="https://img.shields.io/github/actions/workflow/status/dream-faster/drift/sphinx.yml"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/drift/sphinx.yml?logo=readthedocs"></a> -->
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
    Nowcasting on a rolling/expanding window basis.
    <br />
    <a href="https://github.com/dream-faster/drift">View Demo</a>  ~
    <a href="https://github.com/dream-faster/drift/tree/main/src/drift/examples">Check Examples</a> ~
    <a href="https://dream-faster.github.io/drift/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

**drift** is a Nowcasting training and prediction library  on a rolling/expanding window basis (single step ahead prediction on time-series).

drift is from the ground-up extensible and lightweight.

**It can train models without lookahead bias:**
- expanding window
- rolling window
  
**It can also help you with creating complex blended models:**
- ensemble multiple models, 
- stack multiple models,
- meta-label outputs of models

  


  
<br/>

## drift solves the following problems

- Nowcasting is often evaluated on single or multiple (finite) train-test splits.<br/> **→ drift allows to simulate and evaluate like it would be in reality.**
- Complex models have to be individually hard-coded<br/>
**→ drift supports composite model creation**
- Too many dependencies or walled gardens<br/>
**→ drift has few hard dependencies (only core libarries, eg.: sklearn and plotting libraries).**
- Works well in a modular fashion, with Myalo's other open source toolkits (eg.: [Krisi evaluation](https://github.com/dream-faster/krisi) or [Drift Models]([h](https://github.com/dream-faster/drift-models))


<br/>

## Installation


The project was entirely built in ``python``. 

Prerequisites

* ``python >= 3.7`` and ``pip``


Install from git directly

*  ``pip install https://github.com/dream-faster/drift/archive/main.zip ``

<br/>

## Quickstart

You can quickly evaluate your predictions by running:

```python
from drift import ...
```


```
Outputs:
```




## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

The project uses ``isort`` and ``black`` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.


