# Third-party attribution

## Burgers example data

The observations in `data/Burgers/observations.npz` are repackaged from `burgers.mat` in [YuntianChen/SGA-PDE](https://github.com/YuntianChen/SGA-PDE), commit `28893ad1e78e9d38cc58cb9ab66096a4127b6aab`.

- Copyright (c) 2021 Yuntian Chen.
- License: MIT; the original notice is preserved in [`data/Burgers/LICENSE`](data/Burgers/LICENSE).
- Source paper: *Symbolic genetic algorithm for discovering open-form partial differential equations (SGA-PDE)*, [Physical Review Research 4, 023174 (2022)](https://doi.org/10.1103/PhysRevResearch.4.023174).
- Array packaging, field names and data partitions are specified in [`metadata.json`](data/Burgers/metadata.json).

## Experimental data in the movies

The movie descriptions cite the publications accompanying the public experimental datasets. The [study guide](docs/study.md#data-sources) collects these references. Dataset access and reuse follow the terms supplied by their original authors.

## Numerical libraries

SPADE uses [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/) and [SymPy](https://www.sympy.org/). These dependencies are installed separately under their respective licenses.
