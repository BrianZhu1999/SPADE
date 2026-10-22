# Running SPADE

[Repository overview](../README.md) · [Study and data sources](study.md)

## Environment

The bundled example uses Python 3.12, NumPy 2.2.6, SymPy 1.13.1 and PyTorch 2.6.0 with CUDA 11.8. The release check runs on Linux with an NVIDIA RTX 4090 (24 GB). The example configuration uses a 10 GiB computation workspace budget; field caches and candidate records also consume memory.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Verify the numerical operations

```bash
python check_semantics.py
python -m unittest discover -s tests -v
```

This check compares first- through fourth-order spatial derivatives with analytic functions, checks weak integrals and second time derivatives, and tests symbolic equivalence, dimensional rules, cache reuse and shared versus direct evaluation. A successful run prints `"checks": "passed"`. This command uses GPU 0.

## Discover the Burgers equation

```bash
python run_spade_main.py --case Burgers --device cuda:0 --batch-id example
```

`run_spade_main.py` reads [`spade_main_config.json`](../spade_main_config.json) and expands it into a complete search command. To inspect that command:

```bash
python run_spade_main.py --case Burgers --device cuda:0 --batch-id example --print-command
```

The bundled configuration uses three expression layers, up to 512 retained prefixes, a 500,000-candidate budget and 300 s for search. Operator families are visited in a fixed cycle. Training observations determine coefficients. The selected model minimizes validation relative mean-squared error plus `1e-8` times its expression-node count.

For a target vector `y` and prediction `y_pred`, relative mean-squared error is `sum((y_pred - y)**2) / sum(y**2)`. For example, a value of `0.01` corresponds to a 10% relative root-mean-square residual. Here the vectors contain weak integral measurements of the equation terms.

Use a fresh batch ID for each run. Completed result files are preserved. To run another seed:

```bash
python run_spade_main.py --case Burgers --seed 1 --device cuda:0 --batch-id example-seed1
```

## Read the output

The output directory is `results/<batch-id>/SPADE/Burgers/clean_core/seed<seed>/`.

| File | Interpretation |
|---|---|
| `result.json` | Selected equations, coefficients, validation error, complexity, resource use and source hashes |
| `candidates.json` | Candidate expression trees and training-row count |
| `weak_features.npz` | Integral measurements of the retained equation terms and target |
| `metadata_snapshot.json` | Input fields, coordinates and data split used for the run |
| `scheduler.json` | Operator-family evaluation record |

The expected Burgers terms are `Dxx(u)` and `Dx(Mul(u,u))`, with coefficients approximately `0.1` and `-0.5`. They represent the second spatial derivative of the field and the spatial derivative of its square. Their sum yields `u_t = 0.1 u_xx - 0.5 (u²)_x`.

[`examples/burgers-result.json`](../examples/burgers-result.json) contains the selected equation, frontier and measured run statistics from the release check. [`examples/validation.json`](../examples/validation.json) records the environment, analytic checks and SHA-256 hashes of the tested code and data.

## Observation format

Each input dataset occupies `data/<case>/` with two files:

- `observations.npz`: coordinate arrays `t`, `x` and optional `y`, `z`; field arrays named `field_<name>`.
- `metadata.json`: field names, spatial axes, boundary conditions, target time derivatives and training/validation/test partitions.

Fields use the shape `(trajectory, time, x[, y, z])` on a uniformly sampled Cartesian grid. Spatial axes and their array order are explicit in the metadata. Periodic boundary conditions select spectral spatial differentiation. Other boundaries use local differentiation and compact integration weights.

The Burgers package has one trajectory, 201 time samples and 256 spatial points. It uses time indices `[0, 120)` for training, `[120, 160)` for validation and `[160, 201)` for testing. The search loader reads the training and validation partitions. Each partition constructs its own differentiation and integration windows.

For additional fields, prepare both files using this schema, add the case and time budget to `case_wall_seconds` in the configuration, and invoke the same entry point. Update `data_sha256` in the metadata to the SHA-256 of `observations.npz`.

## Code and data versions

The code in this repository is the release snapshot used for the bundled example. The [study guide](study.md) connects the paper's experiments to their supplementary movies and source publications. The release verification record identifies the code and data used in the executable example.

The bundled Burgers field comes from the SGA-PDE authors' `burgers.mat`, at commit [`28893ad1e78e9d38cc58cb9ab66096a4127b6aab`](https://github.com/YuntianChen/SGA-PDE/tree/28893ad1e78e9d38cc58cb9ab66096a4127b6aab). The arrays are repackaged in NumPy format; source-file and package hashes are recorded in [`metadata.json`](../data/Burgers/metadata.json).
