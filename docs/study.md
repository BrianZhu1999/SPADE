# The SPADE study

**Discovering partial differential equations through shared symbolic computation**

Ziyuan Zhu, Keyu Hu, Yuhao Shi, Ming Bao, Yan Gao, Zhifei Chen and Minghui Lu. Ziyuan Zhu and Keyu Hu contributed equally.

SPADE identifies how algebraic operations and spatial derivatives combine to govern physical fields. Shared intermediate fields support recursive expression construction. Weak-form evaluation relates these expressions to observations, and validation selects compact equations. The study covers 16 numerical systems and six experimental systems, yielding eight scalar equations from measured fields.

[Watch the ten supplementary movies](https://brianzhu1999.github.io/SPADE/) · [Repository overview](../README.md)

## From classical equations to coupled fields

**Figure 2 — Classical PDEs.** Eight systems span nonlinear transport, dispersion, reaction, higher-order dynamics, heterogeneous media and acoustic propagation. The recovered equations are evaluated through their structure, coefficients and predictions under held-out conditions.

- [Movie 01 · Eight basic PDEs](https://brianzhu1999.github.io/SPADE/#movie-01)

**Figure 3 — Composite differential operators.** Five systems test the placement of derivatives within nonlinear expressions: gravity flow, Fujita–Storm diffusion, thin-film dynamics, nonlinear acoustics and graph-curvature flow. The animations connect observed fields to symbolic construction and subsequent field prediction.

- [Movie 02 · Five open-form PDEs](https://brianzhu1999.github.io/SPADE/#movie-02)

**Figure 4 — Coupling, noise and sparse observations.** Three-dimensional Euler dynamics connect five conservation laws. Two-dimensional Navier–Stokes dynamics connect velocity and pressure. Darcy and Gray–Scott tests examine equation recovery from noisy or spatially subsampled fields.

- [Movie 03 · Three-dimensional Euler equations](https://brianzhu1999.github.io/SPADE/#movie-03)
- [Movie 04 · Navier–Stokes and Gray–Scott](https://brianzhu1999.github.io/SPADE/#movie-04)

## From measurements to physical equations

**Figure 5 — Six experimental systems.** The experiments connect measured pressure, velocity, image intensity, surface elevation and orientational order to their differential relationships. Each movie follows the observations, equation discovery and the corresponding evaluation.

| Panels | Physical question | Movie |
| --- | --- | --- |
| a–c | How do sparse pressure and particle-velocity measurements reveal coupled acoustic balances and energy transport? | [05 · Standing-wave tube](https://brianzhu1999.github.io/SPADE/#movie-05) |
| d | How does measured room pressure reveal directional wave propagation and an equivalent sound speed? | [06 · MeshRIR acoustic field](https://brianzhu1999.github.io/SPADE/#movie-06) |
| e | How do measured wake velocities reveal conservative vorticity transport? | [07 · Cylinder wake](https://brianzhu1999.github.io/SPADE/#movie-07) |
| f | How do electrophoresis images reveal drift and directional diffusion? | [08 · Gel electrophoresis](https://brianzhu1999.github.io/SPADE/#movie-08) |
| g | How does surface elevation reveal propagation, dispersion and nonlinear differential structure? | [09 · Near-breaking surface waves](https://brianzhu1999.github.io/SPADE/#movie-09) |
| h–i | How do flow and orientation measurements reveal advection, rotation and flow alignment? | [10 · Active nematic dynamics](https://brianzhu1999.github.io/SPADE/#movie-10) |

## Data sources

The standing-wave tube measurements were collected for this study and are publicly available in [`data/StandingWaveTube/`](../data/StandingWaveTube/). The package contains the complete calibrated 385 Hz and 395 Hz records at 12 measured probe positions, the original discovery input, calibration tables and response factors, spatial and temporal coordinates, data partitions and SHA-256 checksums.

The five external experimental systems use the following published data sources. Fixed versions identify the files used in this study.

| System | Source publication | Data |
| --- | --- | --- |
| MeshRIR | Koyama et al. (2021), *MeshRIR: A Dataset of Room Impulse Responses on Meshed Grid Points for Evaluating Sound Field Analysis and Synthesis Methods*. [DOI](https://doi.org/10.1109/WASPAA52581.2021.9632672) | [Zenodo 10852693](https://doi.org/10.5281/zenodo.10852693), `S1-M3969_npy.zip` |
| Cylinder wake | Renn et al. (2026), *Experimental dataset of sub-critical cylinder wake velocity fields*. [DOI](https://doi.org/10.1038/s41597-026-07368-z) | [CaltechDATA](https://doi.org/10.22002/g2z1f-nex65), `Re_240.h5` |
| Gel electrophoresis | Both et al. (2021), *DeepMoD: Deep learning for model discovery in noisy data*. [DOI](https://doi.org/10.1016/j.jcp.2020.109985) | [DeePyMoD array](https://github.com/PhIMaL/DeePyMoD/blob/5fad44f8194a90c899d880d4fa7d6d2f2f2763a4/paper/Advection_diffusion/diffusion_advection_29longb.npy), commit `5fad44f8194a90c899d880d4fa7d6d2f2f2763a4` |
| Near-breaking surface waves | Xu et al. (2025), *Generative discovery of partial differential equations by learning from math handbooks*. [DOI](https://doi.org/10.1038/s41467-025-65114-2) | [KD-data array](https://huggingface.co/datasets/timeoutHao/KD-data/blob/24f6727e66a71c902f8466cf3ab0cae35165268b/bench/wave_breaking/WaveBreaking.pkl), revision `24f6727e66a71c902f8466cf3ab0cae35165268b` |
| Active nematics | Golden et al. (2023), *Physically informed data-driven modeling of active nematics*. [DOI](https://doi.org/10.1126/sciadv.abq6120) | [Zenodo 7249653](https://doi.org/10.5281/zenodo.7249653), `observables.mat` |
