# BAMBOO-TC: Thermal Conductivity Branch of BAMBOO

This branch contains the code and benchmark assets for our thermal conductivity simulation work based on **BAMBOO**, developed for predicting the thermal conductivity of organic liquids with a machine learning force field (MLFF).

The corresponding paper is:

- **Accelerated Machine Learning Force Field for Predicting Thermal Conductivity of Organic Liquids**
- DOI: https://doi.org/10.1016/j.mtener.2025.102178
- arXiv: https://arxiv.org/pdf/2512.01627

## Overview

This branch is built on top of the original **BAMBOO** framework and contains the modifications used in our thermal-conductivity study.

Compared with the default BAMBOO branch, this branch mainly includes:

- modifications to the **model architecture** and related **training code**;
- introduction of the **GEDT model** (**Graph Equivariant Differential Transformer**), which extends the original BAMBOO model for improved learning of liquid-state force fields;
- benchmark assets and simulation workflows for **thermal conductivity prediction**.

This repository is intended to provide the implementation and benchmark materials corresponding to our paper. For detailed methodology, theoretical derivations, and benchmark analysis, please refer to the paper rather than this README.

## What is changed in this branch

Relative to the default BAMBOO implementation, the main changes in this branch are:

1. **Model changes**

   - The original BAMBOO model is extended with the **GEDT** architecture.
   - The GEDT model is the major model-side update in this branch.
2. **Training code changes**

   - The corresponding training pipeline is updated to support the GEDT-based model.
   - These changes are mainly located in the `models` part and the associated training code.
3. **Thermal conductivity workflow**

   - This branch includes the workflow used for thermal conductivity simulations of organic liquids.
   - The full scientific details, including density alignment, rNEMD setup, and model acceleration, are described in the paper.

## Benchmark folder

The `benchmark/` folder contains materials for reproducing and understanding the thermal conductivity simulations reported in our work. In particular, it includes:

- **MD simulation setup files** for thermal conductivity calculations;
- the **checkpoints** used in the benchmark;
- required **input files** for simulations;
- simulation outputs such as **log files**, **temperature profiles**, and other intermediate results;
- **data post-processing** and **thermal conductivity analysis notebook**;
- **bash scripts** used to run the simulations.

Since the MD simulations for thermal conductivity are long, we provide **sliced / segmented simulation results** instead of only a single monolithic trajectory output. These segments can be used together for post-processing and analysis.

## Scope of this branch

This branch is mainly for:

- code release corresponding to our thermal conductivity work;
- reference implementation of the GEDT-enhanced BAMBOO model;
- benchmark reproduction and result inspection.

It is **not** intended to replace the original BAMBOO documentation. For general usage of BAMBOO, installation, and other workflows, please refer to the original BAMBOO repository and its main documentation.

## Citation

If you use this branch or find it helpful in your research, please cite:

```bibtex
@article{feng2025bamboo_tc,
  title = {Accelerated Machine Learning Force Field for Predicting Thermal Conductivity of Organic Liquids},
  journal = {Materials Today Energy},
  year = {2025},
  doi = {10.1016/j.mtener.2025.102178}
}
```
