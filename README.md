# Fractional Brownian Motion-Based Inpainting

This repository provides implementations of diffusion-based inpainting methods for reconstructing incomplete Euclidean distance matrices (EDMs) of fractional Brownian motion (fBm), as presented at [NeurIPS 2024](https://neurips.cc/virtual/2024/100031). The proposed methods leverage the intrinsic memory properties of fBm to perform statistical imputation, outperforming standard bioinformatics techniques in applications such as chromosomal segment reconstruction.
## Overview
Fractional Brownian motion (fBm) is a stochastic process with both randomness and long-range correlations. This project examines how diffusion models can effectively reconstruct missing data in corrupted fBm-derived EDMs. The framework is evaluated on synthetic and experimental datasets, including:
- A dataset of fBm-derived EDMs at various memory exponents (H values).
- Fluorescence In Situ Hybridization (FISH) microscopy data with missing chromosomal segment distances.

## Features
- Implements multiple inpainting techniques, including:
  - **DDNM-based inpainting**, a generalization of DDRM, RePaint.
  - **Optimization-based reconstruction** methods.
- Provides scripts to generate and evaluate corrupted EDMs.
- Computes **Frechet Inception Distance (FID)** to assess the quality of reconstructions.

## Workflow
1. **Dataset Generation:** Use `generate_exact_distance_matrices.py` to create fBm-derived EDMs.
2. **Training:** Train a diffusion model using `train.py` to generalize samples from fBm EDMs.
3. **Testing Reconstruction:** Generate corrupted EDMs using `generate_corrupted_mask.py`.
4. **Baseline Comparison:** Evaluate different reconstruction methods using:
   - `reconstruct_database_search.py`
   - `reconstruct_fista.py`
   - `reconstruct_traj_optimization.py`
   - `reconstruct_nearest_neighbor.py`
5. **Main Reconstruction Method:** Run `reconstruct_inpainting.py` to apply diffusion-based inpainting and compare results with baselines.

## Citation
If you use this code in your research, please cite our work:
```
@inproceedings{lobashevfbm,
  title={fBm-Based Generative Inpainting for the Reconstruction of Chromosomal Distances},
  author={Lobashev, Alexander and Guskov, Dmitry and Polovnikov, Kirill},
  booktitle = {Machine Learning and the Physical Sciences Workshop at NeurIPS},
  year      = {2024},
  month     = {December},
address   = {Vancouver, Canada}
}
```

## License
This project is licensed under the MIT License.

