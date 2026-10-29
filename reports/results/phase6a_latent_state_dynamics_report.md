# Phase 6A: Exploratory latent-state dynamics

Original-train-only five-fold subject cross-fitting; validation and test were not loaded.

| Arm | Latent MAE | NLL | W2 | 95% coverage | Memory composite MAE |
|---|---:|---:|---:|---:|---:|
| exact_gaussian | 0.0943 | -0.6569 | 0.0189 | 0.8494 | 0.0884 |
| latent_locf | 0.0949 | -0.5820 | 0.0214 | 0.7995 | 0.0883 |
| latent_wasserstein | 0.0910 | -0.6862 | 0.0181 | 0.8363 | 0.0858 |
| precision_filter | 0.0910 | -0.6861 | 0.0181 | 0.8359 | 0.0857 |
| reference_filter | 0.0909 | -0.6864 | 0.0181 | 0.8356 | 0.0857 |
| reference_mri_filter | 0.0905 | -0.6908 | 0.0180 | 0.8360 | 0.0851 |
| shuffled_mri_filter | 0.0910 | -0.6859 | 0.0181 | 0.8361 | 0.0858 |

Candidate-minus-LOCF participant latent-MAE difference: **-0.00346**, 95% CI **[-0.00436, -0.00255]**.

Real-minus-shuffled MRI latent MAE: **-0.00051**.

This is exploratory because the Phase 5F measurement gate remains closed; it cannot establish a confirmatory continuous-time state claim.
