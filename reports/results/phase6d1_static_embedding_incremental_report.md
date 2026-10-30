# Phase 6D1: static native-T1 embedding incremental value

Original-train-only five-fold subject cross-fitting; validation and test were not loaded.

| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Latent MAE |
|---|---:|---:|---:|---:|
| cognition_only | 0.08529 | -0.7840 | 0.9497 | 0.09163 |
| embedding_static | 0.08434 | -0.7934 | 0.9505 | 0.09102 |
| metadata_only | 0.08511 | -0.7844 | 0.9494 | 0.09143 |
| probability_plus_embedding | 0.08431 | -0.7936 | 0.9507 | 0.09104 |
| probability_static | 0.08455 | -0.7902 | 0.9497 | 0.09114 |
| subject_shuffled_embedding | 0.08541 | -0.7817 | 0.9479 | 0.09153 |

## Prespecified embedding comparisons

- embedding minus `cognition_only` participant MAE: **-0.001048** (95% CI **[-0.001486, -0.000587]**); folds **4/5**, horizons noninferior **4/4**.
- embedding minus `probability_static` participant MAE: **-0.000240** (95% CI **[-0.000571, +0.000091]**); folds **3/5**, horizons noninferior **4/4**.
- embedding minus `subject_shuffled_embedding` participant MAE: **-0.000836** (95% CI **[-0.001328, -0.000349]**); folds **5/5**, horizons noninferior **4/4**.

Static embedding gate passed: **`false`**. Phase 6D2 is **closed**.
