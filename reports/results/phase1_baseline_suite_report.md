# Formal cognitive forecasting baseline suite

Status: completed under the frozen baseline protocol.

## Data and training

- Dictionary-verified scalar inputs: 68
- Forecast samples: {'train': 16577, 'val': 2268, 'test': 4951}
- Neural seeds: [2026, 2027, 2028]
- Trainable parameter counts: {'GRU': 219269, 'LSTM': 286341, 'T-LSTM': 318341, 'GRU-D': 220173, 'Transformer': 449541}
- Validation-selected strongest neural architecture: **GRU-D**
- Test data were loaded only after all validation-trained checkpoints completed.

## Test macro normalized MAE

| model | normalized_mae | coverage |
| --- | --- | --- |
| GRU-D ensemble | 0.3799 | 1 |
| GRU ensemble | 0.3818 | 1 |
| T-LSTM ensemble | 0.3857 | 1 |
| Transformer ensemble | 0.389 | 1 |
| LSTM ensemble | 0.3903 | 1 |
| LOCF | 0.3983 | 0.9622 |
| LME | 0.4193 | 1 |

## Validation seed scores

| architecture | validation_mean | validation_std |
| --- | --- | --- |
| GRU | 0.4027 | 0.001312 |
| LSTM | 0.4042 | 0.0006785 |
| T-LSTM | 0.4018 | 0.002216 |
| GRU-D | 0.4003 | 0.003553 |
| Transformer | 0.4114 | 0.003335 |

## Paired subject bootstrap

Negative normalized-MAE differences favor the neural ensemble.

| candidate | baseline | n_paired_targets | n_subjects | normalized_mae_difference | ci_lower | ci_upper | bootstrap_probability_candidate_better | replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GRU ensemble | LOCF | 19757 | 495 | -0.02142 | -0.0356 | -0.008523 | 1 | 2000 |
| GRU ensemble | LME | 20007 | 495 | -0.0376 | -0.048 | -0.02688 | 1 | 2000 |
| LSTM ensemble | LOCF | 19757 | 495 | -0.01809 | -0.03308 | -0.00509 | 0.9965 | 2000 |
| LSTM ensemble | LME | 20007 | 495 | -0.03423 | -0.04549 | -0.0226 | 1 | 2000 |
| T-LSTM ensemble | LOCF | 19757 | 495 | -0.01744 | -0.03282 | -0.004066 | 0.9935 | 2000 |
| T-LSTM ensemble | LME | 20007 | 495 | -0.03363 | -0.04473 | -0.0221 | 1 | 2000 |
| GRU-D ensemble | LOCF | 19757 | 495 | -0.0196 | -0.03425 | -0.006466 | 0.997 | 2000 |
| GRU-D ensemble | LME | 20007 | 495 | -0.03585 | -0.04617 | -0.02519 | 1 | 2000 |
| Transformer ensemble | LOCF | 19757 | 495 | -0.01741 | -0.0322 | -0.003274 | 0.993 | 2000 |
| Transformer ensemble | LME | 20007 | 495 | -0.0336 | -0.04401 | -0.02255 | 1 | 2000 |

## Interpretation limits

- Time is the nominal ADNI visit month, not a verified MRI acquisition date.
- LOCF coverage can be below one when no prior observation of a target scale exists.
- MoCA 6-month targets are sparse and must not be interpreted alone.
- This suite contains no semantic, MRI, probabilistic-belief, SFT or RL input.
- Test results cannot be used to alter this run ID or tune these baselines.
