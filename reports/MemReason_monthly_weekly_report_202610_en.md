# MemReason-RL Monthly Progress Report

**Reporting period: Four-week project summary**
**Project title: MemReason-RL — Memory-State Modeling from Asynchronous Longitudinal Cognition and MRI in ADNI**
**Current stage: Formal cognitive baselines, representation learning, probabilistic belief/calibration, MRI Phase 3A–3F, and reasoning Phase 4A–4D have been completed. Phase 4C grammar-constrained decoding passed on new structured-reasoning subjects. Phase 4D then completed real on-policy cognitive-only GRPO, but no checkpoint exceeded the SFT policy on internal development subjects; the GRPO gate failed and original-test reasoning confirmation remained closed.**

## Brief Description

This project studies memory-state modeling from incomplete and asynchronous longitudinal cognitive assessments and structural MRI in ADNI. Its long-term goals are to support future cognitive forecasting, cross-instrument inference, uncertainty estimation, and evidence-grounded longitudinal reasoning. The work began with data auditing, subject-level splitting, and nominal-month MRI matching. It then established classical and deep longitudinal baselines and sequentially evaluated fixed semantic descriptors, a response-aware semantic encoder, typed response representations, task-specific gates, and explicit probabilistic belief dynamics.

By the end of this reporting period, GRU-D remained the formally supported primary model for cognitive forecasting. The typed direct Gaussian model showed a task-specific advantage for interpolating hidden intermediate visits. Fixed semantic, response-aware semantic, and explicit predict-update belief models did not meet the complete prespecified gates. The current evidence therefore does not support a general semantic-superiority claim or progression to formal MRI Stage II under the original protocol.

## Rationale

- **Establish trustworthy data before modeling.** Longitudinal prediction is vulnerable to visit-level split leakage, future-information leakage, incorrect item coding, and inferred scan dates. The project therefore began with subject-wise splitting, strict temporal cutoffs, and item-level provenance.
- **Build strong baselines that new components must exceed.** LOCF, LME, GRU, LSTM, T-LSTM, GRU-D, and a time-aware Transformer were compared before introducing semantic representations, MRI, or reinforcement learning.
- **Evaluate semantic value through controlled comparisons.** Numeric, variable-name, task-semantic, and response-aware arms used the same inputs, tasks, training budgets, and matched model capacities wherever applicable.
- **Test representations across complementary partial-observation tasks.** Masked-item reconstruction, masked-instrument reconstruction, cross-scale prediction, and future forecasting were evaluated jointly to determine whether a representation generalized across missing-data settings.
- **Treat negative results as formal findings.** After the complete Stage I gate failed, semantic-architecture tuning on the same validation subjects was stopped to limit validation overfitting.
- **Evaluate uncertainty in addition to point predictions.** The belief-dynamics study reported MAE, Gaussian NLL, CRPS, interval coverage, calibration error, and associations between uncertainty, prediction error, and missingness.
- **Enforce stage gates.** IDA now provides auditable Study Dates for 91.52% of primary MRI scans. Phase 4B raw generation missed its strict gate because of truncation. Phase 4C selected a token-trie decoder on internal dev subjects and confirmed it once on 234 reasoning-new subjects. This opens a separately frozen cognitive-only constrained-GRPO protocol; no on-policy GRPO result exists yet.

---

# Week 1: Data Audit, Longitudinal Cohort Construction, and MRI Source Verification

## Description

Week 1 focused on data usability, temporal validity, and alignment across sources. The work covered raw cognitive-data auditing, MRI-root comparison, subject-wise splitting, asynchronous event construction, future-target indexing, and nominal-month MRI matching.

## Rationale

Model results are interpretable only when subjects are mutually exclusive across splits, prediction targets occur strictly after the history cutoff, real missing values are preserved, and the source of MRI timing is explicitly documented. This work established a common data contract for all subsequent cognitive, MRI, and longitudinal experiments.

## To-do List

### Task 1: Audit the raw cognitive-assessment data

- [x] Verified the raw CSV dimensions, fields, missingness, and visit distribution.
- [x] Confirmed 4,316 participants, 15,453 visits, and 161 original columns.
- [x] Preserved raw missing values without prior interpolation or imputation.
- [x] Flagged candidate outliers and special codes without silently changing source values.

### Task 2: Identify the primary MRI source

- [x] Compared the `ADNI_Linear`, `surfGT`, and legacy UniSurf longitudinal directories.
- [x] Selected `/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear` as the primary MRI directory.
- [x] Indexed 10,230 longitudinal T1 scans by subject and session.
- [x] Linked IDA metadata by exact Image ID and subject, adding Study Dates to 9,362 of 10,230 primary MRI scans.
- [x] Documented the remaining 868 scans without dates and the continued absence of scanner metadata and clinical QC.

### Task 3: Create subject-level data splits

- [x] Created 70%/10%/20% train/validation/test splits with seed 2026.
- [x] Obtained 3,021/431/864 mutually exclusive participants.
- [x] Assigned every cognitive visit, MRI scan, event, and future target according to subject membership.
- [x] Prohibited further row-wise splitting of derived CSV files.

### Task 4: Construct longitudinal events and forecasting indices

- [x] Generated asynchronous instrument and MRI event streams.
- [x] Generated 6-, 12-, 24-, and 36-month forecasting indices.
- [x] Ensured that every target event occurred strictly after its history cutoff.
- [x] Created same-month matches, nearest-MRI-month matches, and a conservative ±6-month subset.

## Weekly Summary

The foundational data pipeline was completed. All 4,316 participants and 15,453 cognitive visits were retained. The primary MRI source contained 10,230 T1 scans from 1,981 participants, and 9,362 scans received exact IDA Study Dates. The 90-day protocol produced 9,446 cognitive–MRI matches, of which 9,222 passed a real-date window. Train, validation, and test splits remained isolated at the subject level, and all forecasting targets occurred strictly after their history cutoffs. Clinical QC remains unavailable, so passing the date window does not establish formal grounding eligibility.

---

# Week 2: Cognitive Forecasting Baselines, Item-Code Verification, and Fair Semantic Comparison

## Description

Week 2 established robust cognitive-forecasting baselines and verified the official fields and response coding in the cognitive item dictionary. Numeric, variable-name, and task-semantic representations were then compared using the same data, tasks, model capacity, and training budget.

## Rationale

The value of semantic or multimodal modeling must be assessed against a strong numeric baseline. Because language descriptions can amplify item-coding errors, official field and response-code verification was required before evaluating semantic representations.

## To-do List

### Task 1: Complete Phase 1A numeric forecasting

- [x] Constructed numeric inputs from cognitive items, missingness masks, and nominal visit months.
- [x] Forecast ADAS13, CDR-SB, FAQ, MMSE, and MoCA at 6, 12, 24, and 36 months.
- [x] Compared LOCF, LME, GRU-D, and a time-aware Transformer.
- [x] Used multiple random seeds, endpoint-specific metrics, and subject-cluster bootstrap intervals.

### Task 2: Complete the extended baseline suite

- [x] Added GRU, LSTM, and T-LSTM baselines.
- [x] Reran GRU-D and the Transformer under a common training budget.
- [x] Completed 15 neural-model checkpoints.
- [x] Selected the model on validation data before loading the test set.

### Task 3: Verify the cognitive item dictionary and response coding

- [x] Matched 107 ADAS, CDR, FAQ, MMSE, and MoCA item fields to official sources.
- [x] Verified coding and observed values for 99 fields.
- [x] Kept eight MMSE spelling fields disabled because auditable response coding was unavailable.
- [x] Selected 68 verified scalar-comparable inputs with known response direction.

### Task 4: Complete the Phase 1B numeric-versus-semantic comparison

- [x] Compared `numeric_only`, `variable_name`, and `task_semantics` arms.
- [x] Encoded descriptors once with a locally stored frozen Bio_ClinicalBERT.
- [x] Kept model inputs, capacity, training procedure, and prediction tasks aligned across arms.
- [x] Completed formal evaluation on future forecasting and two cross-scale tasks.

## Weekly Summary

In the extended baseline suite, GRU-D achieved the best test macro normalized MAE of **0.3799**, compared with 0.3818 for GRU, 0.3857 for T-LSTM, 0.3890 for the Transformer, 0.3903 for LSTM, 0.3983 for LOCF, and 0.4193 for LME. Subject-cluster bootstrap analysis supported the pooled normalized absolute-error improvements of GRU-D over both LOCF and LME.

In Phase 1B, task semantics produced only small cross-scale improvements with confidence intervals crossing zero. Future forecasting was slightly worse than the numeric-only arm. The semantic-value gate therefore failed, and the experiment did not support a claim that fixed task descriptors produced stable gains.

---

# Week 3: Response-Aware Joint Stage I and Representation-Learning Development

## Description

Week 3 implemented a response-aware semantic encoder and jointly trained masked-item, masked-instrument, cross-scale, and future-forecasting objectives. Following the negative formal Phase 1C result, validation-only development proceeded through typed responses, metric-aligned losses, and task-specific gates.

## Rationale

A fixed question descriptor may not capture the relationship among a question, its observed response, and its scoring direction. A response-aware representation was therefore evaluated. Complete Stage I success required joint evidence across partial-observation reconstruction, cross-instrument transfer, and future forecasting rather than improvement on a single task.

## To-do List

### Task 1: Implement the response-aware semantic encoder

- [x] Encoded the task description, observed response, valid range or category label, and score direction.
- [x] Kept Bio_ClinicalBERT frozen.
- [x] Constructed matched numeric, task-semantic, and response-aware arms.
- [x] Kept 290,993 trainable parameters per arm and used the same joint-task sampling procedure.

### Task 2: Complete the formal joint Stage I experiment

- [x] Completed masked-item reconstruction.
- [x] Completed masked-instrument reconstruction.
- [x] Completed ADAS+CDR→MoCA and MMSE+CDR→ADAS13 cross-scale prediction.
- [x] Completed 6-, 12-, 24-, and 36-month future forecasting.
- [x] Ran 3 arms × 3 seeds for nine checkpoints and independently verified the artifacts.

### Task 3: Complete Phase 1D typed representation learning

- [x] Implemented continuous, binary, and ordinal typed response encoders.
- [x] Implemented an item-conditioned decoder and response-type-specific losses.
- [x] Compared joint and two-stage training.
- [x] Used train and validation data only and did not load the test set.

### Task 4: Complete Phase 1E metric-aligned gated representation learning

- [x] Implemented per-item macro reconstruction loss.
- [x] Added categorical cross-entropy and expected-score Huber loss.
- [x] Added a zero-initialized semantic residual gate.
- [x] Tested whether semantic benefits extended beyond masked-item reconstruction.

### Task 5: Complete Phase 1F task-specific gating

- [x] Added separate semantic and value gates for five task families.
- [x] Retained the Phase 1E metric-aligned loss.
- [x] Completed the prespecified final representation revision on the current validation split.
- [x] Issued a terminal decision to stop further semantic tuning on these validation subjects.

## Weekly Summary

In Phase 1C, the response-aware arm significantly improved only MMSE+CDR→ADAS13. It performed worse on masked-item and masked-instrument reconstruction, while future forecasting did not satisfy the prespecified non-inferiority margin. The complete Stage I gate therefore failed.

Phases 1D–1F showed that typed, item-conditioned representations were useful for selected tasks. In Phase 1F, the semantic arm achieved a validation macro score of **0.38334**, compared with 0.38826 for the numeric arm. Masked-item reconstruction improved significantly, and masked-instrument and future-forecasting point estimates were favorable, but their confidence intervals did not satisfy the gate. Neither cross-scale task improved consistently. Further semantic-architecture tuning on the same validation subjects was therefore stopped. The task-specific partial-observation benefit was retained, while the general semantic-superiority claim was rejected.

---

# Week 4: Irregular-Time Belief Dynamics, Calibration, and Stage Decision

## Description

Week 4 shifted the focus from semantic superiority to cognitive-only irregular-time dynamics. The experiment compared GRU-D Gaussian, typed direct Gaussian, and typed predict-update belief models across hidden-intermediate-visit interpolation and 6-, 12-, 24-, and 36-month extrapolation. Predictive variances were calibrated on a subject-disjoint calibration subset.

## Rationale

Longitudinal medical forecasting requires reliable uncertainty estimates in addition to low point-prediction error, especially when follow-up is sparse, history lengths vary, and instruments are missing. To prevent calibration leakage, model fitting, tuning, variance calibration, and final evaluation were separated by subject.

## To-do List

### Task 1: Establish an isolated dynamics and calibration protocol

- [x] Split the official training subjects into fit and tuning subsets.
- [x] Split validation subjects into calibration and evaluation subsets.
- [x] Fit normalization statistics using fit subjects only.
- [x] Kept calibration and evaluation subjects disjoint and did not load the test set.

### Task 2: Train three dynamics models

- [x] Trained the strong GRU-D Gaussian baseline.
- [x] Trained the typed direct Gaussian model.
- [x] Trained the explicit typed predict-update belief model.
- [x] Ran seeds 2026, 2027, and 2028 for each model, producing nine checkpoints.

### Task 3: Evaluate interpolation, forecasting, and calibration

- [x] Evaluated normalized MAE, Gaussian NLL, and CRPS.
- [x] Evaluated 50%, 80%, 90%, and 95% coverage and interval width.
- [x] Evaluated multilevel calibration error.
- [x] Measured Spearman associations between uncertainty, prediction error, and missingness.
- [x] Compared candidate models with GRU-D using subject-cluster bootstrap intervals.

### Task 4: Complete independent verification and stage gating

- [x] Verified fit-only normalization.
- [x] Verified calibration/evaluation subject isolation.
- [x] Verified ensemble total variance and variance-temperature calculations.
- [x] Issued the cognitive-dynamics gate and MRI-progression decision.

### Task 5: Consolidate engineering and scientific status

- [x] Updated the README, BUILD_STATUS, experiment protocols, and gate records.
- [x] Distinguished completed experiments from components that currently have interfaces only.
- [x] Retained all negative results and stopping decisions.
- [x] Defined frozen cognitive robustness/clinical probing and MRI data readiness as the next priorities.

## Weekly Summary

Across three-seed ensembles, the typed direct Gaussian model achieved the best point estimates: normalized MAE **0.3609** and calibrated NLL **0.6061**, compared with 0.3715 and 0.6163 for GRU-D Gaussian. Its overall MAE confidence interval still crossed zero, but its interpolation improvement was **-0.01437 with a 95% CI of [-0.02971, -0.00419]**, supporting its use as an interpolation-specific candidate.

The explicit typed predict-update belief model achieved a normalized MAE of 0.3813 and a calibrated NLL of 0.6919. Although its 95% coverage and calibration error met the absolute calibration criteria, it did not outperform GRU-D in accuracy or NLL. The cognitive-dynamics gate therefore failed. The month-end decision was to retain GRU-D as the confirmed primary cognitive model, retain typed direct Gaussian as an interpolation-specific candidate, and stop further semantic or belief-architecture tuning on the current validation subjects.

---

# Monthly Summary

This reporting period completed the experimental chain from real-data preparation to cognitive longitudinal modeling: data auditing and splitting, strong numeric baselines, item-code verification, fair numeric-versus-semantic comparison, response-aware joint Stage I, three validation-only representation-development phases, and irregular-time probabilistic forecasting with calibration. The project now has reproducible data products, configurations, checkpoints, predictions, bootstrap analyses, gates, and independent verification scripts.

The most defensible findings are:

1. GRU-D is the currently confirmed primary cognitive-forecasting model.
2. Typed, item-conditioned representations provide a stable task-specific benefit for masked-item reconstruction.
3. Typed direct Gaussian modeling provides a statistically supported improvement for hidden-visit interpolation.
4. Fixed task semantics, response-aware semantics, and explicit predict-update belief dynamics do not provide stable improvements across all required tasks.
5. Concurrent MRI spatial attention has reproducible subject/spatial/query dependence and a favorable test point estimate, but its overall test CI crossed zero; item-level and longitudinal MRI gates also failed, so multimodal SFT/GRPO remain closed.

## Post-reporting-period MRI Update (13 October 2026)

MRI Phase 3A evaluated whether correctly paired global structural MRI features added forecasting value beyond cognition alone. The experiment used only train and validation subjects, required a verified 90-day MRI/cognitive date window and target dates after the MRI scan, and compared cognition-only, observed MRI, shuffled-subject MRI, and shuffled-time MRI with identical architectures across three seeds. All 6,465 included scans passed the project technical checks, and the four arms shared 1,877 validation samples and 7,580 observed targets.

The observed-MRI ensemble obtained macro normalized MAE 0.3954, compared with 0.3795 for cognition-only. Its paired error difference was +0.01679 (95% CI [0.00511, 0.02951]), indicating worse validation performance. It also failed to outperform either shuffled control, and the 12- and 24-month criteria failed. The frozen global MRI-anchor gate was therefore not met. The test split remained sealed; later Phase 3B opened concurrent grounding through a separately frozen representation protocol. See the [Phase 3A report](../outputs/phase3a_global_mri_anchor_dev_20261013/report.md).

## MRI Representation Update (14 October 2026)

Phase 3B separated MRI representation quality from longitudinal fusion. It used 5,684/780 train/validation MRI anchors and compared demographics, global MRI features, and a directly trained 3D SFCN across three seeds, including different-subject MRI controls. Concurrent macro normalized MAE was 0.6031 for SFCN, 0.6736 for global MRI, and 0.7466 for demographics. SFCN improved over demographics by -0.16736 (95% CI [-0.23183, -0.11306]) and over shuffled SFCN by -0.26002 (95% CI [-0.31901, -0.20226]). Both MRI representations passed the concurrent signal criteria.

MRI did not improve 12/24-month cognitive-change prediction after controlling for current score, age, and sex. The control achieved 0.4648 normalized MAE, compared with 0.5187 for SFCN and 0.5489 for global MRI. The project may therefore proceed to a separately frozen concurrent question-to-anatomy grounding experiment, while longitudinal MRI fusion remains closed. The test split was not loaded. See the [Phase 3B report](../outputs/phase3b_mri_representation_dev_20261014/report.md).

## Spatial Grounding Update (14 October 2026)

Phase 3C froze the three validated SFCN backbones and compared parameter-matched global-pooling and target-conditioned spatial-attention heads. Spatial attention achieved normalized MAE 0.5994 versus 0.6281 for global pooling. Its paired difference was -0.02219 (95% CI [-0.04346, -0.00140]). Spatial-coordinate permutation increased error to 0.6626, query permutation increased it to 0.8054, and different-subject MRI increased it to 0.8630.

Cross-seed attention maps achieved median Spearman correlation 0.3956 and median top-20% token overlap 0.50. All frozen Phase 3C criteria passed, opening the multimodal concurrent-grounding stage. Attention centers are currently reported only in aligned-template millimeter coordinates; no atlas-region or longitudinal MRI claim is made. The test split remained sealed. See the [Phase 3C report](../outputs/phase3c_concurrent_spatial_grounding_dev_20261014/report.md).

## Multimodal Item-Grounding Update (14 October 2026)

Phase 3D combined the 68 verified cognitive item queries with the frozen 36×64 SFCN spatial tokens. It generated 22,728/3,120 train/validation masked-item and masked-instrument samples and trained parameter-matched cognition-only, global-MRI, and item-spatial-attention models for three seeds. Masked responses were explicitly zeroed in model inputs, and test subjects remained sealed.

For masked-item reconstruction, normalized MAE was 0.3787 for cognition-only, 0.3835 for global MRI, and 0.3841 for spatial attention. For masked-instrument reconstruction, the corresponding values were 0.3696, 0.3741, and 0.3776. Spatial attention was worse than cognition-only overall by +0.01106 (95% CI [0.00618, 0.01633]) and worse than global MRI by +0.00712 (95% CI [0.00291, 0.01231]). Correct subject, spatial order, and item query all carried significant information, but cross-seed attention stability was poor: median Spearman 0.0185 and top-20% overlap 0.25. The frozen item-level multimodal gate therefore failed. Phase 3C remains the supported concurrent total-score result; item-level and longitudinal MRI fusion remain closed. See the [Phase 3D report](../outputs/phase3d_multimodal_item_grounding_dev_20261014/report.md).

The next reporting period should preserve the negative multimodal findings and avoid further tuning on the same MRI validation subjects. On the cognitive-only reasoning branch, the next experiment is a separately frozen constrained-GRPO protocol with new rollout/evaluation subjects; Phase 4C confirmation subjects must not be used for hyperparameter selection.

## Frozen Test-Confirmation Update (14 October 2026)

Phase 3E loaded the sealed test split once after freezing the Phase 3C checkpoints, cohort rules, controls, and statistics. No training or checkpoint selection was performed. The evaluation included 404 test subjects, 1,905 unique MRIs, and 7,843 concurrent cognitive targets.

Spatial attention achieved normalized MAE 0.6351 versus 0.6454 for global pooling. The paired difference was -0.01088, but its 95% CI [-0.02522, 0.00427] crossed zero. Only MMSE improved significantly, below the frozen two-endpoint requirement. Subject shuffle, spatial permutation, and query permutation all degraded performance significantly, while cross-seed attention stability replicated (median Spearman 0.4018; top-20% overlap 0.50). The complete confirmatory gate therefore failed.

Secondary frozen subgroup analysis showed heterogeneity: spatial attention improved significantly in AD subjects (-0.08484, 95% CI [-0.12142, -0.04863]) and worsened significantly in MCI subjects (+0.03082, 95% CI [0.01265, 0.05157]). This is exploratory because diagnosis conditioning was not part of the trained model. The used test split must not become a tuning set; any diagnosis-conditioned revision requires a new development/confirmation split or external cohort. See the [Phase 3E report](../outputs/phase3e_frozen_spatial_grounding_test_20261014/report.md).

## Diagnosis-Heterogeneity Cross-Fit Update (14 October 2026)

Phase 3F tested the Phase 3E subgroup pattern using only the original training cohort. It assigned 1,349 subjects and 5,684 MRI scans to five subject-disjoint folds and produced out-of-fold predictions from 30 fixed-epoch models. Diagnosis was used only for stratified analysis. Validation and test outcomes were not used.

Spatial attention improved over global pooling in AD by -0.2593 (95% CI [-0.2896, -0.2310]) and in all five folds. MCI also improved by -0.0210 (95% CI [-0.0282, -0.0143]) and in all five folds, so the test-set MCI harm did not replicate. The AD-minus-MCI interaction remained significant at -0.2383 (95% CI [-0.2688, -0.2110]), indicating a larger AD benefit rather than opposite effects. Mechanism controls and cross-fold mean-map stability passed, but cross-seed top-20% token overlap was 0.375, below the frozen 0.40 threshold. The complete heterogeneity gate therefore failed. The current cohort should not be used for another diagnosis-conditioned architecture search; confirmation now requires an external cohort. See the [Phase 3F report](../outputs/phase3f_diagnosis_heterogeneity_crossfit_20261014/report.md).

## Frozen Cognitive Structured-Reasoning Update (14 October 2026)

Phase 4A created a new subject-disjoint build/dev/confirmation partition within the original training cohort: 1,336/167/168 subjects and 13,238/1,582/1,757 reports. Three GRU-D models used build-only normalization and dev-only checkpoint selection. Their ensemble mean and across-seed standard deviation supplied the forecast and confidence fields.

The experiment generated 16,577 machine-parseable five-section JSON reports. On the 1,757 confirmation reports, format validity, evidence precision/recall, numeric/time citation accuracy, domain accuracy, current-state consistency, temporal consistency, and forecast/uncertainty preservation were all 1.000. Hallucinated-evidence and future-evidence leakage rates were zero. Confirmation macro normalized forecast MAE was 0.3515; ensemble uncertainty had a descriptive Spearman association of 0.1870 with normalized absolute error. An independent verifier reconstructed every report from source events, build-only normalization, and frozen ensemble predictions. The cognitive-only Phase 4A gate passed and Phase 4B SFT development is ready. No LLM or GRPO training was performed. See the [Phase 4A report](../outputs/phase4a_frozen_cognitive_reasoning_20261014/report.md).

## Cognitive Structured-Reasoning SFT Update (14 October 2026)

Phase 4B trained a rank-16 LoRA adapter on the local DeepSeek-R1-Distill-Qwen-1.5B model. Stable subject-balanced selection produced 2,471/488/168 train/dev/confirmation reports covering 1,336/167/168 subjects. Prompt-plus-completion length was at most 3,328 tokens, with no training-target truncation. One completion-only epoch optimized 4,358,144 parameters and reached train/dev losses of 0.00149/0.00113. MRI, diagnosis, original validation/test subjects, and future observed outcomes were excluded.

On frozen raw greedy generations, the base model produced 0/168 valid reports, while SFT produced 166/168 valid reports and 131/168 exact teacher matches. All-report evidence precision/recall was 0.9881/0.9851; current-state and temporal consistency were 0.9802/0.9771; forecast and uncertainty preservation were both 0.9881; future-evidence leakage was zero. The two invalid outputs reached the frozen 1,280-token limit. Among the 166 valid outputs, evidence precision, numeric/time/domain accuracy, and forecast/uncertainty preservation were all 1.000. The strict 100% format and 99% evidence/anchor gate still failed, so `cognitive_grpo_ready=false`. See the [Phase 4B report](../outputs/phase4b_cognitive_reasoning_sft_20261014/report.md).

## Grammar-Constrained Structured-Decoding Update (15 October 2026)

Phase 4C made no additional parameter updates. Decoder selection used one report from each of the 167 Phase 4A internal-dev subjects. The frozen priority rule selected the model-driven token-trie arm after it achieved 1.000 JSON validity and anchor preservation, 0.9920 current-state consistency, and 0.9943 temporal consistency. The previous 168 Phase 4B confirmation subjects were not reused.

The selected decoder was then evaluated once on 234 subjects from the original ADNI validation split who had not participated in Phase 4A/B structured-reasoning training, development, or confirmation. Raw SFT produced 226/234 valid reports. Grammar-constrained decoding produced 234/234 valid reports, with 1.000 evidence precision/recall, numeric/time/domain accuracy, forecast and uncertainty preservation, zero hallucinated or future evidence, and current-state/temporal consistency of 0.9946/0.9940. The JSON-validity improvement over raw SFT was 0.0342 (subject-bootstrap 95% CI [0.0128, 0.0598]); all reported primary improvement intervals excluded zero. Independent artifact and provenance verification passed. `phase4c_structured_decoding_passed=true` and `cognitive_grpo_ready=true`, allowing a separately frozen cognitive-only constrained-GRPO protocol. No on-policy GRPO experiment has yet been performed. See the [Phase 4C report](../outputs/phase4c_grammar_constrained_decoding_20261015/report.md).

## Cognitive-Only Constrained GRPO Update (15 October 2026)

Phase 4D performed real on-policy optimization of the Phase 4B LoRA inside the Phase 4C enum trie. Sixty-four build subjects produced four candidates each, yielding 256 persisted rollouts with complete reward decomposition. Training completed 32 clipped GRPO updates and saved four checkpoints. Mean categorical KL was 0.01236, the maximum update-mean KL was 0.05168, 71.88% of rollout groups had reward variation, and no safety stop fired. Frozen evidence, domain, format, forecast, and uncertainty components remained auditable; state and trend actions supplied the group-relative learning signal. Real future outcomes were availability-normalized reward data and never entered prompts.

Checkpoint selection used 167 internal-dev subjects. The original SFT policy achieved a direct state/trend action composite of 0.97784. All GRPO checkpoints were lower; update 32 was best at 0.97246, with state/trend label accuracy of 0.97485/0.97006. It retained perfect anchored fields and passed KL, reward-variation, and safety criteria, but failed the required improvement over SFT. The frozen decision was therefore `cognitive_rlvr_supported=false`; original-test reasoning confirmation was not opened. An independent verifier recomputed all rollout rewards and 835 development reports. See the [Phase 4D report](../outputs/phase4d_cognitive_constrained_grpo_20261015/report.md).

## Trajectory-Action GRPO Update (15 October 2026)

Phase 4E addressed Phase 4D's constant future-outcome reward by allowing bounded residual and uncertainty actions for ADAS13, CDR-SB, FAQ, MMSE, and MoCA. It used 256/96/256 mutually exclusive action-SFT, rollout, and new development subjects from the Phase 4A build partition and excluded every Phase 4D rollout subject. Future outcomes were absent from prompts and action-SFT; they entered GRPO only as verifiable reward data.

The run completed 32 updates and 256 on-policy candidates without a safety stop. On the new development cohort, frozen GRU-D achieved normalized MAE 0.35108, while action-SFT and every GRPO checkpoint achieved 0.39970. The best checkpoint's difference from frozen was +0.04862 (95% CI [0.03620, 0.06064]); CRPS and temporal accuracy also deteriorated. Greedy action analysis showed that action-SFT and all GRPO checkpoints selected `lower+wide` for every endpoint, so GRPO did not change the initialized policy. The frozen development gate failed, original test remained unopened, and `trajectory_action_rlvr_supported=false`. See the [Phase 4E report](../outputs/phase4e_trajectory_action_grpo_20261015/report.md).

## Frozen GRU-D + Phase 4C Test Confirmation (15 October 2026)

Phase 4F locked all checkpoints, hashes, decoding settings, metrics, and gates before opening the official test cohort. No training or checkpoint selection was performed. Numeric evaluation covered 495 subjects, 4,951 forecast samples, and 20,007 available targets. Frozen GRU-D achieved macro normalized MAE 0.37510 and significantly improved over LOCF on paired covered rows: subject-mean difference -0.02144, 95% CI [-0.03913, -0.00423]. Normalized uncertainty-error Spearman was 0.1584.

Structured evaluation used one stable-hash report per subject. Raw SFT produced 488/495 valid JSON reports, compared with 495/495 for the grammar model. Grammar decoding achieved perfect evidence precision/recall, numeric/time/domain citation accuracy, forecast/uncertainty preservation, zero hallucinated evidence, and zero future leakage. Nested state consistency was 0.99515. Nested temporal consistency was 0.98990: significantly better than raw SFT by +0.01051 (95% CI [0.00121, 0.02182]), but narrowly below the frozen 0.99000 absolute threshold. This score averages label, start/end severity, and timepoint agreement across five cognitive domains per report. All numeric and timepoint anchors were preserved, while 100 of 2,475 domain-level trend labels differed across 67 of 495 reports. The complete confirmatory gate therefore failed on this single criterion. Test-based repair or threshold revision is prohibited; external validation is required for another confirmation. See the [Phase 4F report](../outputs/phase4f_frozen_grud_phase4c_test_20261015/report.md).

## Key Deliverables

- Data and engineering status: [`BUILD_STATUS.md`](../BUILD_STATUS.md)
- Project overview: [`README.md`](../README.md)
- Formal baseline suite: [`phase1_baseline_suite_formal_20261004/report.md`](../outputs/phase1_baseline_suite_formal_20261004/report.md)
- Joint Stage I experiment: [`phase1c_joint_stage1_20261004/report.md`](../outputs/phase1c_joint_stage1_20261004/report.md)
- Representation-development terminal decision: [`phase1_representation_terminal_decision.md`](experiments/phase1_representation_terminal_decision.md)
- Belief dynamics and calibration: [`phase2_cognitive_belief_dev_20261010/report.md`](../outputs/phase2_cognitive_belief_dev_20261010/report.md)
- Multimodal item grounding: [`phase3d_multimodal_item_grounding_dev_20261014/report.md`](../outputs/phase3d_multimodal_item_grounding_dev_20261014/report.md)
- Frozen test confirmation: [`phase3e_frozen_spatial_grounding_test_20261014/report.md`](../outputs/phase3e_frozen_spatial_grounding_test_20261014/report.md)
- Diagnosis cross-fit: [`phase3f_diagnosis_heterogeneity_crossfit_20261014/report.md`](../outputs/phase3f_diagnosis_heterogeneity_crossfit_20261014/report.md)
- Frozen cognitive structured reasoning: [`phase4a_frozen_cognitive_reasoning_20261014/report.md`](../outputs/phase4a_frozen_cognitive_reasoning_20261014/report.md)
- Cognitive structured-reasoning SFT: [`phase4b_cognitive_reasoning_sft_20261014/report.md`](../outputs/phase4b_cognitive_reasoning_sft_20261014/report.md)
- Grammar-constrained structured decoding: [`phase4c_grammar_constrained_decoding_20261015/report.md`](../outputs/phase4c_grammar_constrained_decoding_20261015/report.md)
- Cognitive-only constrained GRPO: [`phase4d_cognitive_constrained_grpo_20261015/report.md`](../outputs/phase4d_cognitive_constrained_grpo_20261015/report.md)
- Trajectory-action constrained GRPO: [`phase4e_trajectory_action_grpo_20261015/report.md`](../outputs/phase4e_trajectory_action_grpo_20261015/report.md)
- Frozen GRU-D + Phase 4C test confirmation: [`phase4f_frozen_grud_phase4c_test_20261015/report.md`](../outputs/phase4f_frozen_grud_phase4c_test_20261015/report.md)
