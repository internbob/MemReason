# Phase 5C: Interpretable Dynamic Memory State Development Protocol

Status: proposed train-only redevelopment protocol. No result claim has been made.

## Primary question

Can a domain-identified, continuous-time probabilistic memory-state model recover a coherent patient state from asynchronous item observations while remaining non-inferior to GRU-D for future forecasting?

## Data boundary

- Development uses the original ADNI train subjects only.
- Use nested subject-wise cross-validation for all architecture, objective-weight, and stopping decisions.
- Original validation and test subjects have already been consumed and must not be used for selection or renewed confirmation.
- Final confirmatory evidence requires a prospectively frozen external cohort.
- Diagnosis, future FAQ, and MRI are excluded from the state encoder. They are downstream validity variables only.

## Stage 0: construct and coding audit

1. Verify AVLT and Logical Memory definitions, allowable ranges, special missing codes, direction, phase provenance, and exam dates against primary dictionaries.
2. Do not interpret `AVLT_Trial*Int` as recall until the source confirms its meaning; current distributions are consistent with intrusion counts.
3. Freeze the state item set, response family, domain assignment, loading direction, and primary/auxiliary/downstream role before modeling.
4. Report subject/visit coverage and observation co-occurrence for every retained field.

Stage 0 must pass before AVLT or Logical Memory enters the measurement model.

## Stage 1: identified measurement model

Fit a multidimensional typed item-response model with five fixed domains and memory as the primary axis. Compare:

1. population mean/LOCF;
2. confirmatory factor or multidimensional IRT model;
3. unconstrained autoencoder;
4. domain-constrained probabilistic measurement model.

Primary metrics are held-out item log likelihood, response-type-appropriate error, held-out instrument reconstruction, and cross-view posterior agreement. Every metric uses participant-cluster uncertainty estimates.

## Stage 2: continuous-time dynamics

Add continuous prediction and event-driven posterior update. Compare:

1. GRU-D;
2. typed direct Gaussian;
3. existing Phase 2 typed belief dynamics;
4. new smooth unconstrained state dynamics;
5. progressive-burden plus transient-residual dynamics.

Evaluate hidden-visit interpolation and 6/12/24/36-month future prediction with normalized MAE, item/endpoint likelihood, CRPS, interval coverage, and calibration error. The new model must be no more than 0.01 macro normalized MAE worse than GRU-D and should improve at least one state-validity endpoint with a subject-bootstrap confidence interval excluding zero.

## Stage 3: state validity

Prespecified tests:

- posterior agreement from disjoint instrument views;
- omitted-instrument decoding;
- state stability under random item thinning;
- uncertainty growth with elapsed time and reduction after informative observations;
- association of current state/slope with future FAQ and diagnosis transition without using either as input;
- association with frozen MRI representation as secondary construct validity only.

## Stage 4: reasoning faithfulness

Generate the report from executable state quantities: prior, innovation, posterior shift, information gain, progression rate, forecast, and uncertainty. Test:

- top-evidence deletion versus matched random deletion;
- score-direction counterfactual consistency;
- numeric and temporal preservation;
- evidence sufficiency and comprehensiveness;
- no future-evidence leakage.

The Phase 4C decoder may render the final schema, but LLM text quality cannot determine whether the state model passes.

## Development gate

Advance to external validation only if all are true:

1. future macro normalized MAE is non-inferior to GRU-D within +0.01;
2. held-out instrument performance is better than multidimensional IRT or the matched non-dynamic baseline with a 95% participant-bootstrap interval excluding zero;
3. cross-view posterior agreement improves over an unconstrained latent model;
4. calibrated 90%/95% coverage falls within prespecified tolerances;
5. uncertainty grows with evidence gaps and contracts after informative updates in the expected direction;
6. high-information evidence deletion causes a larger state/forecast change than matched random deletion;
7. all conclusions reproduce across nested folds without using ADNI validation/test subjects.

Failure at this gate is reported as a development negative result. Passing it permits protocol freezing and external evaluation; it does not retrospectively reopen the ADNI test set.
