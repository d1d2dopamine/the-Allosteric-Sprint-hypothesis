# ADHD Pupil Dataset — Corrected Behavioral Results

## Dataset

- Figshare record: `7218725`, version 3
- DOI: https://doi.org/10.6084/m9.figshare.7218725
- Data descriptor: https://doi.org/10.1038/s41597-019-0037-2
- License: CC BY 4.0
- Source file: `Pupil_dataset.mat`
- Verified bytes: `1,257,809,856`
- Verified MD5: `d4a1e92c8e125e93831f12797a783d52`
- Verified SHA-256: `44aa997e37815e7d2a003a4fc4e967f69438a86bdf04650b02f37aaa2a81819b`

## Analysis

Validator: `ADHD_PUPIL_VALIDATOR.py`, real-data run from `0.2.0-candidate`.

Decoded:

- 50 participants
- 67 sessions
- 10,720 trials
- 28 ADHD off medication
- 22 controls
- 17 paired ADHD on-medication sessions

Rules:

- Missing `Perform` scored as no-response/omission error.
- Accuracy uses all trials.
- Error rate is not tested as a duplicate of accuracy.
- Distractor levels `3`, `4`, `5`, and `6` are treated categorically.
- Distractor ranges are order-invariant secondary endpoints.
- Pupil vectors are excluded from the reported primary analysis.

## Primary ADHD-off vs control results

| Endpoint | ADHD − control difference | Permutation p | Holm p | Interpretation |
|---|---:|---:|---:|---|
| Median correct RT | −19.5 ms | 0.4479 | 0.4479 | No uniform slowing |
| RT MAD | +61.75 ms | 0.0022 | 0.0088 | Greater robust RT variability |
| RT IQR | +99.625 ms | 0.0024 | 0.0088 | Greater robust RT variability |
| Accuracy | −0.275 | 0.0002 | 0.0010 | Lower all-trial accuracy |
| Omission rate | +0.0375 | 0.1054 | 0.2108 | Median difference not significant |

Brown–Forsythe tests also indicated greater dispersion for RT MAD and IQR (`p < 0.0001`).

## Secondary results

Temporal slopes, load effects, and four-level distractor-range endpoints did not survive Holm correction. Nominal distractor-range p-values are exploratory only.

No paired medication endpoint among 17 participants survived Holm correction. The ADHD-only GMM failed one or more acceptance gates:

```text
stable_two_cluster = False
```

## Interpretation

This dataset independently supports the broad behavioral premise that ADHD-related impairment can involve unstable performance rather than uniform slowing. It strengthens a task-dependent dimensional formulation but does not establish stable subtypes, medication effects, pupil-linked arousal, dopamine or ATP mechanisms, diagnosis, or treatment utility.

## Files

- `adhd_pupil_academic_diagnostic.txt` — complete corrected report
- `adhd_pupil_analysis_config.json` — analysis settings and environment
- `adhd_pupil_reproducibility_log.txt` — verified acquisition and decoding log
- repository `images/` — group, medication-pair, and exploratory-spectrum figures
