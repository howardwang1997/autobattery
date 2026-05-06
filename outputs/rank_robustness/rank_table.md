# Fisher rank robustness study

Dataset: `data/fullfield/fullfield_lfp_degradation.h5` (1200 sims, 100 time pts)
Bootstrap: 200 resamples, ridge α=1e-03

## η-rank table (median [5th–95th] from bootstrap)

| parameterisation | condition | η=1e-02 | η=1e-03 | η=1e-06 | η=1e-09 | η=1e-12 |
|---|---|---|---|---|---|---|
| raw | all_data_singleC_pooled | **2** [1–2] | **3** [2–3] | **4** [4–5] | **5** [5–5] | **5** [5–5] |
| raw | single_C=0.5 | **1** [1–1] | **2** [2–2] | **3** [3–3] | **5** [5–5] | **5** [5–5] |
| raw | single_C=1 | **2** [2–2] | **3** [3–3] | **5** [5–5] | **5** [5–5] | **5** [5–5] |
| raw | single_C=2 | **2** [2–2] | **2** [2–2] | **4** [4–4] | **5** [5–5] | **5** [5–5] |
| raw | joint_multi_rate(3_rates) | **2** [2–2] | **3** [3–3] | **5** [5–5] | **5** [5–5] | **5** [5–5] |
| log | all_data_singleC_pooled | **3** [2–3] | **3** [3–4] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log | single_C=0.5 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log | single_C=1 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log | single_C=2 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log | joint_multi_rate(3_rates) | **3** [3–3] | **5** [5–5] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log_standardised | all_data_singleC_pooled | **3** [2–3] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log_standardised | single_C=0.5 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log_standardised | single_C=1 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| log_standardised | single_C=2 | **2** [2–2] | **2** [2–2] | **6** [6–6] | **7** [7–7] | **7** [7–7] |
| log_standardised | joint_multi_rate(3_rates) | **3** [3–3] | **4** [4–4] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| pca_whitened | all_data_singleC_pooled | **3** [2–3] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| pca_whitened | single_C=0.5 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| pca_whitened | single_C=1 | **2** [2–2] | **3** [3–3] | **7** [7–7] | **7** [7–7] | **7** [7–7] |
| pca_whitened | single_C=2 | **2** [2–2] | **2** [2–2] | **6** [6–6] | **7** [7–7] | **7** [7–7] |
| pca_whitened | joint_multi_rate(3_rates) | **3** [3–3] | **4** [4–4] | **7** [7–7] | **7** [7–7] | **7** [7–7] |

## Per-parameterisation eigenvalue spectra

See `spectrum.png`.

## Reading the table for the paper

- If rank stays at 3 across all four parameterisations and across all η levels above 1e-6, the universal-rank-3 claim is robust.
- If rank rises in `joint_multi_rate(...)` rows above the per-rate rank, you have a quantitative argument for adding new operating conditions (multi-C-rate, multi-T) — this is the physical justification for GITT/HPPC retrofits.
- η=1e-12 should generally show full rank (= n_params); if it doesn't, your dataset has degenerate parameter directions.