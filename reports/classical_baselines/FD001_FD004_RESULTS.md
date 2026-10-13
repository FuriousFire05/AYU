# FD001–FD004 Classical Baseline Results

## Reproducibility checkpoint

The completed CIA1 report and its original FD001 implementation were frozen in
local commit `ae82741` and annotated tag `cia1-fd001-baseline` before this
milestone began. No commit or tag was pushed. Running the untouched CIA1 script
reproduced its recorded validation and official endpoint metrics exactly.

The original baseline was reproducible but limited to FD001: subset names and
plot titles were hard-coded, split/configuration manifests were not persisted,
outputs were divided only into shared figure/prediction folders, and the loader
opened official labels before model selection. The generalized runner addresses
those blockers and retains the CIA1 model defaults.

## Protocol

- Exact uncapped target: `final_cycle - current_cycle`.
- Seed 42 engine-disjoint 80/20 development split.
- Sensor variance filtering fitted on development-training engines only.
- Per-engine causal rolling mean and standard deviation over windows 5, 10, and
  20, with raw sensors retained and `unit_id` excluded.
- Validation metrics are calculated over all cycles from held-out engines.
- The with-cycle and without-cycle XGBoost variants are selected only by lowest
  validation RMSE. The selected variant is refitted on all training engines for
  its validation-selected boosting length.
- Official labels are opened only after selection; official metrics cover one
  last-observation endpoint per test engine. The fixed naive comparator is also
  reported at endpoints but is not eligible for XGBoost variant selection.

## Dataset and feature handling

| Subset | Train rows / engines | Test rows / engines | Dev / validation engines | Selected sensors | Features with / without cycle |
|---|---:|---:|---:|---:|---:|
| FD001 | 20,631 / 100 | 13,096 / 100 | 80 / 20 | 15 | 109 / 108 |
| FD002 | 53,759 / 260 | 33,991 / 259 | 208 / 52 | 21 | 151 / 150 |
| FD003 | 24,720 / 100 | 16,596 / 100 | 80 / 20 | 16 | 116 / 115 |
| FD004 | 61,249 / 249 | 41,214 / 248 | 199 / 50 | 21 | 151 / 150 |

FD002 and FD004 retain all raw sensors under the global variance filter because
their multiple operating regimes create variation in channels that are
near-constant in single-regime subsets. This milestone intentionally does not
introduce operating-condition-aware normalization.

## Engine-disjoint cycle-level validation

| Subset | Model | MAE | RMSE | R² | NASA score | Rounds |
|---|---|---:|---:|---:|---:|---:|
| FD001 | Naive mean | 55.3633 | 65.7154 | -0.0019 | 19,351,068.7551 | — |
| FD001 | XGBoost with cycle | 24.6777 | **32.0213** | **0.7621** | 411,193.2901 | 84 |
| FD001 | XGBoost without cycle | **22.9424** | 32.1111 | 0.7608 | 1,428,466.0422 | 264 |
| FD002 | Naive mean | 54.8312 | 66.0769 | -0.0086 | 898,066,355.8542 | — |
| FD002 | XGBoost with cycle | **26.3119** | **34.7651** | **0.7208** | **3,380,414.3952** | 298 |
| FD002 | XGBoost without cycle | 29.3798 | 38.8091 | 0.6521 | 15,075,184.6455 | 880 |
| FD003 | Naive mean | 73.6788 | 93.2803 | -0.0121 | 579,193,200,673.5009 | — |
| FD003 | XGBoost with cycle | **35.2677** | **52.4588** | **0.6799** | **600,608,836.2805** | 63 |
| FD003 | XGBoost without cycle | 36.9016 | 54.2190 | 0.6581 | 6,131,896,018.4974 | 76 |
| FD004 | Naive mean | 72.4847 | 88.7497 | -0.0012 | 783,024,626,430.6530 | — |
| FD004 | XGBoost with cycle | **42.1621** | **55.9969** | **0.6014** | **187,871,079.6528** | 123 |
| FD004 | XGBoost without cycle | 42.7667 | 59.3966 | 0.5516 | 4,224,987,224.2750 | 505 |

The large cycle-level NASA scores are sums over 4,070 to 11,955 validation
observations and are particularly sensitive to unsafe large positive errors.
They must not be compared directly with endpoint scores.

## Locked official endpoint evaluation

| Subset | Model | Endpoints | MAE | RMSE | R² | NASA score |
|---|---|---:|---:|---:|---:|---:|
| FD001 | Naive mean | 100 | 40.8793 | 52.6249 | -0.6037 | 269,287.4972 |
| FD001 | XGBoost with cycle | 100 | **19.0819** | **25.4518** | **0.6249** | **12,111.8971** |
| FD002 | Naive mean | 259 | 50.8632 | 60.1623 | -0.2515 | 964,911.2885 |
| FD002 | XGBoost with cycle | 259 | **20.9480** | **28.3550** | **0.7220** | **24,764.2330** |
| FD003 | Naive mean | 100 | 63.2709 | 75.1801 | -2.2984 | 5,143,282.7792 |
| FD003 | XGBoost with cycle | 100 | **28.4586** | **35.1873** | **0.2775** | **54,258.8735** |
| FD004 | Naive mean | 248 | 60.0352 | 71.8279 | -0.7355 | 9,250,209.0245 |
| FD004 | XGBoost with cycle | 248 | **29.9654** | **39.1600** | **0.4842** | **796,304.5556** |

## CIA1 comparison and limitations

FD001 is unchanged: the selected with-cycle model still uses 84 rounds;
validation RMSE/R² remain 32.0213/0.7621, and official endpoint MAE/RMSE/R²/NASA
remain 19.0819/25.4518/0.6249/12111.8971. This exact match is code-level evidence
that generalization did not alter CIA1 behavior.

The official raw data is intentionally Git-ignored and must be obtained
separately. Results use one seeded holdout rather than repeated group
cross-validation. Exact uncapped labels make early-life errors and NASA scores
large, especially for dual-fault subsets; results are not comparable to capped
RUL studies without adjustment. These baselines do not implement condition-aware
preprocessing, calibrated uncertainty, or deployment components.
