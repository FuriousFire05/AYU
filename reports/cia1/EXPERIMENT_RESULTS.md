# CIA1 Validated Experiment Results

## Experimental scope

CIA1 is a controlled Remaining Useful Life (RUL) regression experiment on NASA
C-MAPSS **FD001**. It compares a naive mean-RUL baseline with two otherwise
matched XGBoost models: one includes the observable `cycle` value and one excludes
it. The official test labels are reserved for final endpoint evaluation.

## Dataset statistics

| Item | Count |
|---|---:|
| Training rows | 20,631 |
| Training engines | 100 |
| Test rows | 13,096 |
| Test engines | 100 |
| Official test RUL labels | 100 |

The training trajectories extend to failure. The official FD001 test trajectories
stop before failure and provide one RUL label for the last observed row of each
engine.

## Target construction

The training target is the exact, uncapped linear RUL:

```text
RUL = final_cycle_for_engine - current_cycle
```

Consequently, the failure row has RUL 0. No capped or piecewise-constant target is
used.

## Development split and leakage controls

| Partition | Engines | Evaluation rows/cycles |
|---|---:|---:|
| Development training | 80 | Training trajectories for complete engines |
| Validation | 20 | 4,070 |

- Complete engines are assigned to one partition only; individual rows are never
  randomly split.
- The training and validation engine-ID sets have zero overlap.
- `unit_id` is retained only as an identifier and is never a model feature.
- `cycle` is observable engine age at inference time. It is intentionally included
  in one model and excluded from the other; it is not described as leakage.
- Learned sensor selection is derived from the development-training engines only.
- Official test labels are not used for feature selection, early stopping, or model
  choice.

## Sensor analysis

| Item | Result |
|---|---|
| Total sensor channels | 21 |
| Informative sensor channels | 15 |
| Removed near-constant channels | `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` |

### Strongest simple absolute RUL correlations

| Sensor | Absolute Pearson correlation with RUL |
|---|---:|
| `sensor_11` | 0.696228 |
| `sensor_4` | 0.678948 |
| `sensor_12` | 0.671983 |
| `sensor_7` | 0.657223 |
| `sensor_15` | 0.642667 |

These values are descriptive, univariate associations from the full FD001
training corpus. They are not causal effects, multivariate feature importances, or
standalone evidence of predictive utility. The model's near-zero-variance filter
is fitted only on the 80 development-training engines. The current correlation
figure is also generated from that development partition, so its bar heights can
differ slightly from the full-corpus values above.

## Feature engineering

For each of the 15 informative sensors, the models retain the current raw sensor
value and add trailing rolling statistics over windows of 5, 10, and 20 cycles:

- rolling mean;
- rolling standard deviation.

The rolling calculations are grouped by `unit_id`, use the current and previous
observations only, never cross an engine boundary, and keep early-cycle rows with
`min_periods=1`. Standard-deviation values from a one-observation window are set to
zero.

Both XGBoost variants use:

- the three operating settings;
- 15 raw informative sensors;
- 90 rolling features: 15 sensors × 3 windows × 2 statistics.

| Variant | Feature count | Includes `cycle` | Includes `unit_id` |
|---|---:|---:|---:|
| XGBoost with cycle | 109 | Yes | No |
| XGBoost without cycle | 108 | No | No |

## Models and evaluation

### Naive baseline

The baseline predicts the development-training set's mean RUL for every
validation observation.

### XGBoost

The two XGBoost regressors use the same deterministic configuration apart from
the presence or absence of `cycle`: squared-error regression, learning rate 0.03,
maximum depth 6, subsample 0.8, column subsample 0.8, L2 regularization 1.0,
random state 42, and histogram tree construction. A 3,000-tree ceiling is paired
with validation-based early stopping of 50 rounds using the XGBoost 3.4 sklearn
API.

### Metrics

Negative predicted RUL values are clipped to zero before evaluation. Reported
metrics are MAE, RMSE, R², and the NASA asymmetric prognostic score. For prediction
error

```text
d = predicted_RUL - true_RUL
```

the per-observation NASA penalty is:

```text
exp(-d / 13) - 1    when d < 0
exp( d / 10) - 1    when d >= 0
```

Thus, overestimating remaining life receives the steeper penalty.

## Validation results

The following results cover all **4,070 cycle-level observations** from the 20
validation engines.

| Model | MAE | RMSE | R² | NASA score | Best boosting rounds |
|---|---:|---:|---:|---:|---:|
| Naive mean-RUL | 55.3633 | 65.7154 | -0.0019 | 19,351,068.7551 | Not applicable |
| XGBoost with cycle | 24.6777 | **32.0213** | **0.7621** | **411,193.2901** | 84 |
| XGBoost without cycle | **22.9424** | 32.1111 | 0.7608 | 1,428,466.0422 | 264 |

The **with-cycle XGBoost model** was selected because validation RMSE was the
primary selection criterion and it was slightly lower: 32.0213 versus 32.1111.
Its NASA score was also substantially lower. The without-cycle model did obtain
the lower MAE, which should remain visible in the discussion rather than being
discarded.

After selection, the with-cycle feature variant was refitted on all 100 training
engines for the selected 84 boosting rounds. The official test labels were then
used once for final endpoint evaluation.

## Official FD001 test endpoint results

| Selected model | Endpoints | MAE | RMSE | R² | NASA score |
|---|---:|---:|---:|---:|---:|
| XGBoost with cycle | 100 | 19.0819 | 25.4518 | 0.6249 | 12,111.8971 |

Each test engine contributes exactly one prediction: the prediction from its last
provided observation.

## Interpretation and methodological caveats

1. **NASA score scale:** the validation NASA score sums 4,070 cycle-level
   penalties, whereas the official score sums 100 endpoint penalties. Their
   magnitudes must not be directly compared.
2. **Target comparability:** these results use exact uncapped linear RUL. They are
   not directly comparable with studies using capped or piecewise RUL targets.
3. **Cycle interpretation:** `cycle` is observable engine age, not future
   information. The ablation measures dependence on age versus degradation
   signals.
4. **Correlation interpretation:** Pearson correlations are simple absolute
   linear associations. They do not establish causality or account for feature
   interactions.
5. **Subset scope:** conclusions are specific to FD001 and should not be
   generalized to other C-MAPSS operating-condition/fault-mode subsets without
   further experiments.
6. **Test-set discipline:** the official endpoint result is a final evaluation,
   not a source for repeated tuning or variant selection.
