# CIA1 Figure Manifest

This manifest covers every current PNG figure under `outputs/figures`. Proposed
figure numbers are provisional and should be updated after the final report layout
is assembled.

## Figure 1 — FD001 training engine lifetime distribution

**File:** [`outputs/figures/fd001_lifetime_distribution.png`](../../outputs/figures/fd001_lifetime_distribution.png)

- **What it shows:** A 20-bin histogram of the final observed cycle for each of
  the 100 complete FD001 training engines.
- **Suggested report location:** Introduction or the dataset-description part of
  Methodology.
- **Academic takeaway:** FD001 engines have heterogeneous run-to-failure lengths,
  supporting an engine-aware experimental design rather than treating rows as
  independent and identically distributed samples.
- **Interpretation caution:** Histogram shape depends on bin selection. The chart
  describes training-engine lifetimes, not the censored length of official test
  trajectories and not a real-world fleet survival distribution.

## Figure 2 — FD001 RUL distribution

**File:** [`outputs/figures/fd001_rul_distribution.png`](../../outputs/figures/fd001_rul_distribution.png)

- **What it shows:** A 30-bin histogram of every row-level training target created
  by `final_cycle - current_cycle`.
- **Suggested report location:** Methodology, immediately after target
  construction, or the initial exploratory-results subsection.
- **Academic takeaway:** The exact linear target covers a wide RUL range and is
  not capped at an early-life plateau.
- **Interpretation caution:** This is a row-level distribution, so longer-lived
  engines contribute more observations. It should not be described as an
  engine-level distribution, and it is not comparable with plots based on capped
  or piecewise targets.

## Figure 3 — Sensor association with RUL

**File:** [`outputs/figures/fd001_sensor_rul_correlations.png`](../../outputs/figures/fd001_sensor_rul_correlations.png)

- **What it shows:** Absolute Pearson correlations between RUL and the 15
  non-constant sensors, ordered from strongest to weakest. The current runner
  generates this figure from the 80 development-training engines.
- **Suggested report location:** Initial Python Implementation / Results, in the
  exploratory sensor-analysis subsection.
- **Academic takeaway:** Several sensors, especially `sensor_11`, `sensor_4`,
  `sensor_12`, `sensor_7`, and `sensor_15`, show substantial simple association
  with degradation progression and motivate retaining sensor-based features.
- **Interpretation caution:** Absolute correlation hides direction and measures
  only univariate linear association; it is neither causal evidence nor model
  feature importance. Because this figure uses the development partition, its
  values differ slightly from the validated full-corpus descriptive correlations
  recorded in `EXPERIMENT_RESULTS.md`.

## Figure 4 — Example sensor 11 trajectories

**File:** [`outputs/figures/fd001_sensor_11_trajectories.png`](../../outputs/figures/fd001_sensor_11_trajectories.png)

- **What it shows:** Raw `sensor_11` values against cycle for training engines 1,
  2, and 3.
- **Suggested report location:** Initial Python Implementation / Results, after
  the correlation figure and before causal rolling-feature construction.
- **Academic takeaway:** The signal exhibits a noisy but visible late-life trend
  across engines of different lengths, motivating trailing rolling statistics
  that summarize local degradation without using future observations.
- **Interpretation caution:** Three hand-fixed example engines are illustrative,
  not a representative statistical sample. Differences in trajectory length and
  noise must not be interpreted as proof that one sensor alone determines RUL.

## Figure 5 — Official test predicted versus actual RUL

**File:** [`outputs/figures/fd001_predicted_vs_actual.png`](../../outputs/figures/fd001_predicted_vs_actual.png)

- **What it shows:** The selected with-cycle XGBoost model's prediction for the
  last observed row of each of the 100 official FD001 test engines, plotted
  against its official RUL label. The dashed line is ideal equality (`y = x`).
- **Suggested report location:** Initial Python Implementation / Results, in the
  final official-test evaluation subsection.
- **Academic takeaway:** Predictions broadly follow actual endpoint RUL, while
  visible dispersion supports reporting error metrics and asymmetric risk rather
  than relying on visual fit alone.
- **Interpretation caution:** Points above the equality line are RUL
  overestimates, which receive the steeper NASA penalty. This is a final test
  visualization for the selected model, not a hyperparameter-selection plot, and
  it must not be compared directly with studies using capped/piecewise targets or
  different test protocols.

## Insertion checklist for every figure

- Use a numbered caption that states the subset, population, and unit of analysis.
- Cite and discuss the figure in the body text before or immediately after it.
- Keep axis labels and resolution legible at the final report size.
- Preserve the original aspect ratio.
- Explain the relevant interpretation caution in the surrounding prose rather
  than presenting the figure as self-explanatory.
