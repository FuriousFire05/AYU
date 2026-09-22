# CIA1 Report Plan

## Purpose and evidence boundary

This plan maps the required CIA1 structure to the material already available in
AYU. It is a writing plan, not a completed report. Literature claims, paper
details, and references must only be added after the original sources have been
read and verified. The current experiment uses NASA C-MAPSS FD001 and an exact,
uncapped linear RUL target.

Page estimates below assume a conventional academic layout and are only planning
guides. Final pagination will depend on the instructor's template, font, spacing,
tables, and figure placement.

## Section map

| Required section | What belongs in the section | Approximate budget | Existing AYU evidence to use | Still to write or verify |
|---|---|---:|---|---|
| Abstract | The maintenance problem, FD001 scope, engine-disjoint evaluation, causal rolling features, naive baseline, with-cycle/without-cycle XGBoost ablation, selected model, main validation and official endpoint results, and a restrained conclusion. | **400–500 words**; about 1 page | Dataset counts; 80/20 engine split; selected with-cycle model; validation RMSE 32.0213; official endpoint RMSE 25.4518; other metrics from `EXPERIMENT_RESULTS.md`. | Write last, after the full report is stable. Ensure every numerical claim matches the results record and avoid literature-comparison claims based on incompatible targets. |
| Introduction | Predictive maintenance motivation; why RUL supports maintenance decisions; turbofan degradation context; the C-MAPSS setting; supervised regression framing; why temporal causality and engine-disjoint validation matter; the controlled CIA1 scope; and a short roadmap of the report. | **1,000–1,500 words**; about 2–3 pages | FD001 dataset statistics; `fd001_lifetime_distribution.png`; exact RUL definition. | Add verified background citations; explain practical motivation without overstating deployment readiness; define the project boundary and report organization. |
| Detailed Literature Review | Exactly 15 paper-specific paragraphs. Each paragraph should identify the verified paper, research question, data, target definition, method, evaluation protocol, key result, limitation, and relevance to AYU. End with a short synthesis connecting recurring themes. | About **180–250 words per paper**, 2,700–3,750 words total; roughly 6–8 pages plus synthesis | AYU's design provides comparison categories: target definition, split unit, causal features, model family, metrics, and test protocol. | Select and read 15 credible papers; verify every bibliographic fact and metric from the source; write one analytical paragraph per paper. **Do not draft from titles or abstracts alone.** |
| Table of Papers | A compact comparison of all 15 reviewed papers. Suggested columns: reference key, year, dataset, target definition/cap, model, split/evaluation unit, metrics, main finding, limitation, and relevance to AYU. | About 1–2 landscape pages; table text is not a substitute for the 15 paragraphs | A blank evidence framework can be populated once papers are verified. | Populate only from checked sources; make capped, piecewise, and uncapped targets explicit; confirm that table claims agree with the prose and references. |
| Problem Identification | At least five evidence-backed gaps emerging from the literature and project context. Candidate areas to assess include inconsistent target definitions, row-wise leakage risks, unclear causal feature construction, limited age-versus-sensor ablations, asymmetric-risk reporting, and weak separation of development validation from final test evaluation. | About **600–900 words**; 1–2 pages | AYU already addresses engine-disjoint splitting, causal windows, cycle ablation, NASA score reporting, and one-time official test evaluation. | Confirm each claimed gap across the reviewed literature; distinguish a demonstrated gap from a design preference; cite the relevant papers. |
| Final Problem Statements | Three concise, researchable statements derived from the verified gaps: (1) leakage-safe FD001 RUL modelling, (2) measuring dependence on observable engine age versus degradation signals, and (3) evaluating predictive accuracy together with asymmetric prognostic risk. | About **250–450 words**; under 1 page | The implemented split, ablation, and metric suite directly support these themes. | Turn the themes into three precise statements with scope, inputs, outputs, and evaluation boundaries; ensure they do not promise later CIA2/CIA3 work. |
| Project Objectives | Three measurable objectives aligned one-to-one with the final problem statements. They should cover a correct reproducible FD001 pipeline, a controlled with-cycle/without-cycle XGBoost comparison, and evaluation on engine-disjoint validation plus final official endpoints. | About **150–300 words**; under 1 page | All three objectives have implementation evidence and quantitative results. | Phrase each objective with an observable completion criterion; avoid adding deployment or deep-learning objectives not present in CIA1. |
| Methodology and Feasibility | Four required points: (1) data and exact RUL construction, (2) engine-disjoint experimental design and training-only sensor selection, (3) causal features and controlled models, and (4) evaluation, compute/software feasibility, and reproducibility. | About **700–1,000 words**; 1.5–2 pages | Python 3.12 and existing `uv` project; pandas/NumPy/scikit-learn/XGBoost; 20,631 training rows; causal 5/10/20-cycle windows; early stopping; validation/test protocol. | Convert implementation details into academic prose or a four-part subsection; add a simple workflow diagram only if later created and verified; report hardware only after it is known. |
| Original Contributions | At least three bounded contributions: a leakage-conscious engine-level protocol, causal multi-window statistical features with raw sensors retained, and a cycle-dependence ablation interpreted with both symmetric errors and the NASA asymmetric score. Reproducible result artifacts may be a further contribution. | About **300–500 words**; around 1 page | Feature construction, ablation, typed metrics, prediction CSVs, and tests already exist. | Explain what is original within this semester-project context without claiming a novel algorithm or state-of-the-art performance. |
| Initial Python Implementation / Results | Dataset sanity checks; RUL construction; split; sensor filtering and correlations; causal features; baseline and XGBoost configuration; validation table; ablation interpretation; model selection; official endpoint table; and limitations. | About **1,200–1,800 words**; 3–5 pages including figures/tables | All five current figures; validation and official metrics; 84/264 best rounds; saved prediction CSVs; seven passing tests. | Write figure captions and table numbering; explain why RMSE selected the model while noting MAE and NASA score; clearly separate cycle-level validation from endpoint-level official testing. |
| Conclusion | Concise recap of the problem, controlled methodology, strongest findings, limitations, and the next evidence-based step. No new results or literature claims. | About **300–500 words**; around 1 page | With-cycle selection, official endpoint metrics, and methodological caveats. | Draft after all earlier sections; state that conclusions are specific to FD001 and the exact uncapped linear target. |
| References | Complete entries for all cited sources, including the verified NASA C-MAPSS source and the 15 reviewed papers. Every in-text citation must have one entry and vice versa. | Length determined by sources; likely 2–4 pages | No bibliography is fabricated in this plan. | Confirm the instructor's required citation style or report template; verify authors, title, venue, year, pages, DOI/URL, and access details where applicable. |

## Results placement guide

| Evidence | Primary report location | Secondary use |
|---|---|---|
| Dataset and engine counts | Methodology; Initial Implementation / Results | Abstract and Introduction |
| Engine lifetime distribution | Introduction or dataset subsection | Methodology |
| Exact linear RUL distribution | Methodology / target construction | Limitations discussion |
| Sensor/RUL correlation chart | Exploratory analysis subsection | Motivation for sensor analysis |
| Sensor 11 trajectories | Exploratory analysis / causal feature motivation | Introduction to degradation signals |
| Validation comparison table | Model evaluation subsection | Abstract and Conclusion |
| Official predicted-versus-actual plot | Final evaluation subsection | Conclusion |
| Official endpoint metric table | Final evaluation subsection | Abstract and Conclusion |

## Consistency rules for drafting

- Call `cycle` observable engine age, not data leakage.
- State that `unit_id` is an identifier and is excluded from model inputs.
- State that sensor selection is derived from development-training engines only.
- Describe rolling statistics as trailing, per-engine, and causal.
- Keep cycle-level validation metrics separate from endpoint-level official test
  metrics.
- Never compare NASA score totals across sample sets of different sizes without
  explaining that the score is a sum.
- Do not compare the reported metrics directly with capped or piecewise-RUL papers.
- Do not claim deployment readiness, state-of-the-art performance, or causal sensor
  importance from this CIA1 experiment.
