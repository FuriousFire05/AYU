# CIA1 Submission Checklist

Items are ordered by dependency and priority. No citation style is assumed; use
the instructor's required style or template once confirmed.

## Priority 1 — Build and verify the literature evidence base

- [ ] Select 15 papers that directly support predictive maintenance, RUL
  prediction, C-MAPSS methodology, leakage-safe evaluation, or relevant modelling
  choices.
- [ ] Obtain and read the full text of every selected paper.
- [ ] Record verified bibliographic metadata for each paper: authors, title,
  venue, year, pages, DOI/URL, and any required access information.
- [ ] Create an evidence note for each paper covering dataset, RUL target
  definition, model, split protocol, metrics, findings, limitations, and relevance
  to AYU.
- [ ] Explicitly label capped, piecewise, and uncapped target definitions.
- [ ] Replace any paper that cannot be checked from a reliable original source.
- [ ] Write one analytical paragraph per paper only after its evidence note is
  complete.
- [ ] Populate the Table of Papers and cross-check it against all 15 paragraphs.

## Priority 2 — Draft the report argument

- [ ] Draft the 1,000–1,500 word Introduction with verified background citations.
- [ ] Synthesize at least five defensible research gaps from the completed
  literature review.
- [ ] Write three final problem statements that follow directly from those gaps.
- [ ] Write three measurable project objectives aligned one-to-one with the
  problem statements.
- [ ] Draft Methodology and Feasibility as four clear points: data/target,
  experimental split, features/models, and evaluation/reproducibility.
- [ ] Draft at least three bounded original contributions without claiming a new
  algorithm or state-of-the-art result.
- [ ] Draft the Initial Python Implementation / Results section from
  `EXPERIMENT_RESULTS.md`.
- [ ] Discuss the MAE/RMSE nuance: without-cycle has lower MAE, while with-cycle
  has lower RMSE and a much lower NASA score.
- [ ] Keep validation-cycle results separate from official test-endpoint results.
- [ ] Draft the Conclusion without introducing new evidence.
- [ ] Write the 400–500 word Abstract last.

## Priority 3 — Insert results, tables, and figures

- [ ] Add the dataset-statistics table.
- [ ] Add the validation model-comparison table with all four metrics and best
  boosting rounds.
- [ ] Add the official endpoint-results table.
- [ ] Insert all five figures according to `FIGURE_MANIFEST.md`.
- [ ] Write self-contained academic captions and assign final figure numbers.
- [ ] Mention every table and figure in the report body.
- [ ] Check that image text remains legible after export to the submission format.
- [ ] State that correlation values and plots are descriptive associations, not
  causal or model-importance claims.

## Priority 4 — Verify references and factual consistency

- [ ] Confirm the instructor's required citation style or report template.
- [ ] Match every in-text citation to exactly one reference-list entry and ensure
  every listed reference is cited.
- [ ] Verify all author names, years, titles, venues, page ranges, and identifiers
  against original sources.
- [ ] Recheck every reported number against `EXPERIMENT_RESULTS.md`.
- [ ] Confirm the report says the target is exact uncapped linear RUL.
- [ ] Remove or qualify comparisons with papers using capped/piecewise targets.
- [ ] Confirm the report says `cycle` is observable age and `unit_id` is excluded.
- [ ] Confirm that sensor selection and early stopping are described as
  development-training/validation operations, not test-set operations.
- [ ] Explain why validation and official NASA score totals are not directly
  comparable.

## Priority 5 — Final formatting and quality control

- [ ] Apply the instructor's required heading hierarchy, margins, typography,
  numbering, and front matter.
- [ ] Check section word-count requirements, especially Abstract and Introduction.
- [ ] Check table and figure pagination, captions, cross-references, and page
  breaks.
- [ ] Define every abbreviation at first use and use terminology consistently.
- [ ] Run spelling and grammar checks, then proofread the exported document rather
  than relying only on the editor view.
- [ ] Confirm that equations and symbols render correctly in the final file.
- [ ] Confirm that no code path, temporary note, placeholder, or unresolved TODO
  remains in the submitted document.

## Priority 6 — Academic-integrity and Turnitin review

- [ ] Review each literature paragraph against its source to ensure it is an
  original synthesis rather than sentence-level substitution.
- [ ] Add citations wherever an idea, method, result, dataset fact, or quotation
  comes from another work.
- [ ] Put any necessary exact wording in quotation marks and keep quotations
  minimal.
- [ ] Do not mechanically paraphrase to evade similarity detection; improve the
  writing by explaining evidence in the project's own analytical structure.
- [ ] Run the institution-approved similarity/Turnitin check if access is
  provided.
- [ ] Inspect every flagged passage, correct missing attribution, and revise overly
  source-like wording while preserving technical accuracy.
- [ ] Perform a final team authorship review so every member can explain the
  methods, results, and cited claims.

## Submission gate

- [ ] All required CIA1 sections are present in the instructor's requested order.
- [ ] The literature review contains exactly 15 verified paper paragraphs and a
  matching Table of Papers.
- [ ] At least five gaps, three problem statements, three objectives, four
  methodology/feasibility points, and at least three contributions are explicit.
- [ ] The final PDF/document opens correctly and all figures and tables render.
- [ ] File naming and submission procedure match the instructor's instructions.
- [ ] A final backup of the submitted version and source report is retained.
