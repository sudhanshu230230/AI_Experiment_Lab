# ADR-003: Initial dataset and model selection metric

- **Status:** Accepted
- **Date:** 2026-10-10
- **Phase:** 1

## Context
The first implementation needs a small tabular classification dataset that makes
experiments fast, shows class imbalance, and can later demonstrate drift.

## Options considered
1. **Telco customer churn** (IBM sample dataset, available on Kaggle): about 7,000 rows,
   about 20 columns, mixed numeric and categorical, around 26% churn.
2. Credit card fraud: strong imbalance, but large and anonymized features limit
   explainability.
3. Employee attrition: small and clean, less realistic for drift demonstrations.

## Decision
Use Telco customer churn. Verify the licence and source on download, and record the file
hash and download date in `docs/phases/phase-01-*.md`.

Primary selection metric: **F1 on the churn class**, with ROC-AUC and PR-AUC reported.
Rationale: churn is imbalanced, so accuracy is misleading, and both missed churners and
false alarms have a cost. Revisit when a business cost assumption is written down; if
retention offers are cheap, recall may be weighted more heavily.

## Known data issues to handle in Phase 1
- `TotalCharges` is stored as text and contains blank values for new customers; it must be
  coerced to numeric and the missing values imputed.
- `customerID` is an identifier and must be excluded from features.
- `Churn` is a Yes/No string and must be encoded.

(These come from general knowledge of this dataset; confirm them during profiling.)

## Consequences
- Class imbalance is handled with stratified splits and class weights, and is documented.
- Drift demonstrations later will use a deliberately shifted copy of the data.

## Interview note
I chose churn because it is imbalanced and business-meaningful, so accuracy is a poor
metric and I can justify F1 and PR-AUC as selection metrics.
