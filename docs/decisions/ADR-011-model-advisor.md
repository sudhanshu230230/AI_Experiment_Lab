# ADR-011: Model advisor, a capped "upload a CSV, get the algorithm" page

- **Status:** Accepted
- **Date:** 2026-10-19
- **Phase:** Post-v1.0 extension (supersedes option 4 of ADR-010)

## Context
ADR-010 made the platform work for any tabular dataset, but only through Colab or the CLI. A visitor of the live site could not upload their own CSV and ask "which algorithm should I use, and with what settings?". ADR-010 rejected training inside the web app because the free tier has about 0.5 GB of memory, little CPU, no persistent disk, and uploads are untrusted.

## Options considered
1. **Keep Colab/CLI only.** Safe, but the visitor cannot try it.
2. **A separate app on another host.** More infrastructure and a second deployment to keep in sync.
3. **A capped advisor inside the existing API.** Reuses `automl.train_dataset`; every limit is explicit.

## Decision
Option 3. `automl/advisor.py` wraps `train_dataset` and adds what a person needs: a plain-language reason for the winner, the settings explained one by one, a comparison with always guessing the most common answer (or the mean), warnings (score too good to be true, barely better than guessing, small data, rare class, a sampled file), column importance, and a copy-paste snippet. `serving/lab.py` adds `/lab` (page), `/lab/analyze`, `/lab/run`, `/lab/jobs/{id}` and `/lab/jobs/{id}/model` (a zip with `model.skops`, `metadata.json`, `advice.json`).

Limits, all environment variables with safe defaults: 15 MB upload, 100 columns, 20,000 rows used for the search (a random sample, reported), about 60 s of tuning, 10 Optuna trials per algorithm, 420 s hard stop, one run at a time (others get a clear "busy" answer), uploads kept in memory for 30 minutes and never written to disk. `LAB_ENABLED=0` turns the feature off.

Same rules as everywhere else: split first, choose by cross-validation on the training rows, score the unseen rows once. Recommendations come from a small, time-limited search, so the page says to check them on your own data.

## Consequences
- The live demo can be tried with any tabular CSV. Not supported: time series, images, audio, video (the page says so).
- The server now needs optuna (`pip install ".[serve,lab]"`) and runs CPU-heavy work in a background thread; a request during a run is refused rather than queued.
- The 20,000-row sample can favour a smaller model than the full data would; the page notes this when it samples.
- The platform cannot detect target leakage; it only warns when the score is suspiciously high.
