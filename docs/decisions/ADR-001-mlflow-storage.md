# ADR-001: MLflow tracking and artifact storage

- **Status:** Accepted (option 1, validated by a spike on 2026-10-12)
- **Date:** 2026-10-10
- **Phase:** 2

## Context
Training runs in Google Colab, whose local disk is wiped when the runtime ends. MLflow
runs, parameters, metrics and model artifacts must survive between sessions and be
reachable later from the API and dashboard. No paid infrastructure is available.

## Findings from the MLflow documentation (checked 2026-10-10)
Source: [MLflow backend store documentation](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/).
- The **local file system backend is legacy**: marked "to be deprecated soon", in maintenance mode,
  with no further updates. This rules out the `file:` tracking URI that the Colab setup cell
  currently sets (it is unused until Phase 2 code lands, and will be changed after this spike).
- A **relational database is the recommended backend and the default**. SQLite is the default
  (`sqlite:///mlflow.db`); PostgreSQL, MySQL and MSSQL are also supported.
- SQLite allows **one concurrent writer** (stated in the docs for trace processing).
- The documentation says **nothing about running SQLite on a network filesystem** such as a
  Drive mount. SQLite file locking is known to be unreliable on some network and FUSE
  filesystems, so this must be tested, not assumed.

## Options considered
1. **SQLite database and artifacts both on the Drive mount.** Simplest, persistent, free.
   Risk: locking or corruption on the Drive filesystem; slower writes.
2. **SQLite on Colab local disk, artifacts on Drive, database copied to Drive at the end of each session.**
   Avoids locking on Drive. Risk: runs since the last copy are lost if the runtime dies; copying
   needs a discipline (or a helper function).
3. **Hosted MLflow (for example DagsHub).** Persistent, shareable, always-on UI.
   Costs: third-party account, free-tier limits, an extra dependency.
4. **Self-hosted server on a free cloud tier with PostgreSQL.** Closest to production. Costs: setup
   effort; free tiers may sleep or reset storage.
5. **Legacy `file:` store.** Rejected: deprecated (see findings).

## Spike (how option 1 is tested)
`scripts/mlflow_storage_spike.py`, run via `notebooks/spike_mlflow_storage.ipynb`.
It writes 30 runs (3 params, 3 metrics, 1 artifact each), reads them back from a fresh connection,
runs two writers concurrently, and verifies persistence after deleting the Colab runtime.
It was tested on local disk before the Colab run (30 runs in under a second, no errors, concurrent
writers fine), so any failure on Drive is attributable to Drive.

**Option 1 is accepted if all of these hold on Drive:**
1. `write`: all 30 runs written and read back with full metadata and artifacts, zero errors.
2. `verify` after a runtime restart: everything still present and readable.
3. Time per run is under 2 seconds (otherwise logging 50 Optuna trials becomes tedious).
4. `concurrent`: either passes, or fails in a way we can live with (we run one training session
   at a time, and will document "one writer" as a rule).

If 1 or 2 fails, use option 2 (and re-run the spike on that layout) or option 3.

## Spike results
Run in Colab on 2026-10-12, 30 runs per batch (3 params, 3 metrics, 1 artifact each).

| Check | Local disk (Colab VM) | Google Drive |
|---|---|---|
| write: runs written / failed | 30 / 0 | 30 / 0 (two batches: 60 / 0) |
| time per run | 0.09 s | 0.26 s |
| concurrent writers (2 processes x 15 runs, same VM) | not run | 30 / 0 written; 0 "database is locked" messages; 0.42 to 0.47 s per run under contention |
| persistence after deleting the runtime | n/a | 120 runs found, all with full params, metrics and artifacts |

Against the acceptance criteria: (1) pass, (2) pass, (3) pass (0.26 s is far under 2 s), (4) pass for two
processes on one VM.

**Note on the FAIL lines in the first run:** the first version of the script labelled every batch with the
same name and checked the count exactly, so running the cell twice on a non-empty store reported 60 runs
where 30 were expected. No data was wrong: the final count was 120 = 2 x 30 + 2 x 15 + 2 x 15, with nothing
lost or duplicated. The script now gives each run a unique label and reports exact counts on repeat runs.

**What the spike did not test:**
- Two *separate* Colab sessions (different VMs) writing at the same time. The one-writer rule below
  covers this.
- Large artifacts (real trained pipelines, tens of MB). Check logging time when Phase 2 logs actual models.
- Long-term growth of the database.

## Decision
Use a **SQLite database and file artifacts, both on Google Drive** (option 1), with tracking URI
`sqlite:////content/drive/MyDrive/ai-experiment-lab/mlflow/mlflow.db` and artifacts under
`.../mlflow/artifacts`. Options 2 to 4 stay as fallbacks if a later phase needs a shared or always-on
server (for example when the API needs to read the registry from outside Colab, in Phase 5).

## Consequences (apply whichever option is chosen)
- The tracking URI is configuration (`configs/default.yaml`), never hardcoded in code.
- The Colab setup cell now sets the SQLite URI on Drive via `MLFLOW_TRACKING_URI`; the `file:` URI has been removed.
- **Rule: one training session writes to the store at a time** (two Colab sessions were not tested).
- Keep a backup copy of `mlflow.db` (for example copy it to `mlflow.db.bak` at the end of a session). Add a helper in Phase 2.
- When the API (Phase 5) must read the registry from outside Colab, revisit hosting (options 3 or 4).
- Moving to a hosted or self-hosted server later is a config change plus a data migration.

## Interview note
MLflow's file store is being deprecated, so I used a SQLite database. Because Colab is
ephemeral and SQLite behaves differently on network drives, I tested the storage layout
(including concurrent writers and a runtime restart) before building on it.
