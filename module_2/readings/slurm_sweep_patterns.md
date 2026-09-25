# SLURM Sweep Patterns

*Module 2. The reference for Lecture 11 (Fri Sep 25) and Lab 2 (due Mon Oct 5). Runnable companions: `module_2/examples/optuna_sweep.py`, `optuna_sweep.sbatch` and `analyze_study.py`, also in `/project/me5475/examples/` on ARCC.*

Two patterns for parallel hyperparameter search on a SLURM cluster. The choice matters; this is the reference for picking the right one.

---

## Pattern A — SLURM array, one trial per task, nothing shared

Each array task runs ONE hyperparameter configuration and shares nothing with the other tasks. Suitable for grid search or random search.

**The script:**

```bash
#!/bin/bash
#SBATCH --array=0-24           # 25 trials: 5 learning rates x 5 widths
#SBATCH --time=00:10:00        # per trial

TRIAL_ID=${SLURM_ARRAY_TASK_ID}

# Map TRIAL_ID (0..24) -> hyperparameters. For grid search:
LRS=(1e-5 1e-4 1e-3 1e-2 1e-1)
WIDTHS=(16 32 64 128 256)
LR_IDX=$((TRIAL_ID / 5))       # 0..4
WIDTH_IDX=$((TRIAL_ID % 5))    # 0..4
LR=${LRS[$LR_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

srun python train.py --lr $LR --width $WIDTH --out trial_${TRIAL_ID}/
```

The array size must equal the grid size: task 0 gets `lr=1e-5, width=16`, task 24 gets `lr=1e-1, width=256`. A task ID past 24 would index past the end of `LRS` and pass an empty `--lr`. (Bash arrays start at 0; zsh arrays do not, so keep the `#!/bin/bash` line.)

**Pros:**
- Simple: the tasks share no storage, so there is no shared database to lock or corrupt.
- Each trial is fully isolated; one task failing doesn't block others.
- Easy to inspect: one stdout/stderr per trial.

**Cons:**
- No shared observations: as written here, no task sees another task's results, so the search cannot adapt — no Bayesian sampling that learns from earlier trials, and no pruning of a trial by comparison with its peers.
- A poor region of the grid gets the same compute as a good one.

(Those limits come from sharing nothing, not from arrays as such. A task can still early-stop its own run on its own validation loss, and a one-trial-per-task array can instead join a shared Optuna study — that is Pattern B with one trial per worker.)

**When to use:**
- Small search space (< 50 trials).
- Each trial is cheap (< 5 min) so wasted compute is fine.
- Grid search or random search; not Bayesian.

---

## Pattern B — Persistent Optuna workers sharing one study (Lab 2 uses this)

Each worker is a long-running process that pulls trials from a shared queue (the Optuna study), runs them, writes results back, and pulls the next trial. Multiple workers form a parallel sweep.

**The script** (schematic — the runnable version is `module_2/examples/optuna_sweep.sbatch`, which also sets the dataset path, an absolute database path, and a per-worker `--seed`):

```bash
#SBATCH --array=0-9            # 10 workers
#SBATCH --time=00:30:00        # each worker runs ~5 trials

# Each worker pulls from the same shared study + storage.
srun python optuna_sweep.py \
    --study-name lab2_sweep \
    --storage sqlite:///optuna.db \
    --n-trials 5
```

Where the worker does, in outline (schematic — the runnable `optuna_sweep.py` also wraps the SQLite URL in an `RDBStorage` with a 60 s lock timeout, shown below, seeds the sampler per worker, and loads the data with one fixed `--split-seed`):

```python
study = optuna.create_study(
    study_name="lab2_sweep",
    storage="sqlite:///optuna.db",
    sampler=TPESampler(),
    pruner=MedianPruner(),
    load_if_exists=True,   # crucial -- workers join an existing study
)
study.optimize(objective, n_trials=args.n_trials)
```

**Every worker must score its trials on the same rows.** `optuna_sweep.py` keeps two seeds apart: `--seed` (different per worker; the sbatch adds the task ID) varies only the sampler and the model initialisation, while `--split-seed` (default 42, the same for every worker) fixes the train/validation/test split. Each worker prints `Split seed 42 -> split fingerprint <hex>`; the fingerprint must be identical in every worker's log. If it is not, the study is comparing trials scored on different data.

**Pros:**
- Bayesian sampling (TPE) — each new trial's sampler sees every trial already finished and stored in the study.
- Pruning — a trial is compared with its peers in the study and stopped early if it is doing worse.
- Naturally tolerates worker failures (study survives; restart a worker, it picks up).

**Cons:**
- Requires a shared storage backend (an SQLite file, a journal file, or a database server).
- Shared storage means lock contention; how much trouble that causes depends on the backend and the filesystem (next section).
- Slightly more complex to debug (workers compete for trials; logs intermixed).

**When to use:**
- Search space >> trials you can afford (need TPE to be efficient).
- Trials are expensive (>1 min each) so pruning matters.
- You can tolerate slight shared-storage complexity.

**This is what Lab 2 uses.**

---

## Choosing a shared storage backend

For Pattern B, Optuna supports several backends. The options, and where each fits:

### SQLite (Lab 2 default — the course's observed configuration, not a guarantee)

`storage="sqlite:///path/to/optuna.db"`

- Pros: zero setup; one file on a filesystem all workers can reach.
- Cons: SQLite allows one writer at a time, and **every `trial.report()` is a write**. The load on the lock grows with both the number of workers and how often each one writes.
- **Upstream caveat.** Optuna's own documentation advises against parallel optimization with SQLite and warns specifically about network file systems ([Optuna 4.8 FAQ](https://optuna.readthedocs.io/en/v4.8.0/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3)). Your home directory on MedicineBow is a network file system (NFS). Lab 2 keeps SQLite because it needs no setup and it worked in the run below — not because it is a supported NFS configuration.

**What we observed on ARCC (`measured_results.md` §6).** With ten workers reporting **every epoch**, two of the ten crashed with `database is locked` (2026-09-24). Lab 2's `optuna_sweep.py` then changed two things, and with both the same ten workers ran 50 trials with no failures — on 2026-09-24, and again in the corrected re-run on 2026-09-25:

- it reports every **10** epochs (`REPORT_EVERY = 10`), cutting the writes tenfold;
- it opens SQLite with a **60 s lock timeout** instead of the default 5 s:

```python
storage = optuna.storages.RDBStorage(
    "sqlite:///optuna.db", engine_kwargs={"connect_args": {"timeout": 60}}
)
```

That is a before/after from our runs, not a guarantee. The timeout only makes a worker wait longer for the lock; it does not change how NFS handles locking, and under heavier load a worker can still fail. **If you still see `database is locked`:** the trials that finished are safe in the database, so resubmitting continues the study; if it keeps happening, tell the instructor. Journal storage (next) is the fallback.

### Journal storage (the fallback if SQLite keeps locking)

```python
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend   # Optuna 4.x name

storage = JournalStorage(JournalFileBackend("optuna_journal.log"))
```

(Older tutorials write `JournalFileStorage`; that is the pre-4.0 name, deprecated in the Optuna 4.8 we use.)

- Pros: Optuna's recommended file-based alternative to SQLite for parallel runs; it appends each change to a log file and includes lock mechanisms meant for network file systems ([JournalFileBackend documentation](https://optuna.readthedocs.io/en/v4.8.0/reference/generated/optuna.storages.journal.JournalFileBackend.html)).
- Cons: every write still goes through one file and one lock, so it has its own concurrency limits — many workers writing very often will still wait on each other. The file grows with every trial.
- **Not a drop-in replacement for the course scripts.** `optuna_sweep.py`, the `optuna create-study` step, `analyze_study.py` and Lab 2's Task 4 all name the storage as a URL string (`sqlite:///...`); a journal storage is a Python object built as above. Switching means changing every step that names the storage, together — which is why it goes through the instructor rather than being a one-line edit.

### MySQL / PostgreSQL (production scale)

`storage="mysql://user:pass@host:port/dbname"`

- Pros: built for many concurrent writers.
- Cons: requires a database server. This course does not use one.

(Optuna can also keep a journal in Redis. That too needs a running service, and this course does not use it.)

**For Lab 2:** use SQLite with the shipped settings, knowing the caveat above. If lock errors persist after a resubmit, tell the instructor; journal storage is the fallback. Document what happened in `prompts_sweep.md`.

---

## Practical SLURM tips for sweeps

### Sizing the array

```
total_trials = n_array_tasks × n_trials_per_worker
```

For Lab 2: 10 workers × 5 trials = 50 total trials. Measured on ARCC with pruning on (corrected run, 2026-09-25, `measured_results.md` §6): first trial start to last trial finish took **2 min 38 s**; 27 trials completed and 23 were pruned, 21 of those at epoch 100. Queue wait before the workers start is extra and varies. The 30-minute walltime per worker is generous on purpose.

### Pre-staging the study

Create the empty study from a login node, in your `sweep/` folder, **before** submitting workers. Use the same study name everywhere — here and in the lab it is `lab2_sweep`:

```bash
optuna create-study --study-name lab2_sweep \
    --storage "sqlite:///$(pwd)/optuna.db" --direction minimize
```

This avoids race conditions on the first sbatch submission. The `$(pwd)` makes the path absolute, so it names the same file the workers open.

### The environment

Optuna 4.8 is already installed in the course environment. Do **not** `pip install` into it:

```bash
source /project/me5475/setup5475.sh      # activates ml4sm: torch, optuna, plotly
```

`optuna_sweep.sbatch` activates the same environment inside each job and calls its Python by full path.

### Monitoring during the sweep

```bash
# in labs/lab_2/<your-github-handle>/sweep/
squeue -u $USER                                # workers visible
tail -F lab2-*.out                             # all worker logs live
grep -h "split fingerprint" lab2-*.out | sort -u   # must print exactly one line

# in labs/lab_2/<your-github-handle>/analysis/ -- see partial results
python analyze_study.py --storage sqlite:///../sweep/optuna.db --study-name lab2_sweep --out-dir .
```

### Recovering from worker failures

If a worker is killed mid-trial (walltime, `scancel`, node failure), that trial is **left in the RUNNING state** — nothing marks it failed unless you configure a heartbeat (`RDBStorage(..., heartbeat_interval=60)`). `analyze_study.py` counts it under "running". Every *finished* trial is safe in the database, so resubmitting workers continues the study; the lost configuration is simply not retried.

### Avoiding wasted compute

- Use `MedianPruner` (default in `module_2/examples/optuna_sweep.py`).
- Set worker walltime to about 3× (trials per worker × median trial time), not 10×.
- Set `gc_after_trial=True` in `study.optimize(...)` to free memory between trials.

---

## Common errors and fixes

| Error message | Likely cause | Fix |
|---------------|--------------|-----|
| `OperationalError: database is locked` (SQLite) | Lock contention on one SQLite file over NFS — a limitation Optuna warns about | Finished trials are safe: resubmit to continue the study. Keep the shipped `REPORT_EVERY = 10` and 60 s timeout (they helped in our run; no guarantee). If it persists, tell the instructor — journal storage is the fallback |
| `ERROR: dataset not found` (from the sbatch) | Lab 1 CSV not in `sweep/data/` | Copy it there (`lab_2.md` step 2a) |
| Study not found (`KeyError: 'Record does not exist.'`) | Wrong `--study-name`, or a `--storage` path that points at a different file — a relative `sqlite:///` path depends on the folder you run from, and SQLite silently creates an empty file there | Check the study name and the database path first; `optuna studies --storage <url>` lists the studies a database holds |
| Workers' `split fingerprint` lines differ | Mixed `--split-seed` values (or an old copy of `optuna_sweep.py`) | Give every worker the same split seed; move `optuna.db` aside and re-run with a fresh study |
| All trials end immediately with the same error | Bug in `objective()` | Debug one short trial in a throwaway study, inside an interactive allocation (below), not in the production study and not as heavy work on the login node |
| Worker walltime exceeded with trials remaining | Per-worker `n_trials` too high | Reduce `n_trials` per worker; spawn more workers |
| Best value is suspiciously low (< 1e-10) | A data problem — wrong units or scaling, or validation rows leaking into training — or a genuinely easy task (e.g. noise-free, nearly linear data) | Check the units and scaling, check that no validation row is also a training row, and ask whether the task is simply easy; then inspect the best trial |

The debug run, from your `sweep/` folder:

```bash
salloc --account=me5475 --partition=mb --time=00:30:00 --cpus-per-task=2 --mem=4G
# then, in the shell salloc gives you:
source /project/me5475/setup5475.sh
python optuna_sweep.py --data data/single_element.csv \
    --study-name debug --storage sqlite:///debug.db --n-trials 1 --max-epochs 200
```

Delete `debug.db` afterwards; it is not a deliverable.

---

## Pattern A vs B at a glance

```
                  Pattern A (array, nothing shared)   Pattern B (Optuna workers, one study)
                  ---------------------------------   -------------------------------------
Use case          Grid search                         Bayesian + pruning
                  Random search                       Lab 2

Shared state?     None                                Required (SQLite / journal / DB server)

Learns across     No                                  Yes (TPE sees stored trials)
trials?

Pruning?          Local early stopping only           Yes, against peers (Median, Hyperband, ...)

Lock contention?  None from shared storage            Possible; depends on backend + filesystem

Failure mode      Trial dies in isolation             Worker dies; study continues

Complexity        Lowest                              Moderate

Lab 2 uses?       No                                  Yes
```

---

## TL;DR

For Lab 2: **Pattern B** with **SQLite** and the shipped settings — the configuration we observed working on ARCC, not a guaranteed one (Optuna advises against parallel SQLite on NFS). If lock errors persist, tell the instructor; **journal storage** is the fallback. Use one study name (`lab2_sweep`) and one split seed everywhere. Pre-stage the study on a login node, then submit the array, then watch with `squeue` and `tail -F`, and check that the split fingerprints match. Use `analyze_study.py` for the final analysis.
