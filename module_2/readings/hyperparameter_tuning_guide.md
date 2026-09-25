# Hyperparameter Tuning Guide

*Module 2. The reading for Lecture 10 (Wed Sep 23), required before Lecture 11 (Fri Sep 25). Runnable companions: `module_2/examples/lr_finder.py`, `optuna_sweep.py`, `optuna_sweep.sbatch`, `lab2_preliminaries.sbatch` and `analyze_study.py`, also in `/project/me5475/examples/` on ARCC.*

A cheatsheet for Lab 2, and for any later sweep you choose to run (tuning in the midterm project is recommended, not required). Every measured number here comes from `module_2/readings/measured_results.md`, with the section named next to it. Anything labelled *rule of thumb* was not measured in this course.

**Symbols (L9, L10).** η is the **learning rate** (`lr` in code). λ is **weight decay** (`weight_decay`), not L5's Hessian eigenvalue.

**Where it runs on ARCC.** Anything that trains a network runs as a **submitted batch job** on a compute node, never on the login node. Lab 2's range test runs inside `lab2_preliminaries.sbatch`, and the sweep is `optuna_sweep.sbatch`. The login node is for copying files, creating the study, `sbatch`, and reading results.

---

## The two-phase workflow

**Phase 1 — Find the plausible decade for η with the LR range test.**

The range test (Smith, WACV 2017) trains one network for up to a hundred steps, raising η at every step, and records the loss (the script stops early once the loss blows up):

```python
# Pseudocode -- see module_2/examples/lr_finder.py for the real version.
# One model, one optimizer (Adam), full batch; eta rises every step.
for lr in np.geomspace(1e-7, 1e-1, 100):       # --lr-min, --lr-max, --steps
    set_optimizer_lr(lr)
    take_one_training_step()
    record(loss)
plot(losses, x=lrs, log_x=True)
```

Read off the plot:

- Where the loss **starts to fall**. Rates well below this barely train. You read this off the plot; the script does not print it.
- The **wall**. `lr_finder.py` takes the first rate past the loss minimum at which the loss exceeds **twice** that minimum. It is an empirical loss-rise threshold, not a proven stability limit.
- The **suggested rate = the wall ÷ 10**. The factor of ten is the course's heuristic, not a rule from Smith's paper, and the script prints a suggestion only if the sweep reaches such a rate at all.

**It gives a decade, not a number** (`measured_results.md` §3). On the analytic Lab 1 dataset the suggestion depended on the seed: 1.07e-03 for seed 0, 1.00e-02 for seeds 1 and 42, and no suggestion at all for seeds 7 and 123. Seed 42's 1.00e-02 is `--lr-max` ÷ 10, the end of the sweep, not a measurement.

Use it to sanity-check the `lr` range you give Optuna, not to replace the search. It tests one network (`--hidden 32 --layers 4`), full batch, Adam, no weight decay; the sweep varies width, depth, batch size and λ, and uses AdamW. Lab 2 ships `lr` over four decades, [1e-5, 1e-1]. Narrowing it is a choice you can make when your range-test plot supports it; how much that saves was not measured.

**Phase 2 — Run the Optuna sweep.**

What one Lab 2 worker does, cut down from `module_2/examples/optuna_sweep.py`. This is schematic: the real worker also loads the data with one shared `--split-seed`, builds the network, and opens SQLite with a 60 s lock timeout (`slurm_sweep_patterns.md`).

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

REPORT_EVERY = 10                     # each report is a write to the shared database

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                     # eta
    width = trial.suggest_categorical("width", [16, 32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 6)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)  # lambda
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, "full"])
    # ... build the MLP; torch.optim.AdamW(..., lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    for epoch in range(max_epochs):
        # ... one epoch of training, then val_loss on the validation rows ...
        best_val = min(best_val, val_loss)
        if epoch % REPORT_EVERY == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        # ... early stopping, patience 200 ...
    return best_val                   # the best validation loss, not the last one

study = optuna.create_study(
    study_name="lab2_sweep",
    storage="sqlite:///optuna.db",    # the real worker adds the 60 s lock timeout
    sampler=TPESampler(seed=seed),    # --seed: 42 + array task ID, one per worker
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    direction="minimize",
    load_if_exists=True,              # every worker joins the same study
)
study.optimize(objective, n_trials=5) # 5 per worker, 10 workers: 50 trials
```

Lab 2 runs ten of these workers as one SLURM array, all joining one study (Pattern B in `slurm_sweep_patterns.md`). The course's measured run: 50 trials, 27 completed, 23 pruned (`measured_results.md` §6).

---

## Which `suggest_*` method for which hyperparameter

Lab 2's five, exactly as `optuna_sweep.py` searches them:

| Hyperparameter | Method | Lab 2's range | Why |
|----------------|--------|---------------|-----|
| Learning rate η (`lr`) | `suggest_float(..., log=True)` | [1e-5, 1e-1] | Useful values span decades. Drawn uniformly on a linear scale, about nine draws in ten would land above 1e-2. **Log-scale it.** |
| Weight decay λ (`weight_decay`) | `suggest_float(..., log=True)` | [1e-7, 1e-2] | Same reason. Passed to AdamW, which applies the decay outside Adam's adaptive denominator (L9, L10). |
| Width (`width`) | `suggest_categorical` | {16, 32, 64, 128} | A chosen discrete set; powers of two by convention. One width, shared by every hidden layer. |
| Depth (`n_layers`) | `suggest_int` | [2, 6] | An integer range. `n_layers` counts `nn.Linear` layers, which is L, the number of weight matrices; so [2, 6] means 1 to 5 hidden layers. |
| Batch size (`batch_size`) | `suggest_categorical` | {16, 32, 64, "full"} | A discrete set. "full" means one batch of all N training examples. |

Others you may meet later. **None is in Lab 2's sweep**, so no course range is given:

- **Dropout rate p**: `suggest_float` on a linear scale. L10 names dropout but does not use it for this course's small networks; add it only with a plot that shows it helped.
- **Scheduler patience**: `suggest_int`. Relevant from M3 on, when ReduceLROnPlateau comes in (L9).
- **Optimizer choice**: `suggest_categorical`, with the branch written inside the objective. If λ > 0, the Adam-family option should be AdamW, not Adam with an L2 term in the loss (L10).
- **Knobs that exist only for some settings**, such as a separate width for layer 5 only when `n_layers` ≥ 5: call `suggest_*` inside an `if`. TPE copes with such conditional spaces; that is the "tree-structured" in its name. Lab 2 sweeps one shared width instead.

---

## Pruning — what the pruner actually compares

1. **`trial.report(value, step)` followed by `trial.should_prune()` is what makes pruning possible.** No reports, no pruning. Lab 2 reports every **10** epochs, not every epoch: each report is a database write, and with ten workers reporting every epoch to one SQLite file on ARCC, two of them crashed with `database is locked` (`measured_results.md` §6, history).
2. **What `MedianPruner` compares.** At a reported step it takes the trial's **best reported value so far** and compares it with the **median of the completed trials' reported values at that same step**. If the trial is worse, it is pruned. Only **completed** trials are peers; pruned and running trials are not.
3. **The two guards are Lab 2's settings.** `n_startup_trials=5`: no pruning until five trials have completed (Optuna's default is also 5). `n_warmup_steps=100`: no pruning before reported step 100, which is epoch 100 here because the step is the epoch (Optuna's default is 0, no warm-up). Larger values of either make pruning start later.
4. **Pruned is not failed.** A pruned trial is stored as PRUNED with its reports kept, and TPE still uses it (next section). In the course's run 23 of 50 trials were pruned, 21 of them at epoch 100, the first check the warm-up allows (`measured_results.md` §6). That is one observation, not a target rate; zero pruned trials can also be legitimate.

---

## Choosing a sampler

**TPE is Optuna's default for a single objective, and what Lab 2 uses.** TPE (Tree-structured Parzen Estimator; Bergstra et al., NeurIPS 2011) uses previous trials' scores to propose promising settings. In outline, as Optuna 4.8 does it:

1. **Start random.** Until ten trials have *finished* (completed or pruned), it samples at random. Its `n_startup_trials=10` is a different count from the pruner's five *completed*. With ten workers launching at once, more than ten trials can start before ten have finished.
2. **Split.** Of the n finished trials, the best min(⌈0.1 n⌉, 25) are *good* and the rest *bad*. Completed trials are ranked by score first; a pruned trial joins *good* only if there are too few completed ones to fill it.
3. **Model.** It smooths the *good* settings into one density and the *bad* settings into another. That smoothing, a small bump on each trial's setting summed up, is a Parzen estimator.
4. **Propose.** It draws candidates and picks the one where *good* ÷ *bad* is largest.

The alternatives. The course's rule of thumb (L10, not a measured result): 1–2 cheap knobs, grid; 5 or more expensive knobs, TPE; anything between, random.

- **Random search**, `optuna.samplers.RandomSampler`. The baseline: it learns nothing between trials, so it also works as a SLURM array with nothing shared (Pattern A in `slurm_sweep_patterns.md`). Bergstra & Bengio (JMLR 2012): a few hyperparameters often dominate, and random search tries more distinct values of each one than a grid of the same size.
- **Grid search**, `optuna.samplers.GridSampler`. For one or two knobs with a handful of values each; its cost multiplies with every knob added.
- **Gaussian-process Bayesian optimization**, `optuna.samplers.GPSampler`, built into Optuna 4.8. Not used in this course.
- **Population-based training** changes hyperparameters *during* training. It is not an Optuna sampler (Ray Tune implements it) and is not used in this course.

---

## Reading the analysis outputs

`module_2/examples/analyze_study.py` prints a summary (best value and its hyperparameters, the top three completed trials, importance) and writes four HTML files into `--out-dir` (default `study_analysis/`; Lab 2 passes `--out-dir .`). **All four plots show completed trials only**; pruned trials are not drawn (checked in Optuna 4.8), so the course's 50-trial run plots 27 points. Read them in this order:

1. **`optimization_history.html`**: a dot per completed trial, and a line for the best value so far. That line can only step down, by construction, so the useful question is *when* it last stepped. Still stepping near the end suggests more trials could help; flat since early on means later trials found nothing better within these ranges. Look at the other three plots before blaming the sampler.
2. **`param_importances.html`**: fANOVA importance, Optuna's default evaluator. It fits a random forest to the completed trials and normalises the scores to sum to 1. **Treat it as exploratory.** The course's run gave `lr` 0.75, from 27 adaptively chosen completed trials, and a rerun can reorder the small values (`measured_results.md` §6). A short bar means the hyperparameter explained little of the variation within these ranges, in this run; it does not mean the hyperparameter is irrelevant or can be fixed at any value.
3. **`parallel_coords.html`**: one line per completed trial. Optuna puts the objective **first**: the leftmost axis, labelled *Objective Value*, on a linear scale. Start from the lowest values on that axis and follow those lines right to see what their settings share. L11's slide uses a different order and a log scale, so find the labelled objective axis rather than assuming a side.
4. **`slice.html`**: one panel per hyperparameter. Each dot is one completed trial, placed at its value of that hyperparameter. The other hyperparameters differ from dot to dot (nothing is held fixed or averaged out), so a clear trend is a hint and scatter is expected.

---

## When something looks wrong

| Symptom | Possible cause | What to try |
|---------|----------------|-------------|
| Most pruned trials stop at epoch 100 | Epoch 100 is the first check the warm-up allows; the course's run did the same (21 of 23, `measured_results.md` §6) | Not a fault by itself. If you suspect slow starters are being killed, try a larger `n_warmup_steps` in a fresh study and compare (not measured in this course) |
| No trials pruned | Can be legitimate. Pruning also needs reports, five completed trials, trials that run past epoch 100, and a trial whose best is behind the median | Check the worker calls both `trial.report` and `trial.should_prune`; otherwise accept it |
| Best value no better than a hand-picked baseline | The ranges miss the good region, or too few trials | Compare the `lr` range with your range-test decade; widen the range of the hyperparameter that looks most important; run more trials |
| Best trials sit at the edge of a range | The optimum may lie outside it | Widen that range on that side, in a fresh study so every trial draws from the same ranges |
| Importance step prints `Cannot evaluate parameter importances with only a single trial.` | Only one completed trial (checked in Optuna 4.8) | Let more trials complete. With only a few, a ranking is computed but means little |
| Importance and plots seem to disagree | Importance is a random-forest estimate from a few adaptively chosen trials | Treat the ranking as exploratory; check the slice plot; do not rebuild your search on one run's ranking |
| Workers' `split fingerprint` lines differ | Mixed `--split-seed` values | See *Common errors and fixes* in `slurm_sweep_patterns.md` |

---

## A note on reproducibility

The Lab 2 worker uses three seeds, each with one job:

| Seed | Where it is used | What it fixes |
|------|------------------|---------------|
| `--split-seed` (default 42) | `load_and_split()` | The train/validation/test rows and their scaling. **The same for every worker**, and for Task 4. Every worker logs a split fingerprint, and they must all match. |
| `--seed` (42 + array task ID) | `TPESampler(seed=...)` | That worker's sampler. Different per worker, on purpose. |
| `--seed` + trial number | `torch.manual_seed` and `np.random.seed`, inside `objective()` | Initialisation and shuffling for that trial. |

**What that buys.** One trial's training can be rerun with the same hyperparameters, seed and split. L10's `reproducibility_checklist.py` checks the seeding part on CPU (two short runs with one seed give identical final weights); GPU determinism is a manual check.

**What it does not buy: the same study twice.** With ten workers, which trials have finished when a worker asks for its next suggestion depends on timing, so TPE's proposals and the pruner's decisions (made against the trials completed so far) change from run to run. Optuna's own FAQ says a seeded sampler is reproducible when a study runs sequentially, and that parallel or distributed mode has inherent non-determinism ([Optuna 4.8 FAQ](https://optuna.readthedocs.io/en/v4.8.0/faq.html#how-can-i-obtain-reproducible-optimization-results)). Treat one sweep as one draw: record the study name, storage, seeds and git SHA, and do not expect a rerun to give trial *n* the same hyperparameters.

---

## Sources

- Smith, *Cyclical Learning Rates for Training Neural Networks*, WACV 2017 (the range test).
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13, 2012.
- Bergstra, Bardenet, Bengio & Kégl, *Algorithms for Hyper-Parameter Optimization*, NeurIPS 2011 (TPE).
- Akiba, Sano, Yanase, Ohta & Koyama, *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 (AdamW).
