# SLURM Cheatsheet — Reference for the Whole Semester

A compact one-pager you should keep handy whenever you submit anything to ARCC.

---

## The four commands you use every day

| Command | What it does |
|---------|--------------|
| `sbatch script.sbatch` | Submit a batch job. Returns a job ID. |
| `squeue -u $USER` | Show all your queued and running jobs. |
| `scancel <jobid>` | Cancel a job (queued or running). |
| `sacct -j <jobid>` | After-the-fact resource accounting (CPU time, memory peak). |

Bonus: `salloc` reserves nodes for an *interactive* session (`salloc --time=1:00:00 --partition=...`) — useful for debugging.

---

## Anatomy of an `#SBATCH` directive

Every directive lives at the top of the script, before any executable command. The form is `#SBATCH --key=value`. They are read by SLURM as comments by bash.

### Identity and accounting

```bash
#SBATCH --job-name=descriptive_name      # shows in squeue
#SBATCH --account=me5475                 # this course's allocation
#SBATCH --mail-type=END,FAIL             # when to email
#SBATCH --mail-user=you@uwyo.edu         # where to email
```

### Where the job runs

```bash
#SBATCH --partition=mb                   # which queue (mb = CPU; mb-l40s,mb-a30 = GPU)
#SBATCH --nodes=1                        # how many nodes
#SBATCH --ntasks=1                       # total MPI ranks (across all nodes)
#SBATCH --ntasks-per-node=4              # MPI ranks per node (alternative to --ntasks)
#SBATCH --cpus-per-task=1                # OpenMP threads per rank
#SBATCH --gres=gpu:1                     # GPUs (if applicable)
```

**The mental model.** Total cores requested = `nodes × ntasks-per-node × cpus-per-task`. For pure-MPI MOOSE on 4 cores: `--nodes=1 --ntasks=4 --cpus-per-task=1`. For OpenMP-only PyTorch on 8 threads: `--nodes=1 --ntasks=1 --cpus-per-task=8`. For hybrid: both bigger than 1.

### How long, how much memory

```bash
#SBATCH --time=01:00:00                  # walltime HH:MM:SS (or D-HH:MM:SS)
#SBATCH --mem=8G                         # memory per NODE
#SBATCH --mem-per-cpu=2G                 # memory per CORE (alternative)
```

### Output

```bash
#SBATCH --output=run-%j.out              # %j = SLURM job ID
#SBATCH --error=run-%j.err               # separate stderr (optional)
```

### Job arrays (Lab 0 uses this!)

```bash
#SBATCH --array=0-4                      # run as 5 tasks, ID 0..4
#SBATCH --output=run-%A_%a.out           # %A = job, %a = array task
```

Inside the script, `$SLURM_ARRAY_TASK_ID` holds the current task ID. Use it to vary inputs:

```bash
moose-opt -i input.i Mesh/uniform_refine=$SLURM_ARRAY_TASK_ID
```

---

## Inside the script — the runtime portion

After the directives, write the actual workload as plain bash:

```bash
# 1. Reset modules to a clean state.
module purge

# 2. Load what this job needs.
module load python/3.11
module load openmpi/4.1.5                 # if MPI is needed

# 3. Activate any conda env (after module loads).
source /project/yourgroup/envs/moose/bin/activate

# 4. Move to the working directory (sbatch starts you in the submit dir,
#    but be explicit).
cd $SLURM_SUBMIT_DIR

# 5. Launch with srun. srun inherits SLURM's task topology automatically.
srun python your_script.py
# OR
srun moose-opt -i input.i
```

---

## Common SLURM environment variables

Available inside the script:

| Variable | Meaning |
|----------|---------|
| `$SLURM_JOB_ID` | The unique job ID |
| `$SLURM_JOB_NAME` | The `--job-name` you set |
| `$SLURM_SUBMIT_DIR` | The directory you ran `sbatch` from |
| `$SLURM_NTASKS` | Total MPI ranks |
| `$SLURM_CPUS_PER_TASK` | OpenMP threads per rank |
| `$SLURM_NODELIST` | Compute nodes the job runs on |
| `$SLURM_ARRAY_TASK_ID` | (Array jobs only) which task this is |
| `$SLURM_ARRAY_JOB_ID` | (Array jobs only) the parent array job ID |

---

## Reading `squeue` output

```
JOBID PARTITION  NAME      USER  ST  TIME  NODES  NODELIST(REASON)
12345   mb     plate_conv  jdoe   R  0:32      1  mbcpu-001
12346   mb     plate_conv  jdoe  PD  0:00      1  (Priority)
```

`ST` is the most important column:

- `R` — running
- `PD` — pending (waiting for resources)
- `CG` — completing (cleaning up)
- `F` — failed; check the `.err` and `.out` files
- `CA` — cancelled
- `TO` — timed out (walltime exceeded)

`NODELIST(REASON)` shows why a PD job is waiting: `(Priority)`, `(Resources)`, `(QOSMaxJobs)`, etc.

---

## After a job finishes — what to check

1. `tail run-<jobid>.out` — did the workload finish or die mid-way?
2. `tail run-<jobid>.err` — any errors / warnings?
3. `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,NodeList` — actual resources used. If `MaxRSS` is close to your `--mem` request, increase memory for next time. If `Elapsed` is far below `--time`, ask for less walltime to schedule faster.

---

## Five idioms you will use often

**1. Submit and immediately watch.**

```bash
JOBID=$(sbatch script.sbatch | awk '{print $NF}')
watch -n 2 squeue -u $USER
```

**2. Submit an array and tail all outputs as they appear.**

```bash
sbatch script.sbatch
tail -F run-*_*.out
```

**3. Submit only after another job finishes.**

```bash
sbatch --dependency=afterok:$JOBID downstream.sbatch
```

**4. Interactive shell on a compute node, for debugging.**

```bash
salloc --account=me5475 --partition=mb --time=1:00:00 --ntasks=4 --mem=8G
# now your shell is on a compute node
srun --pty bash    # if needed
```

**5. Hold a queued job (do not run yet), then release.**

```bash
scontrol hold <jobid>
scontrol release <jobid>
```

---

## When SLURM yells at you

| Error message | Likely cause | Fix |
|---------------|--------------|-----|
| `error: Batch job submission failed: Invalid account or account/partition combination specified` | Wrong `--account` for this partition | Check your account string and partition allow-list |
| `JOB <id> CANCELLED AT <time> DUE TO TIME LIMIT` | Walltime too short | Increase `--time` |
| `slurmstepd: error: Detected 1 oom-kill event` | Ran out of memory | Increase `--mem` (or `--mem-per-cpu`) |
| `srun: error: Unable to create step for job <id>: More processors requested than permitted` | Asked for more cores than the node has | Reduce `--ntasks` × `--cpus-per-task` to fit the node |

---

## MedicineBow (UW ARCC) — course-specific values

Confirmed 2026-06-09. The course uses one dedicated allocation, `me5475`:

| Use case | Account | Partition | Notes |
|----------|---------|-----------|-------|
| CPU work — MOOSE, light ML training, RVE data generation | `me5475` | `mb` | General MedicineBow CPU pool (25 nodes) |
| GPU work — PINN, CNN, U-Net, FNO, LSTM | `me5475` | `mb-l40s,mb-a30` | L40S / A30 GPUs; needs `--gres=gpu:1` |

**Login:** `ssh <netid>@medicinebow.arcc.uwyo.edu`

**Storage:**
- `/home/<netid>` — small, backed up. Code only.
- `/project/me5475/` — shared course storage (5 TB). Conda env, MOOSE binary, reference datasets.

**Environment bootstrap** (already in every SLURM script):

ML-only jobs (GPU: PINN, CNN, FNO, LSTM; CPU: MLP, Optuna, GNN):
```bash
module purge
module load miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
```

MOOSE jobs (M0 convergence study, M6 RVE data generation):
```bash
module purge
module load arcc/1.0 gcc/14.2.0 openmpi/5.0.5 \
    "hdf5/1.14.3__hl_True__fortran_False-ompi" miniconda3/24.3.0
source /apps/u/opt/linux/miniconda3/24.3.0/etc/profile.d/conda.sh
conda activate /project/me5475/envs/ml4sm
GCC_LIBSTDCXX_DIR="$(dirname "$(gcc --print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"
export MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt
```
