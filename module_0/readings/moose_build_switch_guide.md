# The course MOOSE build has moved: how to update a job script

*Course reading · ME 5475 · Released Wed Oct 14, 2026 with Lab 4. Updated as later course scripts move to the new
build.*

## What changed

The course is moving to a new MOOSE build on ARCC:

```
/project/me5475/software/moose_torch/me5475/me5475-opt
```

It is the same MOOSE version as before, with the same solid-mechanics capabilities. It also supports evaluating a
trained neural network inside a MOOSE simulation; Module 5's instructions for that come with Module 5.

**The checked examples reproduce their previous results.** The eight course MOOSE inputs (the Lab 0 cantilever
and plate, the Lab 1 load sweep, the two diffusion problems and the plate reference from Module 3, the Lab 4 J2
input, and the Module 6 RVE) were run on both builds, and every value in their CSV outputs agrees to round-off
(`module_0/readings/measured_results.md` §E). The Lab 4 dataset regenerates byte for byte
(`module_4/readings/measured_results.md` §1.2). The old executable keeps working for the rest of the semester, so
nothing you already ran is invalid.

## What is updated in this release

**Updated in this release (Wed Oct 14, with Lab 4):** `module_4/examples/generate_data.sbatch`, Lab 4's
dataset job. Lab 4's MOOSE input `j2_cyclic_loading.i` changed only in its comments.

Scripts not listed still name the old executable. They keep working, and later releases update them. To get the
updated scripts:

```bash
git checkout main
git pull
```

## Editing your own job script (two lines)

If you kept your own copy of a job script, make both edits below.

**1. The executable.** Replace the old line:

```bash
MOOSE_EXE=/project/me5475/software/rom_opt_arcc/rom_opt-opt
```

with:

```bash
MOOSE_EXE=/project/me5475/software/moose_torch/me5475/me5475-opt
```

**2. The library path.** The old scripts added a second directory after the compiler's library directory:

```bash
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:/project/me5475/software/moose/framework/contrib/hit:${LD_LIBRARY_PATH:-}"
```

The new build does not need it. Use:

```bash
export LD_LIBRARY_PATH="$GCC_LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"
```

Keep the `GCC_LIBSTDCXX_DIR` part. It stops the Python environment's older C++ library from being picked up. Leave
the `module load` lines as they are.

**Make both edits.** If you change only the first line, the job still runs. But we checked, and the new executable
then loads one of its libraries from the old build through that second directory: a mixture of the two builds that
the course does not support.

## Checking that it works

After editing a job script, rerun an example you have already run with it, from the folder that holds its input
and job script. For example, for the Lab 0 cantilever: make both edits in your copy of
`module_0/examples/run_cantilever.sbatch`, then, in that folder:

```bash
sbatch run_cantilever.sbatch
```

The CSV output should match your earlier run.

## If something goes wrong

Email the instructor with the job ID and the `.err` file. Meanwhile, you can go back to the old pair: restore
**both** the old `MOOSE_EXE` line **and** the old `LD_LIBRARY_PATH` line, or use your previous copy of the job
script. The old executable is kept for the rest of the semester.
