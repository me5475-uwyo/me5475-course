# Lab 0 — Setup, Agentic Workflow, and a First Convergence Study

**Module:** 0
**Released:** Fri Aug 28, 2026 (before classes begin — you can start Task 2 after Lecture 2)
**Due:** Fri Sep 11, 2026, 11:59 PM Mountain Time (the day L5 is taught)
**Weight:** 30% / 9 ≈ 3.33% of total course grade (all nine labs are weighted equally)
**Submission:** Pull request to your private course repo (github.com/me5475-uwyo), on a branch named `lab_0`. Merge after instructor sign-off.

---

## What this lab does

Lab 0 closes Module 0 by combining everything you saw across the three lectures into a single end-to-end workflow: AI coding agents (Lecture 2) drive a mesh-refinement convergence study (Lecture 3) running on ARCC (Lecture 3) for a plate-with-hole problem (Lectures 2 & 3) that returns in Module 3 with PINNs. By the time you submit, you will have checked two physical metrics against appropriate references: plane-strain peak von Mises stress and the hoop stress at the hole top.

---

## Deliverables

Submit a single PR with the following directory structure:

```
labs/lab_0/<your-github-handle>/
├── in_class/
│   ├── weak_form_derivation_lead.md       (from L2 in-class lab)
│   ├── weak_form_derivation_review.md     (from L2 in-class lab)
│   ├── mlp_sine_lead.py                   (from L2 in-class lab)
│   ├── mlp_sine_review.md                 (from L2 in-class lab)
│   └── prompts_l2.md                      (the prompts you used in L2)
├── moose_local/
│   ├── install_log.txt                    (output of the install agent)
│   ├── plate_with_hole_screenshot.png     (von Mises field from ParaView)
│   └── prompts_l3_install.md
├── arcc_hello/
│   ├── hello.sbatch                       (your modified copy)
│   ├── hello-<jobid>.out                  (the SLURM output)
│   └── prompts_l3_hello.md
├── convergence_study/
│   ├── input_files/
│   │   ├── plate_with_hole_refine_0.i     (each refinement level as a separate file
│   │   ├── plate_with_hole_refine_1.i      OR a single .i + a wrapper SLURM script
│   │   ├── plate_with_hole_refine_2.i      that varies uniform_refine via CLI override.
│   │   ├── plate_with_hole_refine_3.i      Either is acceptable -- show your choice.)
│   │   └── plate_with_hole_refine_4.i
│   ├── run_convergence_study.sbatch       (your SLURM array script)
│   ├── outputs/
│   │   ├── plate_with_hole_refine_0_out.csv
│   │   ├── plate_with_hole_refine_1_out.csv
│   │   ├── plate_with_hole_refine_2_out.csv
│   │   ├── plate_with_hole_refine_3_out.csv
│   │   └── plate_with_hole_refine_4_out.csv
│   ├── extract_stress.py                  (your version, evolved from the starter)
│   ├── convergence.png                    (the log-log plot)
│   └── prompts_convergence.md
└── reflection.md                          (<= 400 words; see below)
```

---

## Tasks

### Task 1 — In-class outputs (from Lecture 2's lab)

You already produced these in class. Copy them into `in_class/` and commit. Nothing further to do.

### Task 2 — Local MOOSE install (homework assigned in Lecture 2; due before Lecture 3)

This task is the *pre-class homework for Lecture 3*. Complete it between L2 and L3.

**You cannot be blocked by this task.** MOOSE is already installed and ready on ARCC — Task 4
uses it, and so does every later lab that needs a simulation. A local build is a convenience, not a
prerequisite. What this task is actually about is **using an AI coding agent on a genuinely hard,
dependency-heavy install, and then verifying its claims instead of trusting them.** That skill is the
point; MOOSE is just a demanding vehicle for it. **Both outcomes earn full credit:** a working install,
or a documented failure with your logs and review notes. Attempt it seriously — the debugging is where
the learning is — but do not lose a weekend to it.

Open your AI coding agent (ChatGPT/Codex, Copilot, Claude Code, Cursor, or equivalent) and use the prompt in `module_0/examples/install_moose_prompt.md`. Run a review prompt independently after install reports success. Commit the install log and the von Mises screenshot from running `plate_with_hole.i` locally with `uniform_refine = 0`.

**Which operating system you have changes the task, not the credit.**

- **macOS or Linux** — attempt the install as written.
- **Windows with WSL** — attempt it inside WSL, exactly as you would on Linux.
- **Windows without WSL** — do **not** install WSL just for this. MOOSE has no native Windows build,
  so your version of the exercise is to establish that *and prove it*: ask your lead agent whether
  MOOSE can be built natively on Windows and how it knows, then open a fresh session and have a
  review agent check the claim independently. Log both prompts and both answers. An agent that
  asserts a confident wrong answer here — and they sometimes do — is exactly the failure mode
  Lecture 2 is about, and catching it is worth as much as a successful build.

The struggle is the assignment. Wrestling with a dependency-heavy build, reading an error you have
never seen, and deciding whether the agent's explanation is true — that is the skill this task is
training, and it transfers to every install you will ever do. What is *not* useful is grinding past
the point of learning: if the install fails after two earnest attempts, document the failure in
`prompts_l3_install.md`, bring your laptop to L3 anyway (we will troubleshoot in the first 5 minutes),
and run on ARCC instead.

**Getting the screenshot without a local install.** Your Task 4 array job runs `uniform_refine = 0` as
one of its five levels, which produces `plate_with_hole_refine_0_out.e` — the same field you would have
produced locally. Download it (portal → **Files** → **Home Directory** → Download, or `scp`) and open it
in **ParaView**, which has native Windows, macOS, and Linux builds. So the screenshot is reachable on any
operating system. *(MOOSE itself has no native Windows build — a local install on Windows needs WSL.)*

> **If your ARCC account is not active yet:** tell me. Tasks 3 and 4 need cluster access,
> and no deadline or grade penalty attaches to work you could not do for lack of a working
> account (syllabus §4). We will re-time those tasks for you individually.

### Task 3 — ARCC hello-world (from Lecture 3)

Submit the provided `hello.sbatch` after replacing `<your-email>` with your address and confirming the shipped course account (`me5475`) and CPU partition (`mb`). Commit the output log.

### Task 4 — Main task: agent-driven convergence study

This is the substantive part of the lab. Using your lead+review agent workflow:

**4a.** Modify `plate_with_hole.i` to run with `uniform_refine ∈ {0, 1, 2, 3, 4}`. You may either generate five separate `.i` files or pass the value as a CLI override to a single file — your choice. Whichever you pick, document why in `prompts_convergence.md`.

**4b.** Modify the provided `run_convergence_study.sbatch` SLURM array script for your ARCC account/partition. Submit it. Confirm all five jobs complete successfully.

**4c.** Modify the provided `extract_stress.py` to read your five CSV outputs and extract `max_vonmises_stress`, `max_stress_xx`, and `num_elements`. Produce a log-log convergence plot with both peak-stress series and both infinite-plate reference lines:

- **Plane-strain von Mises:** reference ≈2.67 for ν = 0.3, because `3√(1 − ν + ν²) ≈ 2.67`; expect ≈**2.48** at refine 4.
- **Hoop stress:** `max_stress_xx` tracks the hoop component at the hole top; the Kirsch reference is ≈3, and the refine-4 expectation is ≈**2.88**.

The outer boundary is a circular arc carrying the exact uniform-far-field traction, so do not apply a finite-square-plate `W/d` correction.

**4d.** Inspect both series. Confirm that both sequences increase monotonically and that the refine-4 peaks are within ±2% of 2.48 (von Mises) and 2.88 (hoop). Report each finest-level gap from its infinite-plate reference (≈2.67 / ≈3), estimate the observed rate, and explain why the localized peaks are still climbing at refine 4.

**Required artifacts:** all input files, the SLURM script, the five output CSVs, the parsing/plotting script, the final plot.

### Task 5 — Reflection (≤ 400 words)

Write `reflection.md` answering all three:

1. **Numerical analysis question.** Compare both peak-stress sequences across the five refinement levels. Are the refine-4 peaks within ±2% of 2.48 (plane-strain von Mises) and 2.88 (hoop)? What gaps remain to the ≈2.67 / ≈3 infinite-plate references, and why are the localized peaks still climbing at refine 4?
2. **Agent reliability question.** Across your three agent sessions (install, SLURM, plotting), give *one specific instance* where your lead agent produced something the review caught, and *one specific instance* of something neither agent caught that you fixed yourself.
3. **Process question.** Did the agentic workflow save you time on this lab, cost you time, or both? Be specific about which subtask. You are not penalized for an honest answer in either direction — we want to calibrate.

---

## Grading rubric (out of 100)

| Component | Points | What we look for |
|-----------|--------|-----------------|
| In-class outputs (Task 1) | 10 | Files present and committed; prompts logged |
| Local MOOSE install (Task 2) | 10 | Install verified OR a documented failure with fallback to ARCC |
| ARCC hello (Task 3) | 10 | SLURM output file with the expected hostname / job ID printed |
| Convergence study setup (Task 4a + 4b) | 20 | Input files / SLURM array run cleanly; outputs collected |
| Convergence study analysis (Task 4c + 4d) | 25 | Both series are monotone; refine-4 peaks are within ±2% of 2.48 / 2.88; plot includes the ≈2.67 / ≈3 reference lines; continued peak growth is explained |
| Reflection (Task 5) | 15 | All three questions answered concretely with specifics |
| Prompts logged (across all tasks) | 10 | `prompts_*.md` files present with paraphrased prompts |

**Penalty conditions.**

- Submitting agent output without evidence of review: -10 points.
- Convergence plot omits either required reference line: -5 points.
- Empty or generic reflection (no specific examples cited): -10 points.

---

## Hints (read after you have spent at least 30 minutes on each task)

- **Expected baseline behavior** (MedicineBow validation of the shipped `plate_with_hole.i`). For refine 0–4, `max_vonmises_stress ≈ 1.25, 1.56, 1.93, 2.25, 2.48`, while `max_stress_xx ≈ 1.62, 2.00, 2.38, 2.68, 2.88`; both sequences are monotone. Instructor runs through refine 6 reach ≈2.72 / ≈3.10, so the localized peaks are not fully converged at refine 4.
- **MOOSE input file override.** You can pass `Mesh/uniform_refine=N` on the command line to override the value in the .i file. This is how a single .i file plus a SLURM array task can drive all 5 refinement levels.
- **CSV reading.** MOOSE writes the postprocessor values as a CSV with one row per timestep. For a Steady executioner there is just one data row plus the header — `df.iloc[-1]` after `pd.read_csv` gets you the values.
- **Element count.** Look at the `num_elements` column written by the `NumElements` postprocessor. It should grow by ~4× per refinement level.
- **Convergence rate estimate.** If peak stress at level N is σ_N, the observed rate can be estimated as `log2((σ_{N-1} - σ_∞) / (σ_N - σ_∞))` where σ_∞ is your reference (Kirsch). If you do not know σ_∞ accurately, use the finest level as a proxy.
- **Expectations versus references.** Grade the assigned refine-4 run against ≈2.48 (plane-strain von Mises) and ≈2.88 (hoop). The ≈2.67 and ≈3 lines are infinite-plate references, not refine-4 endpoints. Do not compare the von Mises series directly with 3. Localized nodal-max stresses converge slowly; inspect `nt` in the `AnnularMeshGenerator` block when explaining the continued rise.

---

## Academic integrity reminder

Agents are encouraged. Logged prompts are mandatory. The numerical results you submit must be those *your* run produced — copying a classmate's plot is plagiarism. Discussing strategy with classmates is fine and encouraged.

---

## What happens to this work later in the semester

Hang on to your `convergence.png` and your refined `plate_with_hole.i` files. You will see this same problem three more times:

- **Module 3 (Week 6).** You will solve the *same* plate-with-hole problem using a Physics-Informed Neural Network, anchored on Min Lin's DeepXDE notebook (`PINN_Example/2D-Hole-Fix-E-Nu/`). You will compare your PINN's peak stress to the MOOSE convergence study you produced today and to ABAQUS reference data Min Lin already has.
- **Module 5 (Week 10).** You will run *this same MOOSE input file* but with the `ComputeLinearElasticStress` material replaced by a learned PyTorch model — the first NN-augmented MOOSE simulation of the course.
- **Possibly your final project.** Several past students extend this same geometry into a parameter-identification (inverse) problem: given a coarse displacement field, what (E, ν) and what stress concentration produced it?

The point: Module 0 is not throwaway scaffolding. The artifact you produce today is your baseline reference for three later modules.
