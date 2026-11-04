# ME 5475 — Machine Learning for Computational Solid Mechanics

**University of Wyoming · Fall 2026 · MWF 1:10–2:00 PM · Engineering Building 2070**
Instructor: Prof. Xiang Zhang · <xiang.zhang@uwyo.edu> · EERB 335B · office hours MWF 2:00–3:00 PM

This repository is your working copy of the course materials **and** where you submit your work.
The syllabus, deadlines, and announcements live on Canvas — Canvas is authoritative for dates.

---

## Getting started

1. **Set up your machine** — see [`module_0/setup.md`](module_0/setup.md): Python 3.11+, PyTorch 2.x, an editor, and an AI coding agent.
2. **Test your ARCC login.** Accounts are created for you under the course project `me5475`:
   ```bash
   ssh <your-username>@medicinebow.arcc.uwyo.edu
   ```
   If it fails, tell the instructor — no deadline penalizes you for access you don't have yet.
3. **Read the current lab** in `module_<n>/homework/`, and work in `labs/lab_<n>/<your-handle>/`.

## Layout

```
module_<n>/
├── README.md      what the module covers
├── examples/      runnable code and MOOSE input files the lectures use
├── readings/      references and cheat sheets
└── homework/      the lab handout and starter prompts
labs/lab_<n>/<your-handle>/   ← your submitted work
```

Modules are added as the semester progresses (see *Staying up to date* below).

## How to submit

Every lab is a **pull request from a branch named `lab_<n>`** in this repository.

```bash
git checkout -b lab_0
mkdir -p labs/lab_0/<your-handle>
# ... do the work ...
git add labs/lab_0/<your-handle>
git commit -m "Lab 0"
git push -u origin lab_0
```
Then open a pull request on GitHub. Merge after instructor sign-off.

**Every submission includes an AI-use log** (`prompts.md`, or the per-task files a lab specifies):
which tool and model, approximate dates, your lead and review prompts, and which outputs you
used. Paraphrasing is fine. If you used no AI on an assignment, one line saying so is a
complete log. Full policy: syllabus §8.

## Staying up to date

New material is published during the semester, and it arrives on **your repository's `main` on GitHub**: the course
updates every student repository for you. You only need to pull it, in each clone you use (laptop and ARCC):

```bash
git status                                          # first: commit the lab files you want to keep
git checkout main
git pull
git checkout lab_5 && git merge --no-edit main      # only to bring it into a lab branch you already started
```

In these commands, `lab_5` and `lab_6` are examples: type your own lab's number. Start a new lab from an updated
`main`: `git checkout main && git pull && git checkout -b lab_6`.
Why a new file can seem "missing", and more: `module_0/readings/git_basics_and_course_repo_primer.md`.

You don't need an `upstream` remote. If you added one earlier, don't use it: `git pull upstream main` can stop with
"refusing to merge unrelated histories". Don't force it with `--allow-unrelated-histories`; use `git pull` on `main`.

## Where to ask

**Canvas Discussions** for anything between classes — installs, cluster problems, concepts.
Post rather than struggle alone; classmates usually hit the same wall. Office hours are MWF
2:00–3:00 PM right after class. There is no TA this semester.
