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
   If it fails, tell the instructor — no deadline penalises you for access you don't have yet.
3. **Read the current lab** in `module_<n>/homework/`, and work in `labs/lab_<n>/<your-handle>/`.

## Layout

```
module_<n>/
├── README.md      what the module covers
├── lectures/      the full lecture notes (more detail than the slides)
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

New modules are published to the course repository during the semester. Add it once as an
`upstream` remote, then pull when a module is released:

```bash
git remote add upstream https://github.com/me5475-uwyo/me5475-course.git   # once
git checkout main
git pull upstream main
```

## Where to ask

**Canvas Discussions** for anything between classes — installs, cluster problems, concepts.
Post rather than struggle alone; classmates usually hit the same wall. Office hours are MWF
2:00–3:00 PM right after class. There is no TA this semester.
