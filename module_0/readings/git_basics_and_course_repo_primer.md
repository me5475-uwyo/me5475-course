# Git basics and the course repository (optional)

*Optional reading, about fifteen minutes. Part 1 is git in plain words, with a short section on branches. Part 2 is
how your course repository is laid out and how new course material reaches you. The step-by-step submission recipe
stays in `module_0/readings/git_submission_guide.md`.*

---

# Part 1 · Git basics

*In every command below, replace `<your-github-username>`, `<file>` and similar placeholders with your own values,
and remove the angle brackets. If you paste a block and any command fails, stop and read the message before going
on.*

## Five words

- **Repository (repo).** A folder whose history git keeps. Yours is `me5475-<your-github-username>` on GitHub.
- **Clone.** A full copy of the repository on a computer. You may have two: one on your laptop and one in your ARCC
  home folder. Each is separate; each needs its own `git pull`.
- **Commit.** A saved snapshot of the files, with a message. Commits are local until you push them.
- **Push / pull.** `git push` sends your new commits to GitHub; `git pull` brings GitHub's new commits to this clone.
- **Branch.** A named line of commits (below).

## The everyday loop

```bash
git status                      # where am I, and what changed?
git add labs/lab_5/<your-github-username>
git commit -m "Lab 5: exported and verified"
git push                        # first time on a new branch: git push -u origin lab_5
git log --oneline | head        # the last few snapshots
```

`git status` is the command to type whenever you are unsure. Its first line says which branch you are on.

## Branches, briefly

A branch is a **line of snapshots**. Making a branch copies the line you are on **at that moment**; after that, the two
lines grow separately. A branch does **not** follow its parent by itself.

```
main:   ──●────────●────────●──▶      (new commits keep arriving here)
           \
lab_4:      ●──●──●                    (made from main at the first ●; it has none of main's later commits)
```

- `git checkout <branch>` (or `git switch <branch>`) changes which line your folder shows. Committed files that exist
  only on the other line disappear from the folder and come back when you switch back; nothing committed is deleted.
  **Uncommitted** changes are different: git may carry them along to the other branch, or refuse to switch. So run
  `git status` and commit your lab files before you switch.
- `git merge main`, run while on a lab branch, brings `main`'s new commits into its history.
- `HEAD` is git's name for "where you are now". It is not a branch to move to.

---

# Part 2 · The course repository

## What is in it

```
me5475-<your-github-username>/
├── README.md                  getting started, how to submit
├── .gitignore
├── module_0/ … module_5/      course material; more modules arrive during the semester
│   ├── README.md              what the module covers
│   ├── examples/              runnable code, MOOSE inputs, job scripts (e.g. module_5/examples/run_nn_simulation.sbatch)
│   ├── readings/              guides, references, measured results
│   └── homework/              the lab handout (lab_5.md) and starter prompts
└── labs/
    ├── lab_0/README.md …      some labs include a course README
    └── lab_5/<your-github-username>/    ← YOUR work; nobody else writes here
```

Create your own submission folder as the lab handout says. Two kinds of files, in two places: **course material** in `module_*/` (the course writes it; read it, copy from it,
don't edit it), and **your work** in `labs/lab_N/<your-github-username>/`. Keeping them apart is what makes updates
painless.

## Two kinds of branches

- **`main` — where course updates arrive.** When material is released, the course adds it to **your repository's
  `main` on GitHub**. Do lab work on a lab branch, not on `main`.
- **`lab_0`, `lab_1`, … — your lines, one per lab.** You make each from `main`, do the lab there, push it, and open a
  pull request into `main`. That pull request is your submission; merge it only after instructor sign-off.

## How new material reaches you

```
course release ──▶ your main on GitHub ──(git pull, on main)──▶ your clone's main ──(git merge main)──▶ your lab branch
```

It stops at GitHub until **you** pull it, in **each** clone (laptop and ARCC), **on `main`**.

## What happened this semester

```
GitHub main:   ──●──────────────●──────────────────●───────────▶
               Oct 14           Oct 27             Nov 4
               Lab 4 released   Lab 5 released     a commented job script added
                  │
your lab_4:       └──●──●──●   (your Lab 4 commits; no module_5/)
```

A `lab_4` made around Oct 14 shows the course as it was that day. On Oct 27 `module_5/` arrived on `main`. Someone
still on `lab_4` who runs `git pull` gets only `lab_4`'s updates (or "no tracking information" if it was never pushed),
so `module_5/` seems missing. Nothing is broken; it is the wrong branch.

## The recipes

**Before any of these:** run `git status`. Commit the lab files you mean to keep on your lab branch
(`git add labs/lab_5/<your-github-username>/<file>`, then `git commit`), and deal with anything else it lists, so the
working tree is clean ("nothing to commit") before you switch branches or merge.

**Get the latest course material:**
```bash
git checkout main
git pull
ls module_5/examples            # check
```

**Start the next lab** (always from an up-to-date `main`):
```bash
git checkout main && git pull
git checkout -b lab_6
```

**Bring new course files into a lab you already started:**
```bash
git checkout main && git pull
git checkout lab_5
git merge --no-edit main        # --no-edit: no editor pops up for the merge message
git push                        # your pull request still shows only your own changes
```

## When git refuses

- **"Your local changes … would be overwritten by checkout."** You have uncommitted edits. Run `git status`, add and
  commit the lab files you mean to keep on your lab branch (not generated outputs such as `.e` or `.log` files), then
  switch.
- **A merge conflict.** Usually a course file was edited on your lab branch. Resolve it before continuing; if you're
  unsure, `git merge --abort` returns you to where you were before the merge (reliably so when the working tree was
  clean, as the recipes require), and then ask.
- **You committed on `main` by mistake.** See "When something goes wrong" in the submission guide; nothing is lost.

## The one line to remember

**New course files arrive on `main`: switch to `main`, pull, then branch or merge from it.**
