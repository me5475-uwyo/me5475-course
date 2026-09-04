# Submitting Your Work with Git and GitHub

*A step-by-step guide for every lab in this course. If you have never used Git, start here and follow it literally — you do not need to understand Git to use it correctly this week, and the understanding comes with repetition.*

Every lab is submitted the same way, so this page is written once and referenced from every handout. Substitute the lab number and it works for all nine.

---

## What you are doing, in one paragraph

Your work lives in a **repository** — a folder whose history GitHub keeps. Yours is private: only you and I can see it. You copy it to your computer (**clone**), do the work on a separate **branch** so the main copy stays clean, save snapshots as you go (**commit**), send them to GitHub (**push**), and finally open a **pull request** — a formal "here is my work" that carries a timestamp. The pull request *is* the submission.

---

## Step 0 · One-time setup (do this once, ever)

**Install Git.** macOS: `git --version` in Terminal will offer to install it. Windows: install [Git for Windows](https://git-scm.com/download/win), or use WSL. Linux: your package manager.

**Tell Git who you are** — this is stamped on every commit:

    git config --global user.name "Your Name"
    git config --global user.email "you@uwyo.edu"

**Set up authentication.** This is the step that stops most people, so read it carefully: **GitHub does not accept your account password from the command line.** Typing your password into a `git push` prompt will fail no matter how correct it is. Pick one of these instead:

- **GitHub CLI — simplest.** Install [cli.github.com](https://cli.github.com/), then run `gh auth login` and follow the browser prompts. It configures Git for you and you never think about it again. *This is what I recommend.*
- **GitHub Desktop.** [desktop.github.com](https://desktop.github.com/) — sign in once in the app, and it handles authentication for everything you do through it.
- **VS Code.** Sign in via the Accounts icon at the bottom-left; VS Code then authenticates Git operations for you.
- **Personal access token.** If you insist on plain HTTPS: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained → generate one with repository access, and paste *that* when Git asks for a password.

---

## Step 1 · Accept the repository invitation

Your repository is private, so **it is invisible to you until you accept**. Go to **https://github.com/notifications**, or open GitHub's invitation email, or go directly to:

    https://github.com/me5475-uwyo/me5475-<your-username>/invitations

and click **Accept invitation**. If it says the invitation is for another account, you are signed in as the wrong one.

---

## Step 2 · Clone it to your computer

    git clone https://github.com/me5475-uwyo/me5475-<your-username>.git
    cd me5475-<your-username>

*GitHub Desktop:* File → Clone repository → GitHub.com → select your repo → Clone.
*VS Code:* press F1, type `Git: Clone`, paste the URL.

You should now see `module_0/` and `module_1/` with the handouts, readings and example files already in them.

---

## Step 3 · Make a branch for this lab

    git checkout -b lab_0

That creates a branch named `lab_0` and switches you onto it. Check with `git branch` — the asterisk marks where you are. One branch per lab: `lab_0`, `lab_1`, and so on.

*GitHub Desktop:* Current Branch → New Branch → name it `lab_0`.

---

## Step 4 · Put your work in the right folder

    mkdir -p labs/lab_0/<your-username>

Everything you produce goes inside, in the structure the handout shows. **The folder names are how I find your work**, so follow the tree in the handout exactly.

---

## Step 5 · Commit and push, repeatedly

A commit is a labelled snapshot. Make several as you go — not one at the end.

    git add labs/lab_0/<your-username>
    git commit -m "Lab 0: convergence study results and plot"
    git push -u origin lab_0

After the first push, later ones are just `git push`. Check what you are about to commit with `git status` first; it tells you which files are new or changed.

*GitHub Desktop:* changed files appear on the left. Tick them, write a summary, **Commit to lab_0**, then **Push origin**.

Good commit messages say what changed: `"Lab 0: add refine 0-4 convergence data"` beats `"update"`. You will read these later.

---

## Step 6 · Open the pull request

Visit your repository on GitHub. A banner offers **Compare & pull request** — click it. Check that **base is `main`** and **compare is `lab_0`**, title it `Lab 0 — Your Name`, and click **Create pull request**.

**That pull request is your submission**, and its timestamp is what I grade against.

**Open it early.** Pushing more commits to `lab_0` updates the same pull request automatically, so there is no cost to opening it when you are half done — and it means a network problem at 11:55 PM cannot cost you the deadline.

---

## When something goes wrong

**`git push` asks for a password and rejects it.** Expected — see Step 0. Run `gh auth login`, or use GitHub Desktop.

**"Permission denied" or "repository not found" on clone.** Either you have not accepted the invitation (Step 1), or you are signed in as a different GitHub account.

**You did the work on `main` by mistake.** Nothing is lost:

    git checkout -b lab_0
    git push -u origin lab_0

Your commits come with you onto the new branch.

**You committed something huge** — an Exodus file, a checkpoint. Commit only what the handout asks for. If a push is rejected for file size, remove the file, then `git rm --cached <file>` and commit again.

**Genuinely stuck.** Post on Discussions with the exact error text. Git's messages are cryptic but consistent, so someone else has almost certainly seen yours. This is also a perfectly good thing to hand to your AI agent — paste the command and the error, and ask what it means.

---

## The short version, once you have done it once

    git checkout main && git pull          # start from current material
    git checkout -b lab_N                  # branch for this lab
    # ... do the work in labs/lab_N/<your-username>/ ...
    git add labs/lab_N/<your-username>
    git commit -m "Lab N: what you did"
    git push -u origin lab_N
    # open the pull request on github.com

That is the whole loop, every lab, all semester.
