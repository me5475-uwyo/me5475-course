# Module 0 setup checklist

Start these steps in Lecture 1. Complete the local Python/PyTorch setup before Lecture 2 and the MOOSE setup before Lecture 3.

## 1. Course accounts

- **ARCC:** your account has been requested through an instructor-submitted Project Change Request. Watch for an email from ARCC. Tell the instructor promptly if you registered late so your username can be added.
- **the course GitHub organization:** submit your GitHub username through the short "Your GitHub username" item on Canvas (due Wed Sep 2). The instructor then creates your private repository `me5475-<username>` in the course organization <https://github.com/me5475-uwyo> and GitHub emails you an invitation — accept it, then clone your repo. All labs are submitted as pull requests there.
- **GitHub Copilot:** verify student status at <https://education.github.com/pack> with your UW email. Approval can take a few days, so start today. Copilot is free after GitHub approves the student status.

## 2. Local Python 3.11 and PyTorch 2.x

Use conda or mamba so the course environment stays separate from your system Python:

```bash
conda create -n me5475 python=3.11 pip -y
conda activate me5475
python -m pip install --upgrade pip
python -m pip install torch
```

Verify the install:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.rand(2, 2))"
```

You should see Python 3.11, a PyTorch 2.x version, and a 2×2 tensor. If you need local NVIDIA GPU support, use the platform-specific command from <https://pytorch.org/get-started/locally/> instead of the final `pip install` command.

## 3. Local MOOSE before Lecture 3

Follow the lead-agent and independent-review workflow in `module_0/examples/install_moose_prompt.md`. The local MOOSE environment is named `moose`; keep it separate from `me5475`.

After installation:

```bash
conda activate moose
moose-opt --version
cd module_0/examples
moose-opt -i plate_with_hole.i
```

Open `plate_with_hole_out.e` in ParaView, color by `vonmises_stress`, and save the screenshot required by Lab 0. If the install still fails after two earnest attempts, document the failure and use the ARCC fallback described in `module_0/homework/lab_0.md`.

**On Windows:** MOOSE has no native Windows build, so a local install needs WSL (see the "Connecting to
ARCC" page on Canvas). Visualization is a different matter — **ParaView has a native Windows build** and
reads `.e` files directly. If you take the ARCC fallback for the install, you can still produce the Lab 0
screenshot: run the simulation on the cluster, download `plate_with_hole_out.e` (portal → Files → Home
Directory → Download, or `scp`), and open it in ParaView on your own machine.

## 4. ARCC values to keep

After your onboarding email arrives:

```bash
ssh <netid>@medicinebow.arcc.uwyo.edu
```

- Account: `me5475`
- CPU partition: `mb`
- GPU partitions: `mb-l40s,mb-a30` with `--gres=gpu:1`
- Course environment: `conda activate /project/me5475/envs/ml4sm`

Copy the complete module and library-path preambles from the supplied `.sbatch` files; do not hand-type them from memory.
