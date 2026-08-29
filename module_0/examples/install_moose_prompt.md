# Install-MOOSE Prompt for the Pre-L3 Homework

**Use:** After Lecture 2 and before Lecture 3. Open your AI coding agent (ChatGPT/Codex, Copilot, Claude Code, Cursor, or equivalent) and paste the prompt below. Replace the bracketed OS line with your laptop's OS.

---

## Prompt

```
I want to install MOOSE (the Idaho National Lab finite element framework) on my
laptop using the conda-based installation path that INL officially supports.

Specifications:
- Use mamba if available, conda otherwise.
- Create a new environment named 'moose' with Python 3.11.
- Install MOOSE from the INL conda channel as described at
  https://mooseframework.inl.gov/getting_started/installation/conda.html
- After install, verify by running 'moose-opt --version' (or the equivalent
  in the installed environment) and report the version string.
- If install fails, diagnose the error and propose a fix before retrying.
  Do not retry blindly more than twice. If two retries fail, stop and ask me.
- Report each step you take so I can follow along.

My OS is: [pick one: macOS arm64 / macOS x86_64 / Linux x86_64 / WSL2 Ubuntu]
```

---

## Notes for the student

- **Watch the agent's first action.** It should call `mamba --version` or `conda --version` to check what is installed. If it tries to call `apt-get install` or `brew install` *before* checking, interrupt and remind it that you already have conda.
- **Be patient at the download step.** The MOOSE binaries are ~3 GB; expect 5–10 minutes on a typical campus connection.
- **If the agent gets stuck.** Read its last error message yourself. Most install failures are version conflicts that a `mamba install --strict-channel-priority` or a fresh environment can fix. If you cannot resolve it with two more agent prompts, switch to using MOOSE on ARCC for the homework.

## Review prompt (run separately after install reports success)

```
Read the install log from the previous session (pasted below). Independently
verify the following items:

1. The environment 'moose' actually exists. (Check 'conda env list'.)
2. moose-opt is on PATH when the environment is active.
3. moose-opt --version prints a reasonable version string.
4. The Python in the environment is 3.11.x.
5. There were no error or warning messages that the install agent silently ignored.

Reply PASS / FAIL per item with one sentence of justification.

Install log:
<paste the install agent's full output here>
```
