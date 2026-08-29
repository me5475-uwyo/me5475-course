# Lecture 2 — AI Agents and the 2-Agent Lead+Review Workflow

**Date:** Wednesday, September 2, 2026
**Module:** 0 — Setup, Agentic Workflow, ARCC + MOOSE Orientation
**Duration:** 50 minutes
**Format:** Lecture (30 min) + in-class hands-on agent lab (20 min)

---

## Learning objectives

By the end of this lecture, students should be able to:

1. Define an AI coding agent as an LLM coupled to tools and a control loop, and distinguish this from a one-shot prompt.
2. Describe the 4-agent framework (theory-lead / theory-review / code-implementer / code-review) and explain the rationale for collapsing to a 2-agent *lead + review* version for routine work.
3. Apply lead + review to (a) a theoretical derivation and (b) a code-implementation task in a mechanics context.
4. Recognize three common failure modes of agentic workflows — hallucinated APIs, plausible-but-wrong derivations, runaway tool loops — and the disciplines that catch them.
5. Configure an authenticated AI coding agent on their laptop and complete a worked exercise.

---

## Mental model — what is an agent? (5 min)

A *one-shot prompt* is what you do in a chat window: you write a message, the model writes one back. A *coding agent* is the same model wrapped in a loop with tools. The agent receives an instruction, decides which tool to call (read a file, run a command, edit code), reads the output, and decides what to do next. The loop terminates when the agent declares the task complete or the user intervenes.

Why this matters: agents fix errors. The most common failure mode of LLMs in 2024 was code that looked right and ran wrong. With a tool loop, the agent can run the code, see the error, and try again — without a human re-prompting between every iteration. That changes what we can ask them to do.

The **Model Context Protocol (MCP)** is an open standard for plugging external tools into agents. You will not write MCP servers in this course, but you may use agents that connect to external services through MCP. File, shell, and edit tools built into an agent harness are native tools, not necessarily MCP calls.

## The 4-agent framework, and why we collapse it (5 min)

In your instructor's research workflow, the full pattern uses four agents:

| Role | What it does |
|------|--------------|
| **Theory lead** | Derives equations, proposes formulations, writes proofs |
| **Theory review** | Checks dimensional consistency, BC handling, sign conventions, physical reasonableness |
| **Code implementer** | Turns the validated formulation into runnable PyTorch / MOOSE input / SLURM scripts |
| **Code reviewer** | Checks correctness, efficiency, reproducibility, edge cases |

The pattern works because each role is *independent of the others*. A reviewer with no memory of how the lead arrived at the answer is harder to fool than the lead checking its own work.

For routine homework in this course we use the **2-agent lead + review** simplification. One agent does the work (whether that work is derivation or implementation); a second agent, with no shared context, reviews it. You — the student — are the human-in-the-loop arbiter.

You will see the full 4-agent extension in the final project. For everything else, lead + review is enough.

## Worked example A — Theory: derive the weak form of linear elasticity (6 min)

The setup we will use as the running example throughout Module 3:

- Domain Ω ⊂ ℝ², boundary Γ = Γ_D ∪ Γ_N
- Strong form: ∇·σ + b = 0 on Ω, u = ū on Γ_D, σ·n = t̄ on Γ_N
- Linear elastic, small strain: σ = ℂ:ε, ε = ½(∇u + ∇uᵀ)

**Lead prompt** (paste into Claude Code or Cursor as a fresh session):

```
Derive the weak form of the equations of linear elastostatics on a 2-D domain Ω
with displacement BC u = ū on Γ_D and traction BC σ·n = t̄ on Γ_N. Assume small
strain and linear elasticity σ = ℂ:ε. Show every step:

1. State the strong form.
2. Multiply by an admissible test function v that vanishes on Γ_D.
3. Integrate over Ω.
4. Apply integration by parts (divergence theorem) on the σ term.
5. Substitute the Neumann BC into the boundary integral.
6. State the final weak form clearly, identifying the bilinear form a(u, v)
   and the linear form ℓ(v).
```

**Review prompt** (fresh session — do not paste the lead's output yet; have the review read it from a file or fresh paste):

```
Read the following derivation and check the following items independently:

1. Dimensional consistency at every step.
2. That the test function v vanishes on Γ_D (and that this is used correctly when
   the boundary integral over Γ_D drops out).
3. That the surface integral over Γ_N matches the traction BC t̄.
4. Sign conventions throughout (especially after integration by parts).
5. Whether the final bilinear form a(u, v) is symmetric.

Reply with PASS / FAIL on each item, plus one sentence of justification per item.
If FAIL, explain what the correct version should be. Do not rewrite the entire
derivation — just point to the specific step.

Derivation to review:
<paste lead's output here>
```

**What to discuss in class.**

- What did the review catch? In a typical run the review will spot a sign error in the integration by parts, or the lead writing ∇·(ℂ:∇u) when it should be ∇·σ.
- What did neither agent catch? Often a missing assumption about the regularity of v (must be in H¹₀(Ω; Γ_D)) — the agents will sometimes wave at this. The human notices.
- Why is the review more trustworthy than the lead self-checking? Because the lead has cognitive momentum — it wants its derivation to be right. A fresh-session review has no investment.

## Worked example B — Code: implement a 1-D MLP that fits y = sin(πx) (6 min)

A simple regression problem to anchor the code-lead / code-review pattern.

**Lead prompt** (fresh session):

```
Write a PyTorch implementation of a 4-layer MLP with width 32 and tanh activations
that fits the function y = sin(πx) on [-1, 1]. Specifications:

- Generate 200 training points uniformly on [-1, 1] with no noise.
- Use Adam optimizer, learning rate 1e-3, 5000 epochs, full-batch (not mini-batch).
- Loss is MSE.
- After training, plot the network prediction against the true function on a
  dense grid of 1000 points, overlay the training data.
- Print the final training loss.

Return a single Python file that runs end-to-end.
```

**Review prompt** (fresh session):

```
Read the following PyTorch code and check the following items independently:

1. Data tensor shapes — does the input have shape (N, 1) as PyTorch expects?
2. Training mode handling — is .train() called appropriately? Does .eval() get
   called before the plotting phase?
3. Loss correctness — does the MSE include all training points, no missing
   factor, no sign error?
4. Gradient flow — does the code zero gradients, call backward, step the
   optimizer, in the right order?
5. Hyperparameter sanity — width 32 + 4 layers + tanh is enough capacity for
   sin(πx); does the learning rate seem reasonable?
6. Reproducibility — is the random seed set?

Reply with PASS / FAIL per item with one sentence of justification.

Code to review:
<paste lead's output here>
```

**Live demo in class.** Instructor runs both prompts via Claude Code. The expected outcomes vary by model release, but typical findings:

- The lead sometimes forgets `optimizer.zero_grad()` or puts the `loss.backward()` after `optimizer.step()`.
- The lead almost always omits `torch.manual_seed(...)`.
- The review almost always catches both.
- A subtler bug that sometimes appears: the lead computes loss on a tensor of shape (200,) instead of (200, 1), broadcasting silently to a (200, 200) MSE that still produces a scalar. The review catches this when prompted on shapes.

## Failure modes and the disciplines that catch them (5 min)

The most important failure mode in this course is not generic "the agent hallucinated an API." It is *the agent producing a mechanics-correct-looking equation that is wrong in a way only a mechanician would spot*. We'll work one example end-to-end so you see what to listen for.

**The case.** A student asks the lead agent: *"Derive the stiffness matrix used for 2-D linear isotropic elasticity in plane strain, in Voigt notation."*

A typical lead-agent response will produce something like:

```
For plane strain (ε_zz = 0), the stress-strain relation in Voigt notation is
σ = C ε with

    C = E/((1+ν)(1-2ν)) · [[1-ν,   ν,     0     ],
                            [ν,     1-ν,   0     ],
                            [0,     0,     (1-2ν)/2]]
```

That happens to be correct. But agents in 2024 frequently produce instead the *plane-stress* stiffness:

```
    C = E/(1-ν²) · [[1,   ν,   0      ],
                    [ν,   1,   0      ],
                    [0,   0,   (1-ν)/2]]
```

…labeled as "plane strain." A reviewer who does not know mechanics will see Voigt notation, a 3×3 matrix, the right symmetries — and *pass* the answer.

The discipline that catches this: a review prompt that asks for *physical limit checks* the agent must actually evaluate.

**Review prompt for this case:**

```
Read the derivation below. Verify independently:

1. Is this plane stress or plane strain? Apply the limit ν → 0.5 (incompressible).
   - For plane strain, the matrix should DIVERGE (because (1-2ν) → 0 in the
     denominator), which is physically correct (incompressible solid resists
     volumetric strain infinitely under plane-strain constraint).
   - For plane stress, the matrix should remain finite at ν → 0.5.
2. Does the leading coefficient match the stated formulation?
3. Is the trace of the upper 2x2 block consistent with a known special case
   (e.g., ν = 0 should give an identity-times-E in the diagonal entries
   for plane stress, but NOT for plane strain)?

PASS / FAIL each item.

Derivation:
<paste lead output>
```

The ν → 0.5 limit check is the magic. Plane strain *must* diverge (denominator goes to zero); plane stress *must not*. The reviewer plugs the limit in and the wrong answer becomes obvious. The lead can't fool a reviewer that does limit checks.

**The general taxonomy of failure modes, with this case as the anchor.**

- **Hallucinated APIs.** Agent calls a function that doesn't exist. Discipline: *always run the code*; never accept syntax-only confidence.
- **Plausible-but-wrong mechanics derivations** — like the plane-strain/plane-stress mix-up above. Discipline: review prompts include *physical limit checks* (incompressibility, zero-load, rigid-body), *dimensional analysis*, and *symmetry checks*. The reviewer must do work, not just nod.
- **Runaway tool loops.** Agent retries the same failing action 20 times. Discipline: set a hard step/turn limit (for example, `--max-turns`) plus reading the trace before accepting the result.
- **The lead writing what it thinks the reviewer wants to see.** Subtle and increasingly important. Discipline: keep lead and review in *independent* sessions with no shared scratchpad. Only the human passes artifacts between them.
- **The reviewer rubber-stamping.** If review prompts are vague ("check this"), a fluent agent will PASS everything. Discipline: review prompts list specific testable items the reviewer must individually answer, not a vague request to "review."

## Course policy on agent use (2 min)

- **Agents are encouraged everywhere except exams** (there are no traditional exams in this course; the closest is the in-class portion of the midterm project critique).
- **Every submitted artifact includes a prompt log** of the lead and review prompts used. The assignment supplies the filename (`prompts_l2.md` for Lab 0's in-class work). Logs need not be word-for-word transcripts — paraphrased prompts are fine — but every distinct prompt-cycle must be captured.
- **One paragraph of reflection per major artifact** stating what the review caught and what you (the human) caught.
- **Submitting unreviewed agent output is treated like submitting unproofread work** — points off.
- **The scientific judgment is yours.** If your lead derived an equation that your review approved and you turn it in, and it is wrong, the responsibility is yours — not the agent's. This is the same standard you would apply to a co-author.

See **syllabus §8** for the full AI-use policy.

---

## In-class hands-on lab (last 20 min)

Each student opens their preferred authenticated AI coding agent (ChatGPT/Codex, Copilot, Claude Code, Cursor, or equivalent) and runs the two worked examples above using the 2-agent lead + review pattern. *Two separate sessions for each example — do not share context between lead and review.*

**Submit to GitHub before leaving the room.** Work on branch `lab_0` and commit to `labs/lab_0/<your-github-handle>/in_class/`:

- `weak_form_derivation_lead.md` — output of the lead agent for Example A
- `weak_form_derivation_review.md` — output of the review agent for Example A
- `mlp_sine_lead.py` — output of the lead agent for Example B
- `mlp_sine_review.md` — output of the review agent for Example B
- `prompts_l2.md` — the lead and review prompts used for both examples

This is the in-class portion of Lab 0. The takehome portion is the MOOSE/ARCC convergence study (see `homework/lab_0.md`).

---

## Assigned reading and homework (before Lecture 3)

**Required homework before L3.** This is the install task; doing it as homework rather than in-class freed Lecture 3 of its biggest live-failure risk:

1. **Install MOOSE locally with an agent.** Use the prompt in `module_0/examples/install_moose_prompt.md` (it is the same agent-driven install prompt we would have run in class). Then run an independent review prompt to confirm install. Commit `install_log.txt` to your Lab 0 directory.
2. **Run `plate_with_hole.i` once.** Default `uniform_refine = 0`. Commit a screenshot of the ParaView von Mises field.
3. **Verify ARCC account.** SSH in and confirm you reach a login node. Email the instructor at least 24 hours before L3 if your account is not active.

If the install fails after two earnest attempts, come to L3 anyway — the first 5 minutes are scheduled for troubleshooting, and the homework can be completed on ARCC.

**Primary reading.**

- Anthropic, *Building effective agents* (anthropic.com/engineering, December 2024). The canonical introduction to agent design patterns.
- Anthropic, *How we built our multi-agent research system* (anthropic.com/engineering, June 2025). A production multi-agent system to compare with our 4-agent framework in Module 10.

**Optional.**

- Schick et al., *Toolformer: Language models can teach themselves to use tools*, NeurIPS 2023. The academic precursor to tool-using LLMs.
- Model Context Protocol specification, modelcontextprotocol.io. For students who want to write their own tools later.

**Reference for the 4-agent framework used in this course.**

The 2-agent *lead + review* pattern introduced today is a deliberate simplification of a larger 4-agent workflow developed in the instructor's group for verified scientific software development in MOOSE. The full workflow uses four AI agents under fixed roles — a Theory Lead (formulation), a Theory Reviewer (independent scientific review), a Code Implementer (execution), and a Code Reviewer (independent technical review) — coordinated by a human PI who routes artifacts and controls all verification gates. Models from three provider families (Anthropic Claude, OpenAI GPT, OpenAI Codex) are assigned to roles based on demonstrated strengths, providing cross-family blind-spot coverage. A contract-driven campaign using this workflow passed all 13 pre-specified verification gates in the first formal cycle, producing a verified FE² computational homogenization framework from scratch — the empirical motivation for the agent emphasis in this course.

- Zhang, Arif, Bist, Hasaninia & Sun, *Human-Orchestrated Multi-Agent Development of Verified Multiscale Constitutive Infrastructure in MOOSE: Workflow, Analysis, and Evaluation Protocol* (manuscript in preparation, 2026). A PDF will be distributed via Canvas at the start of the semester if not yet published.

We return to the full 4-agent workflow in Module 10. Modules 0–9 use the 2-agent simplification so students master the discipline of independent review before scaling to four roles.

---

## Instructor notes (not for student view)

- 30 minutes for the lecture portion is generous. If running long, compress the worked examples to 6 minutes each (skip the discussion in Example A) and use the time gained for the in-class lab.
- The in-class lab assumes each student's preferred AI coding agent is installed and authenticated. Send a pre-class email reminding students to have this working before they walk in. If 2–3 students fail to get authenticated, pair them with classmates rather than holding up the room.
- The "two separate sessions" instruction is the single most important discipline of the course. Spend the extra 30 seconds to emphasize it.
