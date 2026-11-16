# Module 7 — Measured results

Every number quoted in M7 lectures, readings and Canvas pages comes from a run recorded here. Format follows
`module_0/readings/measured_results.md`.

## §1 · L31 DeepONet demo (1-D Burgers)

- **Code:** `module_7/examples/generate_burgers_data.py` (dealiasing fix, Nov 12, 2026) and
  `module_7/examples/deeponet_burgers.py` (true-epoch loop, Nov 12, 2026), defaults: `--n 1000 --N 128 --seed 42`,
  ν = 0.01, T = 1, dt = 1e-3; `--epochs 500 --batch-size 32 --n-query 64 --K 128 --hidden 128 --depth 4 --seed 42`.
- **ARCC job 25201399** (Nov 12, 2026, partition `mb`, node mbcpu-001, 4 CPUs, course environment via
  `setup5475.sh`): 1000 pairs written; data 92 s; training 111 s; whole job 3 min 30 s. **115,841 parameters.**
  **Test relative L² error 8.66 %**: the norm ratio ‖pred − true‖ / ‖true‖ over all 200 × 128 test entries (not a
  mean of per-function errors). Raw job output: `team/reviews/2026-11-12_L31_evidence/l31_demo-25201399.out`.
- **Student-path test: ARCC job 25217102** (Nov 13, 2026, partition `mb`, node mbcpu-019): `deeponet_demo.sbatch`
  (the student job script: same two commands and arguments as the instructor's) submitted from a folder holding only
  the three published files: COMPLETED 0:0 in 5 min 22 s; data 111 s, training 204 s; 115,841 parameters;
  **8.6562 %**, identical to job 25201399 (the CPU result reproduced to the printed digits on a different node).
- **Laptop check** (torch 2.8 CPU, same command): 1000 pairs in 76 s, training 46 s, test relative L² 8.64 %.
  With `--epochs 1000`: 5.46 % (98 s). With `--n-query 128`: 8.67 %. With `--lr 3e-3`: 27.0 %.
- **Solver check** (the generator's first six initial conditions with `--seed 42`, i.e. six successive `sample_grf`
  draws from `default_rng(42)`): the corrected N = 128 solver agrees with an N = 512, dt = 5e-5 run of the same solver
  to 6.6e-11 (relative L², the first condition) and to below 1e-5 on all six. The Reviewer's independent Cole–Hopf
  reference on 4,096 points gave 1.6e-11 to 8.7e-11 on the same six (`team/reviews/2026-11-12_reviewer_L31.md`); the uncorrected solver equaled the heat-equation solution
  (0.00 %) and differed from Burgers by 104–138 % (the first three of the six conditions).
- **Front width in the L31 slide-5 figure** (`Slides/beamer/make_figures_m7.py`; the generator's initial condition #17
  at seed 42, i.e. the 18th draw): at t = 1 the steepest drop is between x = 0.71875 and 0.7265625; across it u falls
  from 0.426 to −0.211; the 10–90 % width (distance between the points where u has fallen 10 % and 90 % of that drop,
  by linear interpolation) is **0.085, about 11 grid spacings** (Δx = 1/128). Reviewer-checked
  (`team/reviews/2026-11-13_reviewer_L31_R1_recheck.md`).
- **Before the fixes** (for the record; never taught): the old loop (500 single batches) on corrected data gave
  28.5 % in 6 s.

## §2 · Lab 7 (published values only; accuracy and timing comparisons are held until Lab 7 closes)

- **Burgers generator across grids** (Nov 13, 2026, laptop, after the dt stability guard): seconds per pair about
  0.07 (N = 64), 0.08 (N = 128), 0.37 (N = 256), 1.9 (N = 512). Against an N = 1024 reference solved with a stable
  step, three seed-42 initial conditions agree to 3–5e-6 (N = 64), 5–8e-11 (N = 128), 6e-14–3e-13 (N = 256) and
  ~2e-15 (N = 512). With the old fixed dt = 1e-3, every sample at N = 256 and 512 diverged. N = 128 output is
  bit-identical before and after the guard.
- **FNO (`fno_burgers.py`, defaults modes 16, width 64, 4 layers):** **549,569 real parameters** (each complex
  spectral weight counted as two real numbers, the convention used for the DeepONet comparison); 287,425 tensor
  entries (PyTorch `numel`, a complex weight counted once), the figure the Nov 13 handout test printed before the
  script reported both.
- **Stretch script (`fno_microstructure.py`, defaults modes 12, width 32, 4 layers, pad 8):** 2,368,259 real
  parameters (a complex weight counted twice). Test of the student path: ARCC job **25222519** (Nov 13, partition
  `mb`, node mbcpu-014) via `operator_train.sbatch fno-micro --data rve_dataset.npz --unet-ckpt unet_stress.pt` on the
  L30 demo's 40-cell field set (seed 9001, not a Lab 6 dataset): COMPLETED 0:0; split 28/6/6, the U-Net's; 200 epochs
  in 172 s on CPU; checkpoint, U-Net comparison and four-column figure written.
- **Lab 7 handout, run as written** (Nov 13, ARCC, a mock repository with the git lines skipped; Tasks 2–3 chained
  after Task 1 for the test): Task 1 data job 25222641, 1000 pairs, 99 s (1 min 50 s elapsed); Task 2 DeepONet job
  25222642, COMPLETED, 115,841 parameters, training 167 s, **8.6562 %** (the public L31 value, reproduced); Task 3
  FNO job 25222643, COMPLETED, 287,425 parameters (its error and time are held); Task 4a test sets, 100 pairs each,
  jobs 25222644–25222647: N = 64 8 s, N = 128 12 s, N = 256 49 s, N = 512 245 s. All seven jobs exit 0:0.
- Reference DeepONet/FNO accuracy, FNO training time and the cross-resolution results (ARCC job 25221081) are in
  `team/reviews/2026-11-13_M7_lab7_HELD_until_Nov23/` and are not to be quoted before Mon Nov 23, 11:59 PM MT.

## §3 · L32 heat-equation teaching example (Nov 15, 2026; analytical/numerical, not a trained model)

`python module_7/examples/heat_fourier_example.py` (NumPy only; local run, deterministic). u_t = ν u_xx on [0, 1)
periodic, ν = 0.01, u(x, 0) = sin 2πx + ½ sin 6πx, grid x_j = j/N.

- N = 64: `rfft` returns 33 coefficients; û_1 = −32i, û_3 = −16i (−iAN/2); amplitudes 2|û_k|/N = 1.000 and 0.500.
- Decay factors at t = 1: mode 1 e^(−4π²ν) = 0.6738; mode 3 e^(−36π²ν) = 0.0286. At t = 0.25, mode 3: 0.4114.
- FFT solution vs the hand solution at t = 1: max |error| 3.9e-16, relative L2 2.5e-16 (N = 64); 3.9e-16 and 2.3e-16
  (N = 256, same amplitudes 1.000 and 0.500). At t = 0.25, N = 64: 6.1e-16 and 2.8e-16.
- Figures for L32 slides 7 and 9 (slides 4 and 6 before the Nov 16 FFT slides): `--figdir Slides/beamer/figs` writes `l32_heat_waves.pdf`, `l32_heat_solution.pdf`
  (and `l32_heat_spectrum.pdf`, not used in the deck).

## §4 · FFT basics examples (Nov 16, 2026; NumPy, not a trained model)

`python module_7/examples/fft_basics_examples.py` (all five examples; `--example N` runs one). Local demo environment
`~/ME5475-demo-env` (Python 3.9.6, NumPy 2.0.2), laptop, Nov 16, 2026; deterministic (noise from
`default_rng(0)`). Run time 0.07 s without figures, 0.9 s with `--figdir` (five PDFs, written to a scratch folder,
not the repo). NumPy default normalization, grid x_j = j/N. Cited by `module_7/readings/fft_primer.md`.

- **Example 1** (u = 2 sin 2πx + 0.5 cos 10πx, N = 64): `rfft` returns 33 coefficients, nonzero at k = 1 and 5.
  k = 1: û = −64i, amplitude 2|û|/N = 2.0000, angle −90° (sine). k = 5: û = 16, amplitude 0.5000, angle 0° (cosine).
- **Example 2** (m = 1 kg, k = 400 N/m, ζ = 0.02; fs = 100 Hz, T = 20 s): ω_n = 20.0 rad/s, f_n = 3.1831 Hz, exact
  f_d = 3.1825 Hz; N = 2000 samples, 1001 bins, Nyquist 50 Hz; bin spacing 1/T = 0.050 Hz; largest bin (mean removed)
  3.200 Hz (bin 64), peak − f_d = +0.0175 Hz (this run: within half a bin, 0.025 Hz; bin spacing is not an error bound); envelope factor at t = T: 3.4e-04.
- **Example 3** (sin 2πx + 0.3 sin 6πx + noise σ = 0.05, seed 0, N = 256; keep k = 0..10, 11 of 129 bins): RMS
  error vs the clean signal 0.0506 before, 0.0155 after.
- **Example 4** (u = sin 2πx + 0.5 sin 6πx, N = 32; max |u′| = 15.71): max |error| vs exact u′: spectral (iκ_k,
  Nyquist zeroed) 4.6e-14; periodic central difference 0.576.
- **Example 5** (periodic 5-point moving average of the Example-3 signal, N = 256): max |direct − via FFT| = 6.7e-16.
  Operation counts printed for a kernel as long as the signal: N² = 65536; N log₂ N = 2048 per transform.
