# Starter Prompts for Lab 7

Lead and review in **separate sessions, no shared context**. Logged prompts mandatory.

---

## Task 4 — Cross-resolution test setup

The hardest task in Lab 7 — testing both architectures fairly across resolutions.

**Setup notes.** Both training scripts, and this test, are CPU-only. On ARCC, jobs use
`source /project/me5475/setup5475.sh` on CPU partition `mb`: `burgers_data.sbatch` makes data at any N (for example
`sbatch burgers_data.sbatch --n 100 --N 256 --seed 1256 --out test_N256.npz`), and
`operator_train.sbatch deeponet|fno` trains. The checkpoints store the weights (`model_state`) and the training
arguments (`args`) as plain values, so a plain `torch.load(path, map_location="cpu")` reads them.

**Lead prompt:**

```
I need to test DeepONet and FNO trained on N=128 1-D Burgers data at
resolutions N in {64, 128, 256, 512}.

Test data: for each N, generate 100 NEW (u_0, u_T) pairs with
generate_burgers_data.py --n 100 --N <N> --seed <1000 + N>, keeping nu,
T and --length-scale the same as training. Do not use seed 42: at N=128
it reproduces the training pairs.

For DeepONet: the branch network expects u_0 at the 128 training sensors
x_i = i/128. To evaluate at N != 128, resample u_0 from the N-point grid
x_j = j/N to those 128 sensors with numpy.interp(..., period=1.0) (the
domain is periodic). The trunk takes a position y in [0, 1), so query it
directly at x_j = j/N on the test grid -- the output is NOT resampled.

For FNO: the architecture is designed to accept any N (its weights are
indexed by Fourier mode; it needs N//2 + 1 >= modes). Feed it u_0 on the
N-point grid directly.

Write test_resolutions.py that:
1. Imports DeepONet from deeponet_burgers.py and FNO1d from fno_burgers.py,
   loads deeponet_burgers.pt and fno_burgers.pt with torch.load(..., map_location="cpu"),
   and rebuilds each model from the checkpoint's saved args (ckpt["args"]:
   K, hidden, depth for DeepONet, with m_sensors=128; modes, width,
   n_layers for FNO) before loading ckpt["model_state"]. No retraining.
2. Loads the test sets (one .npz per N) made as above.
3. DeepONet: resamples u_0 to the 128 sensors and queries the trunk at
   x_j = j/N.
4. FNO: applies the model to u_0 on the N grid directly.
5. Computes the relative L2 error at each N the way the training scripts
   do: ||pred - true|| / ||true|| pooled over all test values at that N
   (not a mean of per-sample errors).
6. Writes resolution_results.txt and plots error vs N on log-log axes.
Everything runs on the CPU, with numpy and torch only (no scipy).
```

**Review prompt:**

```
Read the lead's script. Verify:

1. Test data: fresh pairs at each N from generate_burgers_data.py with a
   seed other than 42 and the training nu, T and length scale. Ground
   truth at each N comes from the solver run at that N, NOT from
   resampling the N=128 ground truth (that would conflate two effects).
2. DeepONet: u_0 is resampled to the 128 training sensors (x_i = i/128)
   with a periodic wrap; the trunk is queried at x_j = j/N directly; the
   prediction is NOT resampled.
3. FNO: applied to u_0 on the N grid directly (the model builds its own
   positions x_j = j/N); N//2 + 1 >= modes at every N tested.
4. Both models are rebuilt from the checkpoints' saved args and their
   saved weights are loaded; nothing is retrained.
5. The relative L2 error is pooled: ||pred - true|| / ||true|| over all
   test values at each N, as in deeponet_burgers.py and fno_burgers.py.

Judge only whether the test is fair to both methods. Do not predict
what the results should be.

PASS / FAIL each.

Script:
<paste>
```

---

## Meta-tip — interpreting the cross-resolution result

Do not decide in advance which method should come out ahead, and do not adjust the test until it shows a pattern you
expected. Judge whether the comparison is fair: the same fresh test functions for both models at each N, a DeepONet
resampling step you can state in one sentence, one pooled error definition everywhere, and no retraining. Then report
what you measured, including results that surprise you, and say what they do and do not show (one trained model of
each kind, one dataset, one ν).

If a number looks wrong, check the setup before blaming the architecture: seed 42 at N=128 reproduces training pairs;
a model rebuilt with arguments other than the saved ones fails to load; and at N=128 the resampling step should return
u_0 unchanged, so check that it does.
