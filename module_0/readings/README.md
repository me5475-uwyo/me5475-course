# Readings — layout and naming convention

Applies to every module. Follow it for all future reading material.

```
module_<N>/readings/
├── reading_list.md              index for the module: what to read, before which lecture, and why
├── <topic>_primer.md            a from-scratch introduction to a topic (fe_1d_primer, autodiff_primer)
├── <topic>_walkthrough.md       a worked example followed end to end (mlp_sine_walkthrough)
├── <topic>_derivation.md        a mathematical derivation (weak_form_derivation)
├── <topic>_reference.md         a lookup document (voigt_notation_reference)
├── <topic>_guide.md             an operational how-to (torchscript_export_guide)
├── measured_results.md          ONLY if the module's readings cite computed numbers — see below
└── figures/*.png                figures referenced by the readings
```

**Rules**

1. **Descriptive snake_case, no letter or number prefixes.** `fe_1d_primer.md`, not `C_fe_1d_primer.md`.
   Ordering lives in `reading_list.md`, which is the one place a reader looks for sequence. Prefixes rot
   the moment a reading is inserted, dropped, or reordered.
2. **The suffix states the genre** — `_primer`, `_walkthrough`, `_derivation`, `_reference`, `_guide`,
   `_anatomy`, `_cheatsheet`. Pick the closest existing one before inventing a new one.
3. **Runnable companions live in `module_<N>/examples/`**, never in `readings/`. A reading references its
   script by path; the script stands alone and runs.
4. **Every computed number in a reading must trace to `measured_results.md`**, which records what was run,
   where, and when (environment, cluster job id, date). Readings cite that file; they never carry a number
   that was not measured. If a value is wanted but unmeasured, write "(not measured)".
5. **`reading_list.md` is the index.** Any new reading is added there with: when to read it, and one line on
   what it covers.
