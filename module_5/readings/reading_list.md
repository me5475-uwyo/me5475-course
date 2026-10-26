# Module 5 — Reading List

A short list: Lab 5 is a deployment lab, not a theory one. **[Primary]** required; **[Optional]** recommended.

Every computed number in Lab 5 and in these readings (run times, errors, Newton iterations, the strain ranges) is
recorded, with what was run where and when, in `module_5/readings/measured_results.md`.

---

## Before Lecture 26 (assigned in Lecture 25)

### [Primary] `module_5/readings/moose_material_interface_reference.md`

What a MOOSE material does at each quadrature point, and what the course material `ComputeTorchScriptStress` does
with your network there: the input block, the tangent by automatic differentiation, σ_zz, and what it cannot do.
You do not write C++, but you need this to read a failed run.

### [Primary] PyTorch documentation, *TorchScript*.

docs.pytorch.org/docs/2.5/jit.html (the torch 2.5 version; checked 2026-10-05). The introduction and the
`torch.jit.trace` / `torch.jit.script` sections are enough. TorchScript is the format libtorch, and therefore MOOSE,
loads.

### [Optional] PyTorch tutorial, *Loading a TorchScript Model in C++*.

github.com/pytorch/tutorials/blob/1428770b040cdb4cfd98f02a187bf50362d2c7fe/advanced_source/cpp_export.rst (the
December 2024 version, from the torch 2.5 era). What the C++ side of `ComputeTorchScriptStress` does when it loads
your file. The current page at pytorch.org/tutorials/advanced/cpp_export.html now shows only a notice that TorchScript
is deprecated in favor of `torch.export` (checked 2026-10-05). The course stays on the tested torch 2.5.1
TorchScript route. `module_5/readings/torchscript_export_guide.md` covers everything Lab 5 needs.

---

## Before Lecture 27 (assigned in Lecture 26)

### [Primary] `module_5/readings/torchscript_export_guide.md`

The export step: the standardization wrapper, float32 files run in float64, the (1, 3) shape, trace vs script, the
PyTorch version, and the mistakes that are measured to break a run.

### [Primary] MOOSE Framework, *Materials System*.

mooseframework.inl.gov/syntax/Materials/. How material properties are computed at quadrature points and consumed
by kernels. Then the solid-mechanics module's stress-calculator pages (for example `ComputeLinearElasticStress`):
`ComputeTorchScriptStress` is a stress calculator of the same kind (it derives from `ComputeStressBase`), not an
inelastic stress update.

### [Optional] Permann et al., *MOOSE: Enabling massively parallel multiphysics simulation*, SoftwareX 11, 2020.

You read this in M0. Re-skim now that you understand more of the framework.

---

## Reference reading (no due date)

### Neural-network constitutive models inside finite elements

- Z. Liu, *Deep material network with cohesive layers: Multi-stage training and interfacial failure analysis*,
  CMAME 363, 112913, 2020, doi:10.1016/j.cma.2020.112913 (arXiv:1908.04858). A data-driven material model run inside
  an FE code.
- Linka & Kuhl, *A new family of constitutive artificial neural networks towards automated model discovery*,
  CMAME 403, 2023. Networks built so their parameters mean something (relevant to M4 too).
- Mozaffar et al., *Deep learning predicts path-dependent plasticity*, PNAS 116(52), 2019 (from M4). The kind of
  model `ComputeTorchScriptStress` cannot deploy, because it keeps no history (see the interface reading).

### libtorch (for reading the material's source)

- PyTorch C++ API documentation, pytorch.org/cppdocs/. `torch::jit::script::Module` and `torch::autograd::grad`
  are the two pieces the material uses. MOOSE wraps the first as `Moose::TorchScriptModule`.

### Writing your own MOOSE material

- MOOSE Framework, *Examples and Tutorials*, mooseframework.inl.gov/getting_started/examples_and_tutorials/. The
  examples that add a custom `Material` show the C++ pattern `ComputeTorchScriptStress` follows.

For Lab 5 the material is course infrastructure, compiled into the course executable. You use it from the input file.
