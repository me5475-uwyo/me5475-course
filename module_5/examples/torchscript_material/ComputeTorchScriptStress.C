//* ME-5475 course app (University of Wyoming, Fall 2026).
//* Neural-network stress from a TorchScript module, for Module 5 / Lab 5.

#include "ComputeTorchScriptStress.h"

registerMooseObject("Me5475App", ComputeTorchScriptStress);

InputParameters
ComputeTorchScriptStress::validParams()
{
  InputParameters params = ComputeStressBase::validParams();
  params.addClassDescription(
      "In-plane stress (sigma_xx, sigma_yy, sigma_xy) from a TorchScript network of the in-plane "
      "Voigt strain (eps_xx, eps_yy, gamma_xy); the tangent comes from automatic differentiation. "
      "sigma_zz is set to zero. Supported: one MOOSE thread per MPI rank.");
  params.addRequiredParam<FileName>("filename", "TorchScript file holding the network.");
  return params;
}

ComputeTorchScriptStress::ComputeTorchScriptStress(const InputParameters & parameters)
  : ComputeStressBase(parameters)
{
#ifdef MOOSE_LIBTORCH_ENABLED
  // One intra-op thread per MPI rank: the evaluations are tiny, and more threads oversubscribe.
  torch::set_num_threads(1);
  _module = std::make_unique<Moose::TorchScriptModule>(getParam<FileName>("filename"));
  _module->to(torch::kFloat64);
  _module->eval();
#else
  mooseError("ComputeTorchScriptStress needs MOOSE configured with libtorch "
             "(./configure --with-libtorch).");
#endif
}

void
ComputeTorchScriptStress::computeQpStress()
{
#ifdef MOOSE_LIBTORCH_ENABLED
  const RankTwoTensor & eps = _mechanical_strain[_qp];

  // Network input: plane Voigt strain with engineering shear.
  const std::vector<double> e = {eps(0, 0), eps(1, 1), eps(0, 1) + eps(1, 0)};
  torch::Tensor x = torch::tensor(e, torch::dtype(torch::kFloat64)).reshape({1, 3});
  x.requires_grad_(true);

  const torch::Tensor out = _module->forward(x);
  if (out.numel() != 3)
    mooseError("ComputeTorchScriptStress '", name(), "': the network in '",
               getParam<FileName>("filename"), "' returned ", out.numel(),
               " values; it must return 3 (sigma_xx, sigma_yy, sigma_xy) for one strain.");
  const torch::Tensor y = out.reshape({3});
  if (!torch::isfinite(y).all().item<bool>())
    mooseError("ComputeTorchScriptStress '", name(), "': the network in '",
               getParam<FileName>("filename"), "' returned a non-finite stress at element ",
               _current_elem->id(), ", quadrature point ", _qp, " (strain ", e[0], ", ", e[1],
               ", ", e[2], ").");

  // Stress: the in-plane components; sigma_zz is not predicted (see the class comment).
  _stress[_qp].zero();
  _stress[_qp](0, 0) = y[0].item<double>();
  _stress[_qp](1, 1) = y[1].item<double>();
  _stress[_qp](0, 1) = _stress[_qp](1, 0) = y[2].item<double>();

  // Tangent J(a, b) = d sigma_a / d v_b, with v = (eps_xx, eps_yy, gamma_xy), by autograd.
  double J[3][3];
  for (unsigned int a = 0; a < 3; ++a)
  {
    const auto g = torch::autograd::grad({y[a]}, {x}, /*grad_outputs=*/{}, /*retain_graph=*/a < 2)[0];
    for (unsigned int b = 0; b < 3; ++b)
      J[a][b] = g[0][b].item<double>();
  }

  // Map to the rank-four tangent with tensor strain. gamma_xy = eps_xy + eps_yx, so
  // d sigma / d eps_xy = d sigma / d eps_yx = d sigma / d gamma_xy; the shear stress fills both
  // (0,1) and (1,0). For isotropic linear elasticity this gives C_xyxy = mu, as it should.
  const unsigned int idx[3][2][2] = {{{0, 0}, {0, 0}}, {{1, 1}, {1, 1}}, {{0, 1}, {1, 0}}};
  _Jacobian_mult[_qp].zero();
  for (unsigned int a = 0; a < 3; ++a)
    for (unsigned int ia = 0; ia < 2; ++ia)
      for (unsigned int b = 0; b < 3; ++b)
        for (unsigned int ib = 0; ib < 2; ++ib)
          _Jacobian_mult[_qp](idx[a][ia][0], idx[a][ia][1], idx[b][ib][0], idx[b][ib][1]) =
              J[a][b];

  _elastic_strain[_qp] = eps;
#endif
}
