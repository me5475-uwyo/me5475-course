//* ME-5475 course app (University of Wyoming, Fall 2026).
//* Neural-network stress from a TorchScript module, for Module 5 / Lab 5.

#pragma once

#include "ComputeStressBase.h"

#ifdef MOOSE_LIBTORCH_ENABLED
#include "TorchScriptModule.h"
#endif

/**
 * In-plane (2-D) stress from a TorchScript network. Supported configuration: one MOOSE thread per
 * MPI rank (torch's intra-op threads are set to 1 per process).
 *
 * The network maps the plane Voigt strain (eps_xx, eps_yy, gamma_xy), with engineering shear
 * gamma_xy = 2 eps_xy, to the plane Voigt stress (sigma_xx, sigma_yy, sigma_xy). Any
 * standardization must be built into the exported module (module_5/examples/export_torchscript.py
 * does this). The module is converted to double precision on load, so the nonlinear solve is not
 * limited by single-precision round-off.
 *
 * The tangent dsigma/deps is computed by automatic differentiation through the network at every
 * quadrature point, so Newton's method gets the network's exact Jacobian.
 *
 * sigma_zz is not predicted by the network and is set to zero; it does not enter the in-plane
 * residual of a 2-D plane-strain model, but it is not the physical out-of-plane stress.
 */
class ComputeTorchScriptStress : public ComputeStressBase
{
public:
  static InputParameters validParams();

  ComputeTorchScriptStress(const InputParameters & parameters);

protected:
  virtual void computeQpStress() override;

#ifdef MOOSE_LIBTORCH_ENABLED
  /// The network, loaded from the TorchScript file and converted to double precision
  std::unique_ptr<Moose::TorchScriptModule> _module;
#endif
};
