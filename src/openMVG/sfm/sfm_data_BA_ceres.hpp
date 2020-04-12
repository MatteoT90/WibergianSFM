// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_DATA_BA_CERES_HPP
#define OPENMVG_SFM_SFM_DATA_BA_CERES_HPP

#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data_BA.hpp"
#include "openMVG/types.hpp"

namespace ceres { class CostFunction; }
namespace openMVG { namespace cameras { struct IntrinsicBase; } }
namespace openMVG { namespace sfm { struct SfM_Data; } }

namespace openMVG {
namespace sfm {

/// Create the appropriate cost functor according the provided input camera intrinsic model
/// Can be residual cost functor can be weighetd if desired (default 0.0 means no weight).
ceres::CostFunction * IntrinsicsToCostFunction
(
  cameras::IntrinsicBase * intrinsic,
  const Vec2 & observation,
  const double weight = 0.0
);

void extract_rotation
(
    const Mat3 R,
    double* angleAxis
);

class Bundle_Adjustment_Ceres : public Bundle_Adjustment
{
  public:
  struct BA_Ceres_options
  {
    bool bVerbose_;
    unsigned int nb_threads_;
    bool bCeres_summary_;
    int linear_solver_type_;
    int preconditioner_type_;
    int sparse_linear_algebra_library_type_;
    double parameter_tolerance_;
    bool bUse_loss_function_;

    BA_Ceres_options(const bool bVerbose = true, bool bmultithreaded = true);
  };
  private:
    BA_Ceres_options ceres_options_;

  public:
  explicit Bundle_Adjustment_Ceres
  (
    const Bundle_Adjustment_Ceres::BA_Ceres_options & options =
    std::move(BA_Ceres_options())
  );

  BA_Ceres_options & ceres_options();

  bool Adjust
  (
    // the SfM scene to refine
    sfm::SfM_Data & sfm_data,
    // tell which parameter needs to be adjusted
    const Optimize_Options & options
  ) override;
  //TODO adding the new function to collect vectors
  bool Adjust
  (
    // the SfM scene to refine
    sfm::SfM_Data & sfm_data,
    // vectors for Jacobian extraction
    int* vec_rows, int g,
    int* vec_cols, int h,
    double* vec_grad, int i
  );

  bool MyAdjust
    (
      double* poses3d, int a,
      double* intrinsics, int b,
      double* observation3d, int c,
      double* observation2d, int e,
      int* track, int f,
      int* cams, int g,
      int* vec_rows, int h,
      int* vec_cols, int i,
      double* vec_grad, int j,
      double* weight, int l,
      double* cost, int m,
      double* residuals, int n,
      double* gradient, int o,
      double* rotations, int p,
      int mode
    );

};
// end new lines

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_DATA_BA_CERES_HPP
