#pragma once

/// @file types.h
/// @brief Fundamental type aliases for the portfolio optimizer.
///
/// Dual precision strategy:
///   - Scalar (float):     GPU path — throughput and VRAM efficiency
///   - ScalarCPU (double): CPU path — optimizer convergence and estimation

#include <Eigen/Core>

namespace cpo {

/// GPU-path scalar: float for throughput and VRAM efficiency.
using Scalar = float;

/// CPU-path scalar: double for optimizer convergence and estimation.
using ScalarCPU = double;

/// Integer index type used throughout the library.
using Index = int;

// ── Eigen typedefs — GPU-precision (float) ─────────────────────────

using VectorXs = Eigen::VectorXf;
using MatrixXs = Eigen::MatrixXf;

// ── Eigen typedefs — CPU-precision (double) ────────────────────────

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

}  // namespace cpo
