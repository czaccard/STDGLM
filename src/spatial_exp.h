#ifndef SPATIAL_EXP_E
#define SPATIAL_EXP_E

#include <RcppEigen.h> // Include necessary headers within the header guard
#include <limits>
#include <cmath>           // For std::log, std::exp, std::isfinite
#include "logsumexp.h"
#include "commons.h"

// Use inline keyword to prevent multiple definition errors if this header
// is included in multiple source files (though less likely in typical Rcpp packages)

// declaration
inline double MH_spatial_correlation_EXP(
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& Z,
    double rho1,
    double min_tau,
    double max_tau
);
inline Rcpp::List MH_spatial_correlation_EXP_fast(
    const Eigen::MatrixXd& Z,
    const double& rho1,
    const std::vector<Eigen::MatrixXd>& Q0,
    const Eigen::VectorXd& allowed_range,
    const Eigen::VectorXd& logdet
);
inline double posterior_conditional_variance_dense(
    const Eigen::MatrixXd& Z,
    const Eigen::MatrixXd& invCorrZ,
    double a1,
    double b1,
    int nstar,
    int tstar
);
inline Rcpp::List logdet_and_Q0_EXP(
    const double& min_tau, const double& max_tau,
    const Eigen::MatrixXd& W,
    const int& T
);




//--------------------------------------------------------------------------
// Function to sample spatial correlation parameter
//--------------------------------------------------------------------------
inline double MH_spatial_correlation_EXP(
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& Z,
    double rho1,
    double min_tau,
    double max_tau
) {
  const int N = Z.rows();
  const int T = Z.cols();

  if (W.rows() != W.cols()) {
    Rcpp::stop("W must be a square matrix.");
  }
  if (W.rows() != N) {
    Rcpp::stop("Number of rows in W must match number of rows in Z.");
  }


  Eigen::VectorXd allowed_range = Eigen::VectorXd::LinSpaced(15, min_tau, max_tau);
  const int nr = allowed_range.size();
  Eigen::VectorXd lpos = Eigen::VectorXd::Zero(nr);

  for (int i = 0; i < nr; ++i) {
    double range = allowed_range(i);
    Eigen::MatrixXd invS_unscaled = (-W.array() * (1.0 / range)).exp().matrix().inverse();
	  Eigen::MatrixXd invS_dense = Eigen::MatrixXd(invS_unscaled) * (1.0 / rho1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_solver(invS_dense);
    if (eig_solver.info() != Eigen::Success) {
		Rcpp::warning("eig_solver did not succeed at range (%.2f).", range);
      lpos(i) = -std::numeric_limits<double>::infinity();
      continue;
    }

    Eigen::VectorXd eigval = eig_solver.eigenvalues();
    /*if ((eigval.array() <= 1e-12).any()) {
		Rcpp::warning("eigval too small at range (%.2f).", range);
      lpos(i) = -std::numeric_limits<double>::infinity();
      continue;
    }*/

    double log_det_invS = eigval.array().pow(-1.0).log().sum();
    if (!std::isfinite(log_det_invS)) {
		Rcpp::warning("log_det_invS infinite at range (%.2f).", range);
      lpos(i) = -std::numeric_limits<double>::infinity();
      continue;
    }

    double quad_form = 0.0;
    for (int t = 0; t < T; ++t) {
      quad_form += Z.col(t).transpose() * invS_dense * Z.col(t);
    }

    lpos(i) = - 0.5 * T * log_det_invS - 0.5 * quad_form;
    if (!std::isfinite(lpos(i))) {
      lpos(i) = -std::numeric_limits<double>::infinity();
    }
  }

  if ((lpos.array() == -std::numeric_limits<double>::infinity()).all()) {
    Rcpp::warning("All log-posterior values are -Inf. Returning middle of range.");
    return (min_tau + max_tau) / 2.0;
  }

  double log_normalizer = log_sum_exp(lpos);
  Eigen::VectorXd log_probx = lpos.array() - log_normalizer;
  Eigen::VectorXd probx = log_probx.array().exp();

  for (int i = 0; i < probx.size(); ++i) {
    if (!std::isfinite(probx(i))) probx(i) = 0.0;
  }

  double sum_probx = probx.sum();
  if (sum_probx <= 1e-12 || !std::isfinite(sum_probx)) {
    probx.setConstant(1.0 / nr);
  } else {
    probx /= sum_probx;
  }

  Rcpp::NumericVector range_vec = Rcpp::wrap(allowed_range);
  Rcpp::NumericVector prob_vec = Rcpp::wrap(probx);

  if (Rcpp::is_true(Rcpp::any(Rcpp::is_na(prob_vec)))) {
    Rcpp::warning("NA values found in sampling probabilities. Using uniform weights.");
    prob_vec = Rcpp::NumericVector(nr, 1.0 / nr);
  }

  Rcpp::NumericVector sampled_value = Rcpp::sample(range_vec, 1, false, prob_vec);
  return sampled_value[0];
}


//--------------------------------------------------------------------------
// Function to sample spatial correlation parameter (fast version)
//--------------------------------------------------------------------------
inline Rcpp::List MH_spatial_correlation_EXP_fast(
    const Eigen::MatrixXd& Z,
    const double& rho1,
    const std::vector<Eigen::MatrixXd>& Q0,
    const Eigen::VectorXd& allowed_range,
    const Eigen::VectorXd& logdet
) {
  const int N = Z.rows();
  const int T = Z.cols();

  const int nr = allowed_range.size();
  Eigen::VectorXd lpos = Eigen::VectorXd::Zero(nr);

  for (int i = 0; i < nr; ++i) {
    Eigen::MatrixXd invS_dense = Q0[i] * (1.0 / rho1);

    double quad_form = 0.0;
    for (int t = 0; t < T; ++t) {
      quad_form += Z.col(t).transpose() * invS_dense * Z.col(t);
    }

    lpos(i) = logdet(i) - 0.5 * N * T * std::log(rho1) - 0.5 * quad_form;

    if (!std::isfinite(lpos(i))) {
      lpos(i) = -std::numeric_limits<double>::infinity();
    }
  }


  if ((lpos.array() == -std::numeric_limits<double>::infinity()).all()) {
    Rcpp::warning("All log-posterior values are -Inf. Returning middle of range.");
    return (allowed_range(6));
  }

  double log_normalizer = log_sum_exp(lpos);
  Eigen::VectorXd log_probx = lpos.array() - log_normalizer;
  Eigen::VectorXd probx = log_probx.array().exp();

  for (int i = 0; i < probx.size(); ++i) {
    if (!std::isfinite(probx(i))) probx(i) = 0.0;
  }

  double sum_probx = probx.sum();
  if (sum_probx <= 1e-12 || !std::isfinite(sum_probx)) {
    probx.setConstant(1.0 / nr);
  } else {
    probx /= sum_probx;
  }

  Rcpp::NumericVector range_vec = Rcpp::wrap(allowed_range);
  Rcpp::NumericVector prob_vec = Rcpp::wrap(probx);

  if (Rcpp::is_true(Rcpp::any(Rcpp::is_na(prob_vec)))) {
    Rcpp::warning("NA values found in sampling probabilities. Using uniform weights.");
    prob_vec = Rcpp::NumericVector(nr, 1.0 / nr);
  }

  Rcpp::NumericVector sampled_value = Rcpp::sample(range_vec, 1, false, prob_vec);
  double range_draw = sampled_value[0];
  int range_index = find_index(range_draw, allowed_range);
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("range_draw") = range_draw,
    Rcpp::Named("range_index") = range_index,
    Rcpp::Named("Q0_draw") = Q0[range_index],
    Rcpp::Named("lpos_draw") = lpos(range_index)
  );
  return result;
}

//--------------------------------------------------------------------------
// Function to sample variance parameter 
// Assumes rho1 ~ IG(a1, b1) prior => 1/rho1 ~ Gamma(a1, b1) prior
//--------------------------------------------------------------------------
inline double posterior_conditional_variance_dense(
    const Eigen::MatrixXd& Z,
    const Eigen::MatrixXd& invCorrZ,
    double a1,
    double b1,
    int nstar,
    int tstar
) {
  const int N = Z.rows();
  const int T = Z.cols();

  if (invCorrZ.rows() != invCorrZ.cols()) {
    Rcpp::stop("invCorrZ must be a square matrix.");
  }
  if (invCorrZ.rows() != N) {
    Rcpp::stop("Spatial dimensions of Z and invCorrZ must match.");
  }

  if (a1 <= 0 || b1 <= 0) {
    Rcpp::warning("Prior parameters a1, b1 must be positive. Using defaults.");
    a1 = std::max(a1, 0.01);
    b1 = std::max(b1, 0.01);
  }

  if (nstar <= 0 || tstar < 0) {
    Rcpp::warning("nstar must be positive, tstar must be non-negative. Using defaults.");
    nstar = std::max(nstar, N);
    tstar = std::max(tstar, T);
  }

  // Posterior shape
  double g1_pos = static_cast<double>(nstar * tstar) / 2.0 + a1;

  // Quadratic form sum
  double quad_sum = 0.0;
  for (int t = 0; t < T; ++t) {
    quad_sum += Z.col(t).transpose() * invCorrZ * Z.col(t);
  }

  // Posterior rate
  double g2_pos = b1 + 0.5 * quad_sum;

  if (g1_pos <= 0 || g2_pos <= 0 || !std::isfinite(g1_pos) || !std::isfinite(g2_pos)) {
    Rcpp::warning("Invalid posterior parameters (shape=%.2e, rate=%.2e). Returning small variance.", g1_pos, g2_pos);
    return 1e-6;
  }

  // Sample precision
  double h_eta_star = R::rgamma(g1_pos, 1.0 / g2_pos);
  if (!std::isfinite(h_eta_star) || h_eta_star <= 1e-12) {
    Rcpp::warning("Sampled precision is non-positive or non-finite (%.2e). Returning small variance.", h_eta_star);
    return 1e-6;
  }

  // Return variance
  double rho1_star = 1.0 / h_eta_star;
  if (!std::isfinite(rho1_star) || rho1_star <= 0) {
    Rcpp::warning("Calculated variance is non-positive or non-finite (%.2e). Returning small variance.", rho1_star);
    return 1e-6;
  }

  return rho1_star;
}


inline Rcpp::List logdet_and_Q0_EXP(
    const double& min_tau, const double& max_tau,
    const Eigen::MatrixXd& W,
    const int& T
) {
  Eigen::VectorXd allowed_range = Eigen::VectorXd::LinSpaced(15, min_tau, max_tau);
  const int nr = allowed_range.size();
  std::vector<Eigen::MatrixXd> R0(nr);
  std::vector<Eigen::MatrixXd> Q0(nr);
  Eigen::VectorXd logdet = Eigen::VectorXd::Zero(nr);

  for (int i = 0; i < nr; ++i) {
    double range = allowed_range(i);
    Eigen::MatrixXd S_unscaled = (-W.array() * (1.0 / range)).exp().matrix();
    R0[i] = S_unscaled;
    Eigen::MatrixXd invS_dense = S_unscaled.inverse();
    Q0[i] = invS_dense;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_solver(invS_dense);
    if (eig_solver.info() != Eigen::Success) {
      logdet(i) = -std::numeric_limits<double>::infinity();
      continue;
    }

    Eigen::VectorXd eigval = eig_solver.eigenvalues();
    if ((eigval.array() <= 1e-12).any()) {
      logdet(i) = -std::numeric_limits<double>::infinity();
      continue;
    }

    double log_det_invS = eigval.array().log().sum();
    if (!std::isfinite(log_det_invS)) {
      logdet(i) = -std::numeric_limits<double>::infinity();
      continue;
    }

    logdet(i) = 0.5 * T * log_det_invS;
    if (!std::isfinite(logdet(i))) {
      logdet(i) = -std::numeric_limits<double>::infinity();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("Q0") = Q0,
    Rcpp::Named("R0") = R0,
    Rcpp::Named("logdet") = logdet,
    Rcpp::Named("allowed_range") = allowed_range
  );
  return result;
}

#endif