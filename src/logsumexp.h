#ifndef LOGSUMEXP
#define LOGSUMEXP

#include <RcppEigen.h> // Include necessary headers within the header guard
#include <limits>
#include <cmath>           // For std::log, std::exp, std::isfinite

// Use inline keyword to prevent multiple definition errors if this header
// is included in multiple source files (though less likely in typical Rcpp packages)

// declaration
inline double log_sum_exp(const Eigen::VectorXd& x);



//--------------------------------------------------------------------------
// Helper function for the log-sum-exp trick
//--------------------------------------------------------------------------
inline double log_sum_exp(const Eigen::VectorXd& x) {
  if (x.size() == 0) {
    return -std::numeric_limits<double>::infinity();
  }

  double max_val = x.maxCoeff();
  if (!std::isfinite(max_val)) {
    return max_val;  // Handles Inf, -Inf, or NaN
  }

  double sum_exp = (x.array() - max_val).exp().sum();
  if (sum_exp <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }

  return max_val + std::log(sum_exp);
}



#endif