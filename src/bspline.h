#ifndef BSPLINE_H
#define BSPLINE_H

#include <RcppEigen.h>

// Cox–de Boor recursion for B-spline basis
double bspline_basis(int i, int k, double x, const Eigen::VectorXd& knots) {
  if (k == 0) {
    return (x >= knots[i] && x < knots[i + 1]) ? 1.0 : 0.0;
  } else {
    double denom1 = knots[i + k] - knots[i];
    double denom2 = knots[i + k + 1] - knots[i + 1];

    double term1 = denom1 > 0 ? (x - knots[i]) / denom1 * bspline_basis(i, k - 1, x, knots) : 0.0;
    double term2 = denom2 > 0 ? (knots[i + k + 1] - x) / denom2 * bspline_basis(i + 1, k - 1, x, knots) : 0.0;

    return term1 + term2;
  }
}


inline Eigen::MatrixXd bspline_basis_matrix(const Eigen::VectorXd& x, int degree, const Eigen::VectorXd& knots) {
  int n_basis = knots.size() - degree - 1;
  int n_points = x.size();
  Eigen::MatrixXd B(n_points, n_basis);

  for (int j = 0; j < n_points; ++j) {
    for (int i = 0; i < n_basis; ++i) {
      B(j, i) = bspline_basis(i, degree, x[j], knots);
    }
  }

  return B;
}

#endif // BSPLINE_H
