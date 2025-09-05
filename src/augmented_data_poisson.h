#ifndef AD_POISSON_H
#define AD_POISSON_H

#include <RcppEigen.h>

// declaration
inline Eigen::VectorXd log_post_eta_PLN(const Eigen::VectorXd& eta_i,
                        const Eigen::VectorXd& theta,
                        const Eigen::VectorXd& Y,
                        const Eigen::VectorXd& variance);

inline std::vector<Eigen::MatrixXd> augmented_data_poisson_lognormal(
    const Eigen::MatrixXd& Y,
    Eigen::MatrixXd eta_tilde,
    const Eigen::MatrixXd& thetay,
    const Eigen::VectorXd& Ht,
    Eigen::MatrixXd pswitch_Y,
    const Eigen::MatrixXd& ctuning);



inline Eigen::VectorXd log_post_eta_PLN(const Eigen::VectorXd& eta_i,
                        const Eigen::VectorXd& theta,
                        const Eigen::VectorXd& Y,
                        const Eigen::VectorXd& variance) {
  Eigen::VectorXd term1 = eta_i.array() * Y.array();
  Eigen::VectorXd term2 = - eta_i.array().exp();
  Eigen::VectorXd term3 = -0.5 * ((eta_i - theta).array().square() / variance.array());

  Eigen::VectorXd logf1 = term1 + term2 + term3;
  return logf1;
}



inline std::vector<Eigen::MatrixXd> augmented_data_poisson_lognormal(
    const Eigen::MatrixXd& Y,
    Eigen::MatrixXd eta_tilde,
    const Eigen::MatrixXd& thetay,
    const Eigen::VectorXd& Ht,
    Eigen::MatrixXd pswitch_Y,
    const Eigen::MatrixXd& ctuning) {
  int N2 = Y.rows();
  int T = Y.cols();
  bool single_variance = (Ht.size() == 1);
  if (!single_variance && Ht.size() != N2 * T) {
    Rcpp::stop("Incompatible dimensions for Ht. Size must be 1 or N2 * T.");
    // Ht must have all locations at time 1, then at time 2, 3, ...
  }
  if (ctuning.size() > 1 && ctuning.size() != N2 * T) {
    Rcpp::stop("Incompatible dimensions for ctuning. Size must be 1 or N2 by T.");
    // ctuning must have all locations at time 1, then at time 2, 3, ...
  }

  for (int i = 0; i < N2; ++i) {
    Eigen::VectorXd variance = Eigen::VectorXd::Zero(T);
    if (single_variance) {
      variance.array() += Ht(0);
    } else {
      for (int t = 0; t < T; ++t) {
        variance(t) = Ht(i + t * N2);
      }
    }

    Eigen::VectorXd Sigmaappo05 = variance.array().sqrt();
    Eigen::VectorXd Ytilde_cand(T);
    Eigen::VectorXd eta_tilde_row = eta_tilde.row(i).transpose();
    Eigen::VectorXd thetay_row = thetay.row(i).transpose();
    Eigen::VectorXd Y_row = Y.row(i).transpose();

    if (ctuning.size() > 1) {
      for (int t = 0; t < T; ++t) {
        double t_draw = R::rt(5.0);  // Student-t with df=5
        Ytilde_cand(t) = eta_tilde_row(t) + ctuning(i, t) * t_draw * Sigmaappo05(t);
      }
    } else {
      for (int t = 0; t < T; ++t) {
        double norm_draw = R::rnorm(0.0, 1.0);
        Ytilde_cand(t) = eta_tilde_row(t) + ctuning(0, 0) * norm_draw * Sigmaappo05(t);
      }
    }

    Eigen::VectorXd lpost_draw = log_post_eta_PLN(eta_tilde_row, thetay_row, Y_row, variance);
    Eigen::VectorXd lpost_can = log_post_eta_PLN(Ytilde_cand, thetay_row, Y_row, variance);
    
    for (int t = 0; t < T; ++t) {
      double accprob = lpost_can(t) - lpost_draw(t);
      double u = R::runif(0.0, 1.0);
      if (std::log(u) < std::min(0.0, accprob)) {
        eta_tilde(i, t) = Ytilde_cand(t);
        pswitch_Y(i, t) += 1.0;
      }
    }
  }

  std::vector<Eigen::MatrixXd> out(2);
  out[0] = eta_tilde;
  out[1] = pswitch_Y;
  return out;
}


#endif // AD_POISSON_H