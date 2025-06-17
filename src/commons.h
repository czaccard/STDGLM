#ifndef COMMONS_H
#define COMMONS_H

#include <RcppEigen.h>

inline Eigen::MatrixXd ar1_correlation_matrix(int size, double phi) {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            R(i, j) = std::pow(phi, std::abs(i - j));
        }
    }
    return R;
}


inline int find_index(const double& value, const Eigen::VectorXd& v) { // equivalent in R -> which(v == value)
    for (int i = 0; i < v.size(); ++i) {
        if (v(i) == value) {
            return i;
        }
    }
    throw std::runtime_error("Error: Value " + std::to_string(value) + " not found in vector.");
}



Eigen::SparseMatrix<double> createBlockDiagonal_dense(const Eigen::MatrixXd& block, const double& initial, int T) {
  int m = block.rows();
  int rows = m * T;
  int cols = m * T;

  typedef Eigen::Triplet<double> Tpl;
  std::vector<Tpl> tripletList;
  tripletList.reserve(T * m * m);

  for (int i = 0; i < m; ++i) {
    tripletList.emplace_back(i, i, initial);  // prior for beta_0
  }

  for (int t = 1; t < T; ++t) {
    int rowOffset = t * m;
    int colOffset = t * m;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        double val = block(i, j);
        if (val != 0.0) {
          tripletList.emplace_back(rowOffset + i, colOffset + j, val);
        }
      }
    }
  }

  Eigen::SparseMatrix<double> result(rows, cols);
  result.setFromTriplets(tripletList.begin(), tripletList.end());
  return result;
}


Eigen::SparseMatrix<double> createBlockDiagonal(const Eigen::SparseMatrix<double>& block, const double& initial, int T) {
  int m = block.rows();
  int rows = m * T;
  int cols = m * T;

  typedef Eigen::Triplet<double> Tpl;
  std::vector<Tpl> tripletList;
  tripletList.reserve(T * m * m);

  for (int i = 0; i < m; ++i) {
    tripletList.emplace_back(i, i, initial);  // prior for beta_0
  }

  for (int t = 1; t < T; ++t) {
    int rowOffset = t * m;
    int colOffset = t * m;
    for (int k = 0; k < block.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it; ++it) {
        tripletList.emplace_back(rowOffset + it.row(), colOffset + it.col(), it.value());
      }
    }
  }


  Eigen::SparseMatrix<double> result(rows, cols);
  result.setFromTriplets(tripletList.begin(), tripletList.end());
  return result;
}


Eigen::VectorXd cpp_prctile(const Eigen::MatrixXd& Y, double P) {
  const int n_rows = Y.rows();
  const int n_cols = Y.cols();
  Eigen::VectorXd quantiles(n_rows);
  
  for (int i = 0; i < n_rows; ++i) {
    std::vector<double> row;
    for (int j = 0; j < n_cols; ++j) {
      double val = Y(i, j);
      if (std::isfinite(val)) {
        row.push_back(val);
      }
    }
    
    if (row.empty()) {
      quantiles(i) = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    
    std::sort(row.begin(), row.end());
    double index = (row.size() - 1) * P;
    size_t lo = static_cast<size_t>(std::floor(index));
    size_t hi = static_cast<size_t>(std::ceil(index));
    double q_lo = row[lo];
    double q_hi = row[hi];
    
    if (index > lo && q_hi != q_lo) {
      double h = index - lo;
      quantiles(i) = (1.0 - h) * q_lo + h * q_hi;
    } else {
      quantiles(i) = q_lo;
    }
  }
  
  return quantiles;
}



Eigen::VectorXd lmfit(const Eigen::MatrixXd& X_data, const Eigen::VectorXd& y_data) {
  return X_data.colPivHouseholderQr().solve(y_data);
}



Eigen::VectorXd extract_last_col(Rcpp::NumericMatrix mat)  {
  int nrow = mat.nrow();
  int ncol = mat.ncol();
  Eigen::VectorXd out(nrow);
  for (int i = 0; i < nrow; ++i) {
    out(i) = mat(i, ncol - 1);
  }
  return out;
}


Rcpp::NumericVector cube_to_array(const std::vector<Eigen::MatrixXd>& cube_like, const double& n_samples_collected)  {
  int rows = cube_like[0].rows();
  int cols = cube_like[0].cols();
  int slices = static_cast<int>(cube_like.size());
  Rcpp::NumericVector arr(Rcpp::Dimension(rows, cols, slices));
  for (int k = 0; k < slices; ++k) {
    const Eigen::MatrixXd& mat = cube_like[k];
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int idx = i + j * rows + k * rows * cols;
        arr[idx] = mat(i, j) / n_samples_collected;
      }
    }
  }
  return arr;
}


Rcpp::List minMaxUpperTriangular(const Eigen::MatrixXd& mat) {
  int rows = mat.rows();
  int cols = mat.cols();
  double maxVal = std::numeric_limits<double>::lowest();
  double minNonZero = std::numeric_limits<double>::max();
  bool foundNonZero = false;
  
  for (int i = 0; i < rows; ++i) {
    for (int j = i; j < cols; ++j) {
      double val = mat(i, j);
      if (val > maxVal) {
        maxVal = val;
      }
      if (val != 0.0 && val < minNonZero) {
        minNonZero = val;
        foundNonZero = true;
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("max") = maxVal,
    Rcpp::Named("min_nz") = foundNonZero ? minNonZero : (maxVal / 100.0)
  );
}




Rcpp::List computeWAIC(const std::vector<Eigen::MatrixXd>& LL) {

  const int p = LL[0].rows(); // number of space points
  const int t = LL[0].cols(); // number of time points
  const int N = p * t; // data points
  const int S = LL.size(); // number of samples

  Eigen::MatrixXd log_lik(S, N);
  for (int k = 0; k < S; ++k) {
    Eigen::Map<const Eigen::RowVectorXd> ll(LL[k].data(), LL[k].size());
    log_lik.row(k) = ll;
  }

  
  Eigen::VectorXd lppd_i(N);
  Eigen::VectorXd elpd_i(N);
  Eigen::VectorXd p_waic_i(N);
  Eigen::VectorXd waic_i(N);

  for (int n = 0; n < N; ++n) {
    Eigen::VectorXd log_lik_n = log_lik.col(n);

    // log-sum-exp trick
    double max_log = log_lik_n.maxCoeff();
    Eigen::VectorXd shifted = (log_lik_n.array() - max_log).exp();
    double mean_exp = shifted.mean();
    lppd_i[n] = std::log(mean_exp) + max_log;

    // variance of log-likelihoods
    double var_log = (log_lik_n.array() - log_lik_n.mean()).square().sum() / (S - 1);
    p_waic_i[n] = var_log;

    // expected log pointwise predictive density
    elpd_i[n] = lppd_i[n] - p_waic_i[n];

    // pointwise WAIC contribution
    waic_i[n] = -2.0 * elpd_i[n];
  }

  double elpd = elpd_i.sum();
  double p_waic = p_waic_i.sum();
  double waic = waic_i.sum();

  // Standard errors
  double se_elpd = std::sqrt((elpd_i.array() - elpd_i.mean()).square().sum() / (N - 1) * N);
  double se_p_waic = std::sqrt((p_waic_i.array() - p_waic_i.mean()).square().sum() / (N - 1) * N);
  double se_waic = std::sqrt((waic_i.array() - waic_i.mean()).square().sum() / (N - 1) * N);


  return Rcpp::List::create(
    Rcpp::Named("elpd") = elpd,
    Rcpp::Named("p_waic") = p_waic,
    Rcpp::Named("waic") = waic,
    Rcpp::Named("se_elpd") = se_elpd,
    Rcpp::Named("se_p_waic") = se_p_waic,
    Rcpp::Named("se_waic") = se_waic
  );

}





# endif // COMMONS_H