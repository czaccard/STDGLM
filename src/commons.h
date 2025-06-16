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



# endif // COMMONS_H