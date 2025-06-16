#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <RcppEigen.h>


inline Eigen::VectorXd randn_vector(int size) {
    Eigen::VectorXd vec(size);
    for (int i = 0; i < size; ++i) {
        vec(i) = R::rnorm(0.0, 1.0);
    }
    return vec;
}


inline Eigen::MatrixXd randn_matrix(int rows, int cols) {
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat(i, j) = R::rnorm(0.0, 1.0);
        }
    }
    return mat;
}


#endif // DISTRIBUTIONS_H