// 2025
// Code prepared by Carlo Zaccardi

#define RCPPDIST_DONT_USE_ARMA

#include <RcppEigen.h>
#include <RcppParallel.h>
#include <RcppDist.h>
#include <algorithm>
#include <variant>
#include <vector>
#include "simulation_smoother.h"
#include "spatial_car.h"
#include "spatial_exp.h"
#include "distributions.h"
#include "commons.h"
#include "bspline.h"
#include "augmented_data_poisson.h"
#include <cmath> // For std::sqrt, std::log, std::abs
#include <random>
#include <limits>
// [[Rcpp::depends(RcppEigen, RcppParallel, RcppDist)]]
// [[Rcpp::plugins(cpp17)]]


inline void scatter_to_rows(Eigen::MatrixXd& dest,
                            const std::vector<int>& rows,
                            const Eigen::VectorXd& src,
                            int col) {
  for (std::size_t r = 0; r < rows.size(); ++r) dest(rows[r], col) = src[r];
}



struct SpacePredWorker : public RcppParallel::Worker {
  // costanti
  const int nr;
  const int q0_idx_space;
  const int q0_idx_st;
  const double rho1_space_k;
  const double rho1_st_k;
  const double phi_st_k;

  // dimensioni tempo
  const int t_new;

  // riferimenti dati condivisi (sola lettura)
  const std::vector<std::vector<int>>& blocks_idx;                 // nblocks x (righe nel blocco)
  const std::vector<Eigen::MatrixXd>& R_cross;                     // size = nblocks * nr (ogniuno ha dim p x p_i)
  const std::vector<Eigen::SparseMatrix<double>>& chol_Corr_pred;  // size = nblocks * nr (p_i x p_i, lower)
  const Eigen::VectorXd& QtB_space;                                // p x 1 = Q1inv_space_dense[k] * Bspacedraw[k]
  const std::vector<Eigen::VectorXd>& U_t;                         // t_new elementi, ognuno p x 1 = Q1inv_st_dense[k] * delta_t
  const std::vector<Eigen::MatrixXd>& W_pred_dense_unused;         // solo per mantenere compatib. se serve in futuro
  const double& ST_k;

  // output condivisi (scrittura su righe disgiunte)
  Eigen::MatrixXd& Bspace_pred_k;        // p_new x 1
  Eigen::MatrixXd& Bst_pred_k;           // p_new x t_new

  // seed base per rng deterministico (opzionale)
  const std::uint64_t base_seed;

  SpacePredWorker(int nr_,
                  int q0_space_,
                  int q0_st_,
                  double rho1_space_k_,
                  double rho1_st_k_,
                  double phi_st_k_,
                  int t_new_,
                  const std::vector<std::vector<int>>& blocks_idx_,
                  const std::vector<Eigen::MatrixXd>& R_cross_,
                  const std::vector<Eigen::SparseMatrix<double>>& chol_Corr_pred_,
                  const Eigen::VectorXd& QtB_space_,
                  const std::vector<Eigen::VectorXd>& U_t_,
                  const std::vector<Eigen::MatrixXd>& W_pred_dense_unused_,
                  const double& ST_k_,
                  Eigen::MatrixXd& Bspace_pred_k_,
                  Eigen::MatrixXd& Bst_pred_k_,
                  std::uint64_t base_seed_)
    : nr(nr_), q0_idx_space(q0_space_), q0_idx_st(q0_st_),
      rho1_space_k(rho1_space_k_), rho1_st_k(rho1_st_k_), phi_st_k(phi_st_k_),
      t_new(t_new_),
      blocks_idx(blocks_idx_), R_cross(R_cross_), chol_Corr_pred(chol_Corr_pred_),
      QtB_space(QtB_space_), U_t(U_t_), W_pred_dense_unused(W_pred_dense_unused_),
      ST_k(ST_k_),
      Bspace_pred_k(Bspace_pred_k_), Bst_pred_k(Bst_pred_k_),
      base_seed(base_seed_) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t j = begin; j < end; ++j) {
      const auto& rows = blocks_idx[j];
      const int pi_new = static_cast<int>(rows.size());

      // RNG locale per thread (seed deterministico da base_seed e j)
      std::seed_seq ss{ static_cast<unsigned>(base_seed),
                        static_cast<unsigned>(j & 0xffffffffULL),
                        static_cast<unsigned>((j>>32) & 0xffffffffULL) };
      std::mt19937_64 gen(ss);
      std::normal_distribution<double> ndist(0.0, 1.0);

      // --------- SPACE: mean e draw ----------
      // mean: rho1 * R_cross^T * (Q1inv_space * Bspace)
      const Eigen::MatrixXd& Rcx = R_cross[j*nr + q0_idx_space]; // p x pi_new
      Eigen::VectorXd mean_space = rho1_space_k * (Rcx.transpose() * QtB_space); // pi_new

      // sd: sqrt(rho1) * L  (L è lower Cholesky del covariance condizionale del blocco)
      const auto& Lspace = chol_Corr_pred[j*nr + q0_idx_space];  // pi_new x pi_new
      Eigen::VectorXd z_space(pi_new);
      for (int r = 0; r < pi_new; ++r) z_space[r] = ndist(gen);
      Eigen::VectorXd draw_space = std::sqrt(rho1_space_k) * (Lspace * z_space) + mean_space;

      // scrivi su Bspace_pred_k(rows, 0)
      scatter_to_rows(Bspace_pred_k, rows, draw_space, /*col=*/0);

      // --------- SPACETIME: loop sui tempi ----------
      // prev = 0 (come nel codice originale)
      if (ST_k == 1.0) {
        Eigen::VectorXd prev = Eigen::VectorXd::Zero(pi_new);

        const auto& Lst = chol_Corr_pred[j*nr + q0_idx_st]; // stesso L per tutti i tempi (dipende solo dalla range)
        for (int it = 0; it < t_new; ++it) {
          const Eigen::MatrixXd& Rcx_st = R_cross[j*nr + q0_idx_st]; // p x pi_new (stesso indice di range)
          // mean_t = phi * prev + rho1_st * R_cross^T * U_t[it]
          Eigen::VectorXd mean_st = phi_st_k * prev + rho1_st_k * (Rcx_st.transpose() * U_t[it]);

          Eigen::VectorXd z_st(pi_new);
          for (int r = 0; r < pi_new; ++r) z_st[r] = ndist(gen);
          Eigen::VectorXd draw_st = std::sqrt(rho1_st_k) * (Lst * z_st) + mean_st;

          // scrivi su Bst_pred_k(rows, it)
          scatter_to_rows(Bst_pred_k, rows, draw_st, /*col=*/it);
          prev.swap(draw_st);
        }
      }
    }
  }
};



// [[Rcpp::export]]
Rcpp::List dlm_cpp(
    Eigen::MatrixXd Y,
    const std::string& family,
    const std::vector<Eigen::MatrixXd>& X,
    const Rcpp::Nullable<Rcpp::List> Z_nullable,
    const Eigen::MatrixXd& offset, // const std::string& transfY,
    const bool& point_referenced,
    const bool& random_walk,
    const Eigen::VectorXd& ST, // ST interaction effect indicator
    const std::vector<std::vector<int>>& blocks_indices,
    const Eigen::MatrixXd& W_dense,
    const std::vector<Eigen::MatrixXd>& W_pred_dense,
    const std::vector<Eigen::MatrixXd>& W_cross_dense,
    const Rcpp::Nullable<Rcpp::List> X_pred_nullable,
    const Rcpp::Nullable<Rcpp::List> Z_pred_nullable,
    const Eigen::MatrixXd& offset_pred,
    const int& nrep,
    const int& nburn,
    const int& thin,
	  const int& print_interval,
    const Eigen::VectorXd& V_beta_0,		  // Prior variance of initial state
    const double& V_gamma,		  // Prior variance of constant coefficients
    const Eigen::VectorXd& a_inn_time,
    const Eigen::VectorXd& b_inn_time,
    const Eigen::VectorXd& a_rho1s,
    const Eigen::VectorXd& b_rho1s,
    const Eigen::VectorXd& a_rho1st,
    const Eigen::VectorXd& b_rho1st,
    const double& a_s2,
    const double& b_s2,
    const double& ctuning,
    const bool& keepY,
    const bool& keepLogLik,
    Rcpp::Nullable<Rcpp::List> out_prev_nullable
) {
  Eigen::setNbThreads(1); // Avoid double-threading (Eigen + RcppParallel)

  const int p = X[0].rows();
  const int t = X[0].cols();
  const int ncovx = X.size();
  const int nblocks = blocks_indices.size();
  const Eigen::VectorXd offset_vec = Eigen::Map<const Eigen::VectorXd>(offset.data(), offset.size());
  
  Eigen::SparseMatrix<double> W;
  Eigen::SparseMatrix<double> D(p, p);
  double min_rho2, max_rho2;
  Rcpp::List logdet_and_Q0;
  std::vector<Eigen::SparseMatrix<double>> Q0;
  std::vector<Eigen::MatrixXd> Q0_dense;
  std::vector<Eigen::MatrixXd> R0;
  Eigen::VectorXd logdet_Q0_space, logdet_Q0_spacetime, allowed_range;
  if (!point_referenced) {
    W = W_dense.sparseView();
    min_rho2 = 0.1;
    max_rho2 = 1.0;
    
    // Adjacency matrix processing
    Eigen::VectorXd rowSums = Eigen::VectorXd::Zero(p);
    // Compute row sums of W
    for (int k = 0; k < W.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(W, k); it; ++it) {
        rowSums(it.row()) += it.value();
      }
    }
    // Set diagonal of D
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(p);
    for (int i = 0; i < p; ++i) {
      triplets.emplace_back(i, i, rowSums(i));
    }
    D.setFromTriplets(triplets.begin(), triplets.end());

    logdet_and_Q0 = logdet_and_Q0_CAR(min_rho2, max_rho2, W, D, 1);
    Q0 = Rcpp::as<std::vector<Eigen::SparseMatrix<double>>>(logdet_and_Q0["Q0"]);
    allowed_range = Rcpp::as<Eigen::VectorXd>(logdet_and_Q0["allowed_range"]);
    logdet_Q0_space = Rcpp::as<Eigen::VectorXd>(logdet_and_Q0["logdet"]);
    logdet_Q0_spacetime = logdet_Q0_space * t;
  } else {
    Rcpp::List min_and_max = minMaxUpperTriangular(W_dense);
    min_rho2 = Rcpp::as<double>(min_and_max["min_nz"])/3.0;
    max_rho2 = Rcpp::as<double>(min_and_max["max"])/3.0;

    logdet_and_Q0 = logdet_and_Q0_EXP(min_rho2, max_rho2, W_dense, 1);
    Q0_dense = Rcpp::as<std::vector<Eigen::MatrixXd>>(logdet_and_Q0["Q0"]);
    R0 = Rcpp::as<std::vector<Eigen::MatrixXd>>(logdet_and_Q0["R0"]);
    allowed_range = Rcpp::as<Eigen::VectorXd>(logdet_and_Q0["allowed_range"]);
    logdet_Q0_space = Rcpp::as<Eigen::VectorXd>(logdet_and_Q0["logdet"]);
    logdet_Q0_spacetime = logdet_Q0_space * t;
  }

  Eigen::VectorXd phi_ar1_time_draw, phi_ar1_space_time_draw;
  if (random_walk) {
    phi_ar1_time_draw = Eigen::VectorXd::Constant(ncovx, 1.0);
    phi_ar1_space_time_draw = Eigen::VectorXd::Constant(ncovx, 1.0);
  } else {
    phi_ar1_time_draw = Eigen::VectorXd::Constant(ncovx, 0.5);
    phi_ar1_space_time_draw = Eigen::VectorXd::Constant(ncovx, 0.5);
  }
  phi_ar1_space_time_draw.array() *= ST.array();

  Eigen::MatrixXd Z_mat;
  int ncovz = 0;
  
  if (Z_nullable.isNotNull()) {
    Rcpp::List Z_list(Z_nullable);
    ncovz = Z_list.size();
    Z_mat.resize(p * t, ncovz);
    for (int k = 0; k < ncovz; ++k) {
      Eigen::Map<Eigen::MatrixXd> Z_k(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Z_list[k]));
      if (Z_k.rows() != p || Z_k.cols() != t) {
        Rcpp::stop("Dimensions of Z must be p x t x ncovz");
      }
      Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(Z_k.data(), Z_k.size());
      Z_mat.col(k) = vec;
    }
  }

  int nr = allowed_range.size();
  int p_new = 0, t_new = 0, h_ahead = 0;
  Eigen::MatrixXd Z_pred_mat;
  std::vector<Eigen::MatrixXd> Btime_pred_draw(ncovx); // Time effect T-beta
  std::vector<Eigen::MatrixXd> Bspace_pred_draw(ncovx); // Space effect S-beta
  std::vector<Eigen::MatrixXd> Bspacetime_pred_obs_sites(ncovx); // Space-time effect ST-beta at observed sites
  std::vector<Eigen::MatrixXd> Bspacetime_pred_draw(ncovx); // Space-time effect ST-beta
  std::vector<Eigen::SparseMatrix<double>> chol_Corr, chol_Corr_pred;
  std::vector<Eigen::MatrixXd> R_cross;
  std::vector<Eigen::MatrixXd> X_pred;
  if (X_pred_nullable.isNotNull()) {
    Rcpp::List X_pred_list(X_pred_nullable);
    X_pred = Rcpp::as<std::vector<Eigen::MatrixXd>>(X_pred_list);
    p_new = X_pred[0].rows();
    t_new = X_pred[0].cols();
    h_ahead = t_new-t;

    if (Z_nullable.isNotNull()) {
      Rcpp::List Z_pred_list(Z_pred_nullable);
      Z_pred_mat.resize(p_new * t_new, ncovz);
      for (int k = 0; k < ncovz; ++k) {
        Eigen::Map<Eigen::MatrixXd> Z_k(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(Z_pred_list[k]));
        if (Z_k.rows() != p_new || Z_k.cols() != t_new) {
          Rcpp::stop("Dimensions of Z_pred must be p_new x t_new x ncovz");
        }
        Eigen::VectorXd vec = Eigen::Map<Eigen::VectorXd>(Z_k.data(), Z_k.size());
        Z_pred_mat.col(k) = vec;
      }
    }

    // Initialize prediction and interpolation effects
    for (int k = 0; k < ncovx; ++k) {
      Btime_pred_draw[k].setZero(1, h_ahead); // Time effect T-beta
      Bspace_pred_draw[k].setZero(p_new, 1); // Space effect S-beta
      Bspacetime_pred_obs_sites[k].setZero(p, t_new); // Space-time effect ST-beta at observed sites
      Bspacetime_pred_draw[k].setZero(p_new, t_new); // Space-time effect ST-beta
    }

    // Initialize various matrices for predictions
    chol_Corr.resize(nr);
    chol_Corr_pred.resize(nr * nblocks);
    R_cross.resize(nr * nblocks);

    for (int i = 0; i < nr; ++i) {
      double range = allowed_range(i);

      for (int j = 0; j < nblocks; ++j) {
        Eigen::MatrixXd Rcross_ij = (-W_cross_dense[j].array() * (1.0 / range)).exp().matrix();
        R_cross[j*nr + i] = Rcross_ij;
        Eigen::MatrixXd Rpred_ij = (-W_pred_dense[j].array() * (1.0 / range)).exp().matrix();
        Eigen::MatrixXd CC = Rpred_ij - (Rcross_ij.transpose() * Q0_dense[i]) * Rcross_ij;
        Eigen::LLT<Eigen::MatrixXd> chol_CC(CC);
        if (chol_CC.info() != Eigen::Success) {
          Rcpp::stop("Cholesky decomposition failed for prediction covariance matrix.");
        }
        Eigen::MatrixXd LCC = chol_CC.matrixL();
        chol_Corr_pred[j*nr + i] = LCC.sparseView();
      }
      Eigen::LLT<Eigen::MatrixXd> chol_R0(R0[i]);
      if (chol_R0.info() != Eigen::Success) {
        Rcpp::stop("Cholesky decomposition failed for R0 matrix.");
      }
      Eigen::MatrixXd LR0 = chol_R0.matrixL();
      chol_Corr[i] = LR0.sparseView();
    }

  }
  
  Eigen::MatrixXd eta_tilde(p, t);
  if (family == "gaussian") {
    // Gaussian family: log link function
    eta_tilde = Y;
    // if (transfY == "sqrt") {
    //   eta_tilde = eta_tilde.array().sqrt();
    // }
    // if (transfY == "log") {
    //   eta_tilde = eta_tilde.array().log();
    // }
  } else if (family == "poisson") {
    // Poisson family: log link function
    eta_tilde = (0.5 + Y.array()).log();
  } else if (family == "bernoulli") {
    // Bernoulli family: logit link function
    eta_tilde = logit(((Y.array() + 0.5) / 2.0).matrix());
  } else {
    Rcpp::stop("Unsupported family: " + family);
  }

  // Handle NaNs
  std::vector<std::pair<int, int>> nan_indices, valid_indices;
  double mean_Y_valid = 0.0, std_Y_valid = 1.0, sum = 0.0, sq_sum = 0.0;
  
  for (int i = 0; i < eta_tilde.rows(); ++i) {
    for (int j = 0; j < eta_tilde.cols(); ++j) {
      double val = eta_tilde(i, j);
      if (std::isfinite(val)) {
        valid_indices.emplace_back(i, j);
        sum += val;
        sq_sum += val * val;
      } else {
        nan_indices.emplace_back(i, j);
      }
    }
  }
  int n_obs = valid_indices.size();


  if (n_obs > 0) {
    mean_Y_valid = sum / n_obs;
    std_Y_valid = std::sqrt(sq_sum / n_obs - mean_Y_valid * mean_Y_valid);
  } else {
    Rcpp::stop("No valid observations found.");
  }
  
  for (const auto& [i, j] : nan_indices) {
    eta_tilde(i, j) = R::rnorm(mean_Y_valid, std_Y_valid);
    if (family == "poisson") {
      Y(i, j) = R::rpois(std::exp(eta_tilde(i, j)));
    } else if (family == "bernoulli") {
      double pr_y1 = 1.0 - R::pnorm(- eta_tilde(i, j), 0.0, 1.0, true, false);
      Y(i, j) = R::rbinom(1, pr_y1);
    }
  }


  
  // --- Initialization ---
  bool restart = !out_prev_nullable.isNull();
  
  // MCMC parameters
  Eigen::VectorXd rho1_space_draw = Eigen::VectorXd::Constant(ncovx, 0.01);
  Eigen::VectorXd rho1_space_time_draw = Eigen::VectorXd::Constant(ncovx, 0.01);
  Eigen::VectorXd rho2_space_draw = Eigen::VectorXd::Constant(ncovx, std::max(0.5*max_rho2, min_rho2));
  Eigen::VectorXd rho2_space_time_draw = Eigen::VectorXd::Constant(ncovx, std::max(0.5*max_rho2, min_rho2));
  Eigen::VectorXd Q1invdraw_time = Eigen::VectorXd::Constant(ncovx, 10.0); // Precision for T-beta
  double s2_err_mis_draw = 0.01; // Measurement error variance
  
  // Effects: each slice is a matrix
  Eigen::VectorXd Bdraw_vec;
  std::vector<Eigen::MatrixXd> Bdraw(ncovx, Eigen::MatrixXd::Zero(1, 1));       // Constant effect
  std::vector<Eigen::MatrixXd> Btimedraw(ncovx, Eigen::MatrixXd::Zero(1, t));   // Time effect T-beta
  std::vector<Eigen::MatrixXd> Bspacedraw(ncovx, Eigen::MatrixXd::Zero(p, 1));  // Space effect S-beta
  std::vector<Eigen::MatrixXd> Bspacetimedraw(ncovx, Eigen::MatrixXd::Zero(p, t)); // Space-time effect ST-beta
  
  // Z coefficients and covariate contributions
  Eigen::VectorXd gamma_draw; // Will be resized later if ncovz > 0
  Eigen::MatrixXd meanZ; // Contribution of Z covariates
  Eigen::VectorXd Zgamma = Eigen::VectorXd::Zero(p * t);
  Eigen::VectorXd Xbdraw = Eigen::VectorXd::Zero(p * t); // Contribution of X covariates
  
  // other stuff
  Eigen::MatrixXd pswitch_Y = Eigen::MatrixXd::Zero(p, t);
  Eigen::VectorXd eta_tilde_vec = Eigen::Map<Eigen::VectorXd>(eta_tilde.data(), eta_tilde.size());

  // Reshape X into (p*t) x ncovx and add Z if present
  Eigen::MatrixXd X_reg(p * t, ncovz + ncovx);
  if (ncovz > 0) {
    X_reg.leftCols(ncovz) = Z_mat;
  }
  for (int k = 0; k < ncovx; ++k) {
    Eigen::Map<const Eigen::VectorXd> x_vec(X[k].data(), X[k].size());
    X_reg.col(ncovz + k) = x_vec;
  }

  if (restart) {
    Rcpp::Rcout << "Restarting MCMC from previous state..." << std::endl;
    Rcpp::List out_prev = Rcpp::as<Rcpp::List>(out_prev_nullable);
    
    // Load Btimedraw, Bspacedraw, Bspacetimedraw
    {
      Rcpp::NumericVector Btime_array = out_prev["Btimedraw"];
      Rcpp::NumericVector Bspace_array = out_prev["Bspacedraw"];
      Rcpp::NumericVector Bst_array = out_prev["Bspacetimedraw"];
      
      Rcpp::IntegerVector Btime_dims = Btime_array.attr("dim");
      Rcpp::IntegerVector Bspace_dims = Bspace_array.attr("dim");
      Rcpp::IntegerVector Bst_dims = Bst_array.attr("dim");
      
      for (int k = 0; k < ncovx; ++k) {
        Btimedraw[k] = Eigen::Map<const Eigen::MatrixXd>(
          &Btime_array[0] + k * Btime_dims[0] * Btime_dims[1],
                                                          Btime_dims[0], Btime_dims[1]);
        
        Bspacedraw[k] = Eigen::Map<const Eigen::MatrixXd>(
          &Bspace_array[0] + k * Bspace_dims[0] * Bspace_dims[1],
                                                             Bspace_dims[0], Bspace_dims[1]);
        
        Bspacetimedraw[k] = Eigen::Map<const Eigen::MatrixXd>(
          &Bst_array[0] + k * Bst_dims[0] * Bst_dims[1],
                                                    Bst_dims[0], Bst_dims[1]);
      }
    }
    
    // Load rho and Q1inv vectors
    rho1_space_draw = extract_last_col(out_prev["rho1_space"]);
    rho1_space_time_draw = extract_last_col(out_prev["rho1_spacetime"]);
    rho2_space_draw = extract_last_col(out_prev["rho2_space"]);
    rho2_space_time_draw = extract_last_col(out_prev["rho2_spacetime"]);
    Q1invdraw_time = extract_last_col(out_prev["sigma2_Btime"]);
    Q1invdraw_time = 1.0 / Q1invdraw_time.array();

    // Load s2_err_mis_draw
    {
      Rcpp::NumericMatrix s2_mat = out_prev["sigma2"];
      s2_err_mis_draw = s2_mat(0, s2_mat.ncol() - 1);
    }
    
    // Load eta_tilde
    {
      Rcpp::NumericMatrix eta_mat = out_prev["eta_tilde"];
      eta_tilde = Rcpp::as<Eigen::MatrixXd>(eta_mat);
      eta_tilde_vec = Eigen::Map<Eigen::VectorXd>(eta_tilde.data(), eta_tilde.size());
    }
    
    // Load gamma_draw and meanZ if ncovz > 0
    if (ncovz > 0) {
      Rcpp::NumericMatrix G_mat = out_prev["gamma"];
      gamma_draw = extract_last_col(G_mat);
      
      Zgamma = Z_mat * gamma_draw;
      meanZ = Eigen::Map<Eigen::MatrixXd>(Zgamma.data(), p, t);
    }
    
    // Load Bdraw
    {
      Rcpp::NumericVector Bdraw_array = out_prev["Bdraw"];
      Rcpp::IntegerVector Bdraw_dims = Bdraw_array.attr("dim");
	    Bdraw_vec.resize(ncovx);
      for (int k = 0; k < ncovx; ++k) {
        Bdraw[k] = Eigen::Map<const Eigen::MatrixXd>(
          &Bdraw_array[0] + k * Bdraw_dims[0] * Bdraw_dims[1],
                                                          Bdraw_dims[0], Bdraw_dims[1]);
		    Bdraw_vec(k) = Bdraw[k](0,0);
      }
      Xbdraw = X_reg.rightCols(ncovx) * Bdraw_vec;
    }

    // Load phi_ar1_time_draw and phi_ar1_space_time_draw
    if (!random_walk) {
      phi_ar1_time_draw = extract_last_col(out_prev["phi_AR1_time"]);
      phi_ar1_space_time_draw = extract_last_col(out_prev["phi_AR1_spacetime"]);
    }

  } else { // Initial LM fit if not restarting
    const int nbasis = 4;
    Eigen::VectorXd knots = Eigen::VectorXd::LinSpaced(nbasis + 2, 0.0, 1.0);
    knots = knots.segment(1, nbasis);
    Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(t, 0.0, 1.0);
    Eigen::MatrixXd basis_time = kronecker(Eigen::MatrixXd::Identity(p, p), bspline_basis_matrix(times, 3, knots));

    Eigen::MatrixXd X_reg_aug(p * t, basis_time.cols() + X_reg.cols());
    X_reg_aug << X_reg, basis_time;

    // Fit linear model
    Eigen::VectorXd bml = lmfit(X_reg_aug, eta_tilde_vec - offset_vec);
    if (bml.size() != ncovz + ncovx + basis_time.cols()) {
      Rcpp::warning("LM fit returned unexpected number of coefficients. Initializing Bdraw to zero.");
      Bdraw_vec = Eigen::VectorXd::Zero(ncovx);
    } else {
      if (ncovz > 0) {
        gamma_draw = bml.head(ncovz);
        Zgamma = Z_mat * gamma_draw;
        meanZ = Eigen::Map<Eigen::MatrixXd>(Zgamma.data(), p, t);
      }
      Bdraw_vec = bml.head(ncovz+ncovx).tail(ncovx);
      Xbdraw = X_reg.rightCols(ncovx) * Bdraw_vec;
    }
    
    Eigen::MatrixXd BB=(basis_time*bml.tail(basis_time.cols())).reshaped(p, t);
    Eigen::MatrixXd thetay_draw = offset + meanZ + BB;
    for (int k = 0; k < ncovx; ++k) {
      Bdraw[k](0,0) = Bdraw_vec(k);
      thetay_draw += X[k]*Bdraw_vec(k);
    }

    double meanBB = BB.mean();
    Btimedraw[0] = BB.colwise().mean().array() - meanBB;
    Bspacedraw[0] = BB.rowwise().mean().array() - meanBB;
    Bspacetimedraw[0] = BB.array() - Btimedraw[0].replicate(p,1).array() - Bspacedraw[0].replicate(1,t).array() + meanBB;

    Eigen::VectorXd Ht = Eigen::VectorXd::Constant(1, 0.001);
    Eigen::MatrixXd Ctuning = Eigen::MatrixXd::Constant(1, 1, ctuning);
    std::vector<Eigen::MatrixXd> AugData = augmented_data_poisson_lognormal(
      Y, eta_tilde, thetay_draw, Ht, pswitch_Y, Ctuning);
    eta_tilde = AugData[0];
    eta_tilde_vec = Eigen::Map<Eigen::VectorXd>(eta_tilde.data(), eta_tilde.size());
  }
  
  
  
  
  
  // Dimensions
  const int q = 1;
  const int q2 = p;
  const int Tq = t * q;
  const int Tq2 = t * q2;
  
  
  
  // Construct sparse difference matrix H (Tq x Tq)
  Eigen::SparseMatrix<double> H(Tq, Tq);
  std::vector<Eigen::Triplet<double>> triplets_H;
  for (int i = 0; i < Tq; ++i) {
    triplets_H.emplace_back(i, i, 1.0);
    if (i >= q) {
      triplets_H.emplace_back(i, i - q, -1.0);
    }
  }
  H.setFromTriplets(triplets_H.begin(), triplets_H.end());
  
  // Construct sparse difference matrix H2 (Tq2 x Tq2)
  Eigen::SparseMatrix<double> H2(Tq2, Tq2);
  std::vector<Eigen::Triplet<double>> triplets_H2;
  for (int i = 0; i < Tq2; ++i) {
    triplets_H2.emplace_back(i, i, 1.0);
    if (i >= q2) {
      triplets_H2.emplace_back(i, i - q2, -1.0);
    }
  }
  H2.setFromTriplets(triplets_H2.begin(), triplets_H2.end());
  
  
  std::vector<Eigen::SparseMatrix<double>> bigG(ncovx);   // pt x t
  std::vector<Eigen::SparseMatrix<double>> bigG2(ncovx);  // pt x pt
  std::vector<Eigen::SparseMatrix<double>> bigG3(ncovx);  // pt x p
  
  for (int k = 0; k < ncovx; ++k) {
    std::vector<Eigen::Triplet<double>> triplets_G;
    std::vector<Eigen::Triplet<double>> triplets_G2;
    std::vector<Eigen::Triplet<double>> triplets_G3;
    
    for (int i = 0; i < t; ++i) {
      const Eigen::VectorXd& x_col = X[k].col(i);  // p x 1
      
      // bigG: pt x t (each column is stacked x_col)
      for (int j = 0; j < p; ++j) {
        triplets_G.emplace_back(i * p + j, i, x_col(j));
      }
      
      // bigG2: pt x pt (block diagonal of diag(x_col))
      for (int j = 0; j < p; ++j) {
        int idx = i * p + j;
        triplets_G2.emplace_back(idx, idx, x_col(j));
      }
      
      // bigG3: pt x p (each block is diag(x_col))
      for (int j = 0; j < p; ++j) {
        triplets_G3.emplace_back(i * p + j, j, x_col(j));
      }
    }
    
    Eigen::SparseMatrix<double> G(p * t, t);
    G.setFromTriplets(triplets_G.begin(), triplets_G.end());
    bigG[k] = G;
    
    Eigen::SparseMatrix<double> G2(p * t, p * t);
    G2.setFromTriplets(triplets_G2.begin(), triplets_G2.end());
    bigG2[k] = G2;
    
    Eigen::SparseMatrix<double> G3(p * t, p);
    G3.setFromTriplets(triplets_G3.begin(), triplets_G3.end());
    bigG3[k] = G3;
  }
  
  std::vector<Eigen::SparseMatrix<double>> Q1invdraw_space(ncovx, Eigen::SparseMatrix<double>(p, p));
  std::vector<Eigen::SparseMatrix<double>> Q1invdraw_spacetime(ncovx, Eigen::SparseMatrix<double>(p, p));
  std::vector<Eigen::MatrixXd> Q1invdraw_space_dense(ncovx, Eigen::MatrixXd::Zero(p, p));
  std::vector<Eigen::MatrixXd> Q1invdraw_spacetime_dense(ncovx, Eigen::MatrixXd::Zero(p, p));
  int default_q0_index = static_cast<int>(std::floor(15.0 / 2.0 - 1.0));
  int q0_index_space = 0, q0_index_spacetime = 0;

  for (int k = 0; k < ncovx; ++k) {
    if (rho1_space_draw(k) == 0.0) {
      Rcpp::stop("Initial rho1_space_draw is zero.");
    }
    if (rho1_space_time_draw(k) == 0.0) {
      Rcpp::stop("Initial rho1_space_time_draw is zero.");
    }
    if (restart) {
      q0_index_space = find_index(rho2_space_draw(k), allowed_range);
      q0_index_spacetime = find_index(rho2_space_time_draw(k), allowed_range);
      if (!point_referenced) {
        Q1invdraw_space[k] = (1.0 / rho1_space_draw(k)) * Q0[q0_index_space];
        Q1invdraw_spacetime[k] = (1.0 / rho1_space_time_draw(k)) * Q0[q0_index_spacetime];
      } else {
        Q1invdraw_space_dense[k] = (1.0 / rho1_space_draw(k)) * Q0_dense[q0_index_space];
        Q1invdraw_spacetime_dense[k] = (1.0 / rho1_space_time_draw(k)) * Q0_dense[q0_index_spacetime];
      }
      
    } else {
      if (!point_referenced) {
        Q1invdraw_space[k] = (1.0 / rho1_space_draw(k)) * Q0[default_q0_index];
        Q1invdraw_spacetime[k] = (1.0 / rho1_space_time_draw(k)) * Q0[default_q0_index];
      } else {
        Q1invdraw_space_dense[k] = (1.0 / rho1_space_draw(k)) * Q0_dense[default_q0_index];
        Q1invdraw_spacetime_dense[k] = (1.0 / rho1_space_time_draw(k)) * Q0_dense[default_q0_index];
      }
    }
     
  }
  
  
  // --- Storage Initialization ---
  if (nrep % thin != 0) {
    Rcpp::warning("nrep is not a multiple of thin, some iterations will be discarded.");
  }
  int collections = nrep / thin;
  int MCMC_samples = nrep + nburn;
  int collect_count = 0;
  
  // Storage matrices
  Rcpp::List out_results;
  Eigen::MatrixXd S2_err_mis_(1, collections);
  Eigen::MatrixXd RHO1_space_(ncovx, collections);
  Eigen::MatrixXd RHO1_space_time_(ncovx, collections);
  Eigen::MatrixXd RHO2_space_(ncovx, collections);
  Eigen::MatrixXd RHO2_space_time_(ncovx, collections);
  Eigen::MatrixXd S2_Btime_(ncovx, collections);
  Eigen::MatrixXd G_out;  // Will be resized only if ncovz > 0
  Eigen::MatrixXd PHI_AR1_time_; // Will be resized only if !random_walk
  Eigen::MatrixXd PHI_AR1_space_time_; // Will be resized only if !random_walk
	std::vector<Eigen::MatrixXd> store_llike(collections, Eigen::MatrixXd::Zero(p, t));
  
  if (ncovz > 0) {
    G_out.resize(ncovz, collections);
  }

  if (!random_walk) {
    PHI_AR1_time_.resize(ncovx, collections);
    PHI_AR1_space_time_.resize(ncovx, collections);
  }
  
  std::vector<Eigen::MatrixXd> YFITTED_out(collections, Eigen::MatrixXd::Zero(p, t));
  Eigen::MatrixXd RMSE_(1, collections);
  Eigen::MatrixXd MAE_(1, collections);
  Eigen::VectorXd chi_sq_fitted_ = Eigen::VectorXd::Zero(collections);
  Eigen::VectorXd chi_sq_obs_ = Eigen::VectorXd::Zero(collections);
  
  
  // Averaging structures
  Eigen::MatrixXd Yfitted_mean = Eigen::MatrixXd::Zero(p, t);
  Eigen::MatrixXd Yfitted2_mean = Eigen::MatrixXd::Zero(p, t); // Sum of squares for variance
  Eigen::MatrixXd Eta_tilde_mean = Eigen::MatrixXd::Zero(p, t);
  Eigen::MatrixXd thetay_mean = Eigen::MatrixXd::Zero(p, t);
  Eigen::MatrixXd meanZmean = Eigen::MatrixXd::Zero(p, t);
  Eigen::MatrixXd meanY1mean = Eigen::MatrixXd::Zero(p, t);
  
  std::vector<Eigen::MatrixXd> B_postmean(ncovx, Eigen::MatrixXd::Zero(1, 1));
  std::vector<Eigen::MatrixXd> B2_postmean(ncovx, Eigen::MatrixXd::Zero(1, 1));
  std::vector<Eigen::MatrixXd> Btime_postmean(ncovx, Eigen::MatrixXd::Zero(1, t));
  std::vector<Eigen::MatrixXd> Btime2_postmean(ncovx, Eigen::MatrixXd::Zero(1, t));
  std::vector<Eigen::MatrixXd> Bspace_postmean(ncovx, Eigen::MatrixXd::Zero(p, 1));
  std::vector<Eigen::MatrixXd> Bspace2_postmean(ncovx, Eigen::MatrixXd::Zero(p, 1));
  std::vector<Eigen::MatrixXd> Bspacetime_postmean(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> Bspacetime2_postmean(ncovx, Eigen::MatrixXd::Zero(p, t));
  
  std::vector<Eigen::MatrixXd> B2_c_t(ncovx, Eigen::MatrixXd::Zero(1, t));
  std::vector<Eigen::MatrixXd> B2_c_s(ncovx, Eigen::MatrixXd::Zero(p, 1));
  std::vector<Eigen::MatrixXd> B2_c_t_s_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_c_t_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_t_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_t_s(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_t_s_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_c_t_s(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_s_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> B2_c_s_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  
  std::vector<Eigen::MatrixXd> E_t_s(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> E_t_st(ncovx, Eigen::MatrixXd::Zero(p, t));
  std::vector<Eigen::MatrixXd> E_s_st(ncovx, Eigen::MatrixXd::Zero(p, t));

  // For predictions
  Eigen::MatrixXd Ypred_mean, Ypred2_mean;
  std::vector<Eigen::MatrixXd> YPRED_out(collections);
  std::vector<Eigen::MatrixXd> Btime_pred_postmean(ncovx);
  std::vector<Eigen::MatrixXd> Btime_pred2_postmean(ncovx);
  std::vector<Eigen::MatrixXd> Bspace_pred_postmean(ncovx);
  std::vector<Eigen::MatrixXd> Bspace_pred2_postmean(ncovx);
  std::vector<Eigen::MatrixXd> Bspacetime_pred_postmean(ncovx);
  std::vector<Eigen::MatrixXd> Bspacetime_pred2_postmean(ncovx);
  std::vector<Eigen::MatrixXd> B_pred2_c_t_s_st(ncovx);
  if (X_pred_nullable.isNotNull()) {
    Ypred_mean.setZero(p_new, t_new);
    Ypred2_mean.setZero(p_new, t_new); // Sum of squares for variance
    if (keepY) {
      for (int ii = 0; ii < collections; ++ii) {
        YPRED_out[ii].setZero(p_new, t_new);
      }
    }
    
    for (int k = 0; k < ncovx; ++k) {
      Btime_pred_postmean[k].setZero(1, h_ahead); // Time effect T-beta
      Btime_pred2_postmean[k].setZero(1, h_ahead); // Time effect T-beta squared
      Bspace_pred_postmean[k].setZero(p_new, 1); // Space effect S-beta
      Bspace_pred2_postmean[k].setZero(p_new, 1); // Space effect S-beta squared
      Bspacetime_pred_postmean[k].setZero(p_new, t_new); // Space-time effect ST-beta
      Bspacetime_pred2_postmean[k].setZero(p_new, t_new); // Space-time effect ST-beta squared
      B_pred2_c_t_s_st[k].setZero(p_new, t_new);
    }
  }
  
  // Diagnostics
  Eigen::VectorXd pvalue_ResgrRespred_sum = Eigen::VectorXd::Zero(valid_indices.size());
  Eigen::VectorXd pvalue_YgrYhat_sum = Eigen::VectorXd::Zero(valid_indices.size());
  double chisq_count = 0.0;
  double quant = 0.95;
  Eigen::VectorXd p95_obs_ = cpp_prctile(Y, quant);
  Eigen::VectorXd percentile95_sum = Eigen::VectorXd::Zero(p);
  double store_CRPS_1_sum = 0.0;
  double store_CRPS_2_sum = 0.0;
  
  
  
  // --- MCMC Loop ---
  Rcpp::Rcout << "Starting MCMC (" << MCMC_samples << " iterations)..." << std::endl;
  
  for (int irep = 0; irep < MCMC_samples; ++irep) {
    // Print progress
    if ((irep + 1) % print_interval == 0) {
      Rcpp::Rcout << "Iteration: " << irep + 1 << " / " << MCMC_samples << std::endl;
      Rcpp::checkUserInterrupt(); // Allow user to interrupt
    }
    
    // Step I.a: Sampling T-beta (time effects)
    // Compute residual y_k
    Eigen::VectorXd y2_vec = eta_tilde_vec - Zgamma - offset_vec - Xbdraw;
    for (int k = 0; k < ncovx; ++k) {
      y2_vec -= bigG2[k] * Eigen::Map<Eigen::VectorXd>(Bspacetimedraw[k].data(), Bspacetimedraw[k].size());
      y2_vec -= bigG3[k] * Bspacedraw[k].col(0);
    }

    for (int k = 0; k < ncovx; ++k) {
      Eigen::VectorXd y_k = y2_vec;
      for (int k2 = 0; k2 < ncovx; ++k2) {
        if (k == k2) continue;
        y_k -= bigG[k2] * Btimedraw[k2].row(0).transpose();
      }
      
      // Construct prior precision matrix K = H^T * invS * H
      Eigen::SparseMatrix<double> invS_diag(Tq, Tq);
      std::vector<Eigen::Triplet<double>> invS_triplets;
      for (int i = 0; i < Tq; ++i) {
        invS_triplets.emplace_back(i, i, Q1invdraw_time(k));
      }
      invS_triplets[0] = Eigen::Triplet<double>(0, 0, 1.0 / V_beta_0(k));  // diffuse prior
      invS_diag.setFromTriplets(invS_triplets.begin(), invS_triplets.end());
            
      // Sample from posterior
      Eigen::MatrixXd bb = posterior_beta(H.transpose() * invS_diag * H, bigG[k], s2_err_mis_draw, y_k); // T x 1
      
      // Reshape and center
      Eigen::MatrixXd Bdrawc = Eigen::Map<Eigen::MatrixXd>(bb.data(), 1, t);
      Btimedraw[k] = Bdrawc.row(0).array() - Bdrawc.row(0).mean();
      
    }
    
    
    
    // Step I.b: Sampling ST-beta (space-time effects)
    // Compute residual y_k
    y2_vec = eta_tilde_vec - Zgamma - offset_vec - Xbdraw;
    for (int k = 0; k < ncovx; ++k) {
      y2_vec -= bigG[k] * Btimedraw[k].row(0).transpose();
      y2_vec -= bigG3[k] * Bspacedraw[k].col(0);
    }
      
    for (int k = 0; k < ncovx; ++k) {
      if (ST(k) == 1.0) {
        Eigen::VectorXd y_k = y2_vec;
        for (int k2 = 0; k2 < ncovx; ++k2) {
          if (k == k2) continue;
          y_k -= bigG2[k2] * Eigen::Map<const Eigen::VectorXd>(Bspacetimedraw[k2].data(), p * t);
        }
        
        // Construct block-diagonal prior precision matrix invS_diag
        Eigen::SparseMatrix<double> invS_diag(p * t, p * t);
        // std::vector<Eigen::Triplet<double>> triplets;
        // for (int i = 0; i < p; ++i) {
        //   triplets.emplace_back(i, i, 1.0 / V_beta_0);  // prior for beta_0
        // }
        // for (int i = 1; i < t; ++i) {
        //   for (int j = 0; j < Q1invdraw_spacetime[k].outerSize(); ++j) {
        //     for (Eigen::SparseMatrix<double>::InnerIterator it(Q1invdraw_spacetime[k], j); it; ++it) {
        //       int row = i * p + it.row();
        //       int col = i * p + it.col();
        //       triplets.emplace_back(row, col, it.value());
        //     }
        //   }
        // }
        // invS_diag.setFromTriplets(triplets.begin(), triplets.end());
        if (!point_referenced) {
          invS_diag = createBlockDiagonal(Q1invdraw_spacetime[k], 1.0 / V_beta_0(k), t);
        } else {
          invS_diag = createBlockDiagonal_dense(Q1invdraw_spacetime_dense[k], 1.0 / V_beta_0(k), t);
        }
        
        // Sample from posterior
        Eigen::MatrixXd bb = posterior_beta(H2.transpose() * invS_diag * H2, bigG2[k], s2_err_mis_draw, y_k); // pt x 1
        // Reshape and apply sum-to-zero constraints
        Eigen::MatrixXd Bdrawc = Eigen::Map<Eigen::MatrixXd>(bb.data(), p, t);
        Bdrawc = Bdrawc.colwise() - Bdrawc.rowwise().mean();  // center columns (space)
        Bdrawc = Bdrawc.rowwise() - Bdrawc.colwise().mean();  // center rows (time)
        Bspacetimedraw[k] = Bdrawc;
      }
    }
    
    
    // Step I.c: Sampling S-beta (spatial effects)
    y2_vec = eta_tilde_vec - Zgamma - offset_vec - Xbdraw;
    for (int k = 0; k < ncovx; ++k) {
      y2_vec -= bigG[k] * Btimedraw[k].row(0).transpose();
      y2_vec -= bigG2[k] * Eigen::Map<Eigen::VectorXd>(Bspacetimedraw[k].data(), Bspacetimedraw[k].size());
    }
    
    for (int k = 0; k < ncovx; ++k) {
      // Compute residual y_k
      Eigen::VectorXd y_k = y2_vec;
      for (int k2 = 0; k2 < ncovx; ++k2) {
        if (k == k2) continue;
        y_k -= bigG3[k2] * Bspacedraw[k2];
      }
      
      // Sample from posterior
      Eigen::MatrixXd bb;
      if (!point_referenced) {
        bb = posterior_beta(Q1invdraw_space[k], bigG3[k], s2_err_mis_draw, y_k); // p x 1
      } else {
        bb = posterior_beta_denseK(Q1invdraw_space_dense[k], bigG3[k], s2_err_mis_draw, y_k); // p x 1
      }
      
      
      // Center (sum-to-zero constraint)
      Eigen::VectorXd centered = bb.col(0).array() - bb.col(0).mean();
      Bspacedraw[k] = centered;
    }
    
    
    // Step I.d: Sampling constant coefficients (C-beta and Z-gamma)
    // Compute residual y2 = eta_tilde - T*beta_t - S*beta_s - ST*beta_st
    Eigen::VectorXd y2_reg = eta_tilde_vec - offset_vec;
    for (int k = 0; k < ncovx; ++k) {
      y2_reg -= bigG[k] * Btimedraw[k].transpose();
      y2_reg -= bigG2[k] * Eigen::Map<const Eigen::VectorXd>(Bspacetimedraw[k].data(), p * t);
      y2_reg -= bigG3[k] * Bspacedraw[k];
    }
        
    // Prior precision (diagonal, small values for diffuse prior)
    Eigen::MatrixXd K_reg = Eigen::MatrixXd::Identity(ncovz + ncovx, ncovz + ncovx) * (1.0 / V_gamma);
    
    // Posterior precision and mean
    Eigen::MatrixXd GinvOmega11_reg = X_reg.transpose() * (1.0 / s2_err_mis_draw);
    Eigen::MatrixXd GinvOmega11G_reg = GinvOmega11_reg * X_reg;
    Eigen::MatrixXd invP_reg = K_reg + GinvOmega11G_reg;
    invP_reg = 0.5 * (invP_reg + invP_reg.transpose());  // Ensure symmetry
    
    Eigen::VectorXd tmp_reg = GinvOmega11_reg * y2_reg;
    
    // Sample from posterior
    Eigen::MatrixXd P_reg = invP_reg.inverse();
    Eigen::LLT<Eigen::MatrixXd> llt(P_reg);
    Eigen::MatrixXd L_reg = llt.matrixL();
    Eigen::VectorXd gamma_beta_draw = P_reg * tmp_reg;
    gamma_beta_draw += L_reg * randn_vector(gamma_beta_draw.size());
	
    
    
    // Update gamma_draw and Bdraw
    if (ncovz > 0) {
      gamma_draw = gamma_beta_draw.head(ncovz);
      Zgamma = Z_mat * gamma_draw;
      meanZ = Eigen::Map<Eigen::MatrixXd>(Zgamma.data(), p, t);
    }
    Eigen::VectorXd Bdraw_vec = gamma_beta_draw.tail(ncovx);
    for (int k = 0; k < ncovx; ++k) {
      Bdraw[k](0,0) = Bdraw_vec(k);
    }
    Xbdraw = X_reg.rightCols(ncovx) * Bdraw_vec;
    
    
    // Step I.e: Computing the current mean component from X
    Eigen::MatrixXd current_meanY1 = Eigen::MatrixXd::Zero(p, t);
    
    for (int k = 0; k < ncovx; ++k) {
      Eigen::MatrixXd B_c_full = Eigen::MatrixXd::Constant(p, t, Bdraw_vec(k));
      Eigen::MatrixXd B_t_full = Btimedraw[k].colwise().replicate(p);  // 1 x t → p x t
      Eigen::MatrixXd B_s_full = Bspacedraw[k].rowwise().replicate(t); // p x 1 → p x t
      const Eigen::MatrixXd& B_st_full = Bspacetimedraw[k];
      Eigen::ArrayXXd appoX_B = X[k].array() * (B_c_full + B_t_full + B_s_full + B_st_full).array();
      current_meanY1 += appoX_B.matrix();
    }
    
    
    // Step II: Sampling variances of state vectors
    for (int k = 0; k < ncovx; ++k) {
      // --- TIME (T-beta) ---
      const Eigen::MatrixXd& Btime_k = Btimedraw[k];  // 1 x t
      Eigen::VectorXd e2(t);
      e2(0) = Btime_k(0, 0) - phi_ar1_time_draw(k) * Btime_k(0, 0);
      e2.segment(1, t - 1) = Btime_k.row(0).segment(1, t - 1) - phi_ar1_time_draw(k) * Btime_k.row(0).segment(0, t - 1);
      double newnu2 = a_inn_time(k) + static_cast<double>(t - 1) / 2.0;
      double newS2 = b_inn_time(k) + 0.5 * e2.squaredNorm();
      Q1invdraw_time(k) = R::rgamma(newnu2, 1.0 / newS2);
      
      // --- SPACE-TIME (ST-beta) ---
      const Eigen::MatrixXd& Bst_k = Bspacetimedraw[k];  // p x t
      Eigen::MatrixXd Btemp = Eigen::MatrixXd::Zero(p, t);
      Btemp.col(0) = Bst_k.col(0) - phi_ar1_space_time_draw(k) * Bst_k.col(0);
      Btemp.block(0, 1, p, t - 1) = Bst_k.block(0, 1, p, t - 1) - phi_ar1_space_time_draw(k) * Bst_k.block(0, 0, p, t - 1);
      
      if (ST(k) == 1.0) {
        if (!point_referenced) {
          Rcpp::List STC_list = MH_spatial_correlation_CAR_fast(Btemp, rho1_space_time_draw(k), Q0, allowed_range, logdet_Q0_spacetime);
          rho2_space_time_draw(k) = STC_list["range_draw"];
          q0_index_spacetime = STC_list["range_index"];
          int pstar = (rho2_space_time_draw(k) == 1.0) ? (p - 1) : p;
          rho1_space_time_draw(k) = posterior_conditional_variance(Btemp, Q0[q0_index_spacetime], 
            a_rho1st(k), b_rho1st(k), pstar, t - 1);
          Q1invdraw_spacetime[k] = (1.0 / rho1_space_time_draw(k)) * Q0[q0_index_spacetime];
        } else {
          Rcpp::List STC_list = MH_spatial_correlation_EXP_fast(Btemp, rho1_space_time_draw(k), Q0_dense, allowed_range, logdet_Q0_spacetime);
          rho2_space_time_draw(k) = STC_list["range_draw"];
          q0_index_spacetime = STC_list["range_index"];
          rho1_space_time_draw(k) = posterior_conditional_variance_dense(Btemp, Q0_dense[q0_index_spacetime], 
            a_rho1st(k), b_rho1st(k), p, t - 1);
          Q1invdraw_spacetime_dense[k] = (1.0 / rho1_space_time_draw(k)) * Q0_dense[q0_index_spacetime];
        }
      } else {
        rho1_space_time_draw(k) = 1e-9;
        if (!point_referenced) {
          Q1invdraw_spacetime[k].resize(p, p);
          Q1invdraw_spacetime[k].setZero();
        } else {
          Q1invdraw_spacetime_dense[k].resize(p, p);
          Q1invdraw_spacetime_dense[k].setZero();
        }
      }
      
      // --- SPACE (S-beta) ---
      const Eigen::MatrixXd& Bs_k = Bspacedraw[k];  // p x 1
      if (!point_referenced) {
        Rcpp::List SC_list = MH_spatial_correlation_CAR_fast(Bs_k, rho1_space_draw(k), Q0, allowed_range, logdet_Q0_space);
        rho2_space_draw(k) = SC_list["range_draw"];
        q0_index_space = SC_list["range_index"];
        int pstar_s = (rho2_space_draw(k) == 1.0) ? (p - 1) : p;
        rho1_space_draw(k) = posterior_conditional_variance(Bs_k, Q0[q0_index_space], a_rho1s(k), b_rho1s(k), pstar_s, 1);
        Q1invdraw_space[k] = (1.0 / rho1_space_draw(k)) * Q0[q0_index_space];
      } else {
        Rcpp::List SC_list = MH_spatial_correlation_EXP_fast(Bs_k, rho1_space_draw(k), Q0_dense, allowed_range, logdet_Q0_space);
        rho2_space_draw(k) = SC_list["range_draw"];
        q0_index_space = SC_list["range_index"];
        rho1_space_draw(k) = posterior_conditional_variance_dense(Bs_k, Q0_dense[q0_index_space], a_rho1s(k), b_rho1s(k), p, 1);
        Q1invdraw_space_dense[k] = (1.0 / rho1_space_draw(k)) * Q0_dense[q0_index_space];
      }
    }
    
    
    // Step III: Sampling the measurement error variance
    Eigen::MatrixXd thetay_draw = current_meanY1 + meanZ + offset;
    Eigen::MatrixXd yhat = eta_tilde - thetay_draw;
    
    double precision_draw = 1.0;
    if (family == "gaussian" || family == "poisson") {
      double sse_2 = yhat.array().square().sum();
      double g1_pos = a_s2 + static_cast<double>(p * t) / 2.0;
      double g2_pos = b_s2 + 0.5 * sse_2;
      
      precision_draw = R::rgamma(g1_pos, 1.0 / g2_pos);
    }
    
    s2_err_mis_draw = 1.0 / precision_draw;
    double current_sd = std::sqrt(s2_err_mis_draw);
    auto norm_mcmc = [&] (double) {return R::rnorm(0.0, current_sd);};


    // Step IV: sample latent process if non-gaussian outcome
    if (family == "poisson") {
      Eigen::VectorXd Ht = Eigen::VectorXd::Constant(1, s2_err_mis_draw);
      Eigen::MatrixXd Ctuning = Eigen::MatrixXd::Constant(1, 1, ctuning);
      std::vector<Eigen::MatrixXd> AugData = augmented_data_poisson_lognormal(
        Y, eta_tilde, thetay_draw, Ht, pswitch_Y, Ctuning);
      eta_tilde = AugData[0];
      pswitch_Y = AugData[1];
    } else if (family == "bernoulli") {
      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < t; ++j) {
          if (Y(i, j) == 0.0) {
            Rcpp::NumericVector trunorm_draw = rtruncnorm(1, thetay_draw(i, j), current_sd, std::numeric_limits<double>::lowest(), -1e-5);
            eta_tilde(i, j) = trunorm_draw[0];
          } else {
            Rcpp::NumericVector trunorm_draw = rtruncnorm(1, thetay_draw(i, j), current_sd, 0.0, std::numeric_limits<double>::max());
            eta_tilde(i, j) = trunorm_draw[0];
          }
        }
      }
    }
    

    // Step V: sample temporal autoregressive coefficients
    if (!random_walk) {
      for (int k = 0; k < ncovx; ++k) {
        // TIME
        const Eigen::MatrixXd& Btime_k = Btimedraw[k];  // 1 x t
        Eigen::VectorXd X2(t);
        X2(0) = Btime_k(0, 0);
        X2.segment(1, t-1) = Btime_k.row(0).segment(0, t - 1).transpose();
        double Prec_prior_phi_t = 1.0;
        double Prec_posterior_phi_t = Prec_prior_phi_t + (X2 * Q1invdraw_time(k)).dot(X2);
        double Mean_posterior_phi_t = (X2 * Q1invdraw_time(k)).dot(Btime_k.row(0).transpose()) / Prec_posterior_phi_t;
        Rcpp::NumericVector phi_t_draw = rtruncnorm(1, Mean_posterior_phi_t, std::sqrt(1.0/Prec_posterior_phi_t), -1.0+1e-5, 1.0-1e-5);
        phi_ar1_time_draw(k) = phi_t_draw[0];

        if (ST(k) == 1.0) {
          // SPACE-TIME
          const Eigen::MatrixXd& Bst_k = Bspacetimedraw[k];  // p x t
          double Prec_prior_phi_st = 1.0;
          double Prec_posterior_phi_st = 0.0;
          // double Mean_posterior_phi_st = 0.0;
          Prec_posterior_phi_st += Bst_k.col(0).transpose() * Q1invdraw_spacetime_dense[k] * Bst_k.col(0);
          // Mean_posterior_phi_st += Bst_k.col(0).transpose() * Q1invdraw_spacetime[k] * Bst_k.col(0);
          double Mean_posterior_phi_st = Prec_posterior_phi_st;
          for (int i = 1; i < t; ++i) {
            Prec_posterior_phi_st += (Bst_k.col(i-1).transpose() * Q1invdraw_spacetime_dense[k] * Bst_k.col(i-1))(0, 0);
            Mean_posterior_phi_st += (Bst_k.col(i-1).transpose() * Q1invdraw_spacetime_dense[k] * Bst_k.col(i))(0, 0);
          }
          Prec_posterior_phi_st += Prec_prior_phi_st;
          Mean_posterior_phi_st /= Prec_posterior_phi_st;
          Rcpp::NumericVector phi_st_draw = rtruncnorm(1, Mean_posterior_phi_st, std::sqrt(1.0/Prec_posterior_phi_st), -1.0+1e-5, 1.0-1e-5);
          phi_ar1_space_time_draw(k) = phi_st_draw[0];
        }
      }
    }
    
    
    // Step VI: Handling missing data (imputation)
    // Generate predicted Y ~ N(thetay_draw, s2_err_mis_draw * I)
    Eigen::MatrixXd Yfitted(p, t);
    if (family == "gaussian") {
      Yfitted = Eigen::MatrixXd::NullaryExpr(p, t, norm_mcmc);
      Yfitted += thetay_draw;
      // Impute missing values in eta_tilde
      for (const auto& [i, j] : nan_indices) {
        eta_tilde(i, j) = Yfitted(i, j);
      }
    } else if (family == "poisson") {
      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < t; ++j) {
          Yfitted(i, j) = R::rpois(std::exp(eta_tilde(i, j)));
        }
      }
      // Impute missing values in Y
      for (const auto& [i, j] : nan_indices) {
        Y(i, j) = Yfitted(i, j);
      }
    } else if (family == "bernoulli") {
      // Eigen::MatrixXd pr_y1(p, t); // Pr (Y = 1 | parameters)
      for (int i = 0; i < p; ++i) {
        for (int j = 0; j < t; ++j) {
          double pr_y1 = 1.0 - R::pnorm(- thetay_draw(i, j), 0.0, 1.0, true, false);
          Yfitted(i, j) = R::rbinom(1, pr_y1);
        }
      }
      // Impute missing values in Y
      for (const auto& [i, j] : nan_indices) {
        Y(i, j) = Yfitted(i, j);
      }
    }
    
    eta_tilde_vec = Eigen::Map<Eigen::VectorXd>(eta_tilde.data(), eta_tilde.size());
    
    // --- Store results post-burn-in and thinning ---
    if (irep >= nburn && (irep - nburn + 1) % thin == 0) {
      // Store posterior samples
      S2_err_mis_(0, collect_count) = s2_err_mis_draw;
      
      for (int k = 0; k < ncovx; ++k) {
        RHO1_space_(k, collect_count) = rho1_space_draw(k);
        RHO1_space_time_(k, collect_count) = rho1_space_time_draw(k);
        RHO2_space_(k, collect_count) = rho2_space_draw(k);
        RHO2_space_time_(k, collect_count) = rho2_space_time_draw(k);
        S2_Btime_(k, collect_count) = 1.0 / Q1invdraw_time(k);
        if (!random_walk) {
          PHI_AR1_time_(k, collect_count) = phi_ar1_time_draw(k);
          PHI_AR1_space_time_(k, collect_count) = phi_ar1_space_time_draw(k);
        }
      }
      
      if (ncovz > 0) {
        for (int j = 0; j < ncovz; ++j) {
          G_out(j, collect_count) = gamma_draw(j);
        }
      }

      if (X_pred_nullable.isNotNull()) {
        std::vector<Eigen::MatrixXd> BBtimedraw(ncovx, Eigen::MatrixXd::Zero(1, t_new));
        for (int k = 0; k < ncovx; ++k) {
          q0_index_space = find_index(rho2_space_draw(k), allowed_range);
          if (ST(k) == 1.0) {
            q0_index_spacetime = find_index(rho2_space_time_draw(k), allowed_range);
          }
          // TIME PREDICTIONS
          BBtimedraw[k].leftCols(t) = Btimedraw[k];
          Bspacetime_pred_obs_sites[k].leftCols(t) = Bspacetimedraw[k]; // Copy existing data
          if (h_ahead > 0) {
            double Btime_pred_mean_tph = phi_ar1_time_draw(k) * Btimedraw[k](0, t - 1);
            for (int h = 0; h < h_ahead; ++h) {
              Btime_pred_draw[k](0, h) = Btime_pred_mean_tph + std::sqrt(1.0 / Q1invdraw_time(k)) * R::rnorm(0.0, 1.0);
              // Update mean for the next prediction step.
              Btime_pred_mean_tph = phi_ar1_time_draw(k) * Btime_pred_draw[k](0, h);
              if (ST(k) == 1.0) {
                Bspacetime_pred_obs_sites[k].col(t + h) =
                  phi_ar1_space_time_draw(k) * Bspacetime_pred_obs_sites[k].col(t + h - 1) +
                  std::sqrt(rho1_space_time_draw(k)) * chol_Corr[q0_index_spacetime] * randn_vector(p);
              }
              
            }
            BBtimedraw[k].rightCols(h_ahead) = Btime_pred_draw[k];
          }

          // ----- PRECOMPUTAZIONI condivise per il covariato k -----
          // Indici range già calcolati poco sopra:
          //   q0_index_space, q0_index_spacetime
          // Dati per lo spazio:
          Eigen::VectorXd QtB_space = Q1invdraw_space_dense[k] * Bspacedraw[k]; // p x 1

          // Dati per lo spazio-tempo: U_t[it] = Q1inv_st_dense[k] * (Bst_k.col(i+1) - phi * Bst_k.col(i))
          std::vector<Eigen::VectorXd> U_t(t_new);
          {
            Eigen::MatrixXd Bst_k_full(p, 1 + t_new);
            Bst_k_full.col(0) = Bspacetimedraw[k].col(0);
            Bst_k_full.rightCols(t_new) = Bspacetime_pred_obs_sites[k];

            for (int i = 0; i < t_new; ++i) {
              Eigen::VectorXd delta_i = Bst_k_full.col(i+1) - phi_ar1_space_time_draw(k) * Bst_k_full.col(i);
              U_t[i] = Q1invdraw_spacetime_dense[k] * delta_i; // p x 1
            }
          }


          // SPACE PREDICTIONS (INTERPOLATION)
          {
            // Nota: se vuoi controllare i thread da R: RcppParallel::setThreadOptions(numThreads = ...)
            // Semina deterministica per avere riproducibilità cross-thread (puoi usare collect_count, k, ecc.)
            std::uint64_t base_seed = 0x9E3779B97F4A7C15ULL
                                    ^ static_cast<std::uint64_t>(k) * 0xBF58476D1CE4E5B9ULL
                                    ^ static_cast<std::uint64_t>(collect_count) * 0x94D049BB133111EBULL;

            SpacePredWorker worker(
                nr,
                q0_index_space,
                q0_index_spacetime,
                rho1_space_draw(k),
                rho1_space_time_draw(k),
                phi_ar1_space_time_draw(k),
                t_new,
                blocks_indices,
                R_cross,
                chol_Corr_pred,
                QtB_space,
                U_t,
                W_pred_dense,               // non usato qui, ma lasciato per futura estensione
                ST(k),
                Bspace_pred_draw[k],
                Bspacetime_pred_draw[k],
                base_seed
            );

            // Lancia il parallelFor sui blocchi
            RcppParallel::parallelFor(0, nblocks, worker);
          }

        }

        Eigen::MatrixXd meanY1_pred = Eigen::MatrixXd::Zero(p_new, t_new);
        for (int k = 0; k < ncovx; ++k) {
          Eigen::MatrixXd B_c_pred_full = Eigen::MatrixXd::Constant(p_new, t_new, Bdraw_vec(k));
          Eigen::MatrixXd B_t_pred_full = BBtimedraw[k].colwise().replicate(p_new);
          Eigen::MatrixXd B_s_pred_full = Bspace_pred_draw[k].rowwise().replicate(t_new);
          const Eigen::MatrixXd& B_st_pred_full = Bspacetime_pred_draw[k];
          Eigen::ArrayXXd appoX_Bpred_ = X_pred[k].array() * (B_c_pred_full + B_t_pred_full + B_s_pred_full + B_st_pred_full).array();
          meanY1_pred += appoX_Bpred_.matrix();
        }

        Eigen::MatrixXd meanZ_pred = Eigen::MatrixXd::Zero(p_new, t_new);
        if (ncovz > 0) {
          Eigen::VectorXd Zgamma_pred = Z_pred_mat * gamma_draw;
          meanZ_pred = Eigen::Map<const Eigen::MatrixXd>(Zgamma_pred.data(), p_new, t_new);
        }
        Eigen::MatrixXd Ypred(p_new, t_new), eta_tilde_pred, thetay_pred = meanY1_pred + meanZ_pred + offset_pred;
        if (family == "gaussian") {
          Ypred = Eigen::MatrixXd::NullaryExpr(p_new, t_new, norm_mcmc);
          Ypred += thetay_pred;
        } else if (family == "poisson") {
          eta_tilde_pred = Eigen::MatrixXd::NullaryExpr(p_new, t_new, norm_mcmc);
          eta_tilde_pred += thetay_pred;
          for (int i = 0; i < p_new; ++i) {
            for (int j = 0; j < t_new; ++j) {
              Ypred(i, j) = R::rpois(std::exp(eta_tilde_pred(i, j)));
            }
          }
        } else if (family == "bernoulli") {
          // Eigen::MatrixXd pr_ypred1(p_new, t_new); // Pr (Ypred = 1 | Y, parameters)
          for (int i = 0; i < p_new; ++i) {
            for (int j = 0; j < t_new; ++j) {
              double pr_ypred1 = 1.0 - R::pnorm(- thetay_pred(i, j), 0.0, 1.0, true, false);
              Ypred(i, j) = R::rbinom(1, pr_ypred1);
            }
          }
        }

        // Update averages on predictions
        Ypred_mean += Ypred;
        Ypred2_mean += Ypred.array().square().matrix();
        for (int k = 0; k < ncovx; ++k) {
          Btime_pred_postmean[k] += Btime_pred_draw[k];
          Btime_pred2_postmean[k] += Btime_pred_draw[k].array().square().matrix();
          Bspacetime_pred_postmean[k] += Bspacetime_pred_draw[k];
          Bspacetime_pred2_postmean[k] += Bspacetime_pred_draw[k].array().square().matrix();
          Bspace_pred_postmean[k] += Bspace_pred_draw[k];
          Bspace_pred2_postmean[k] += Bspace_pred_draw[k].array().square().matrix();
          
          Eigen::MatrixXd Bpred_c_t_s_st = Eigen::MatrixXd::Ones(p_new, t_new) * Bdraw_vec(k)
            + BBtimedraw[k].colwise().replicate(p_new)
            + Bspace_pred_draw[k].rowwise().replicate(t_new)
            + Bspacetime_pred_draw[k];
          B_pred2_c_t_s_st[k] += Bpred_c_t_s_st.array().square().matrix();
        }
        if (keepY) {
          YPRED_out[collect_count] = Ypred;
        }
      }
      
			// Calculate point-wise log-likelihood and generate Yfitted2
      Eigen::MatrixXd Yfitted2(p, t);
      if (family == "gaussian") {
        // Eigen::MatrixXd term1 = -0.5 * yhat.array().square() / s2_err_mis_draw;
        // Eigen::MatrixXd term2 = Eigen::MatrixXd::Constant(p, t, -0.5 * std::log(2.0 * M_PI));
        // Eigen::MatrixXd term3 = Eigen::MatrixXd::Constant(p, t, -0.5 * std::log(s2_err_mis_draw));
        // store_llike[collect_count] = term1 + term2 + term3;
        store_llike[collect_count] = -0.5 * yhat.array().square() / s2_err_mis_draw;
        store_llike[collect_count].array() += -0.5 * std::log(2.0 * M_PI);
        store_llike[collect_count].array() += -0.5 * std::log(s2_err_mis_draw);
        Yfitted2 = Eigen::MatrixXd::NullaryExpr(p, t, norm_mcmc);
        Yfitted2 += thetay_draw;
      } else if (family == "poisson") {
        for (int i = 0; i < p; ++i) {
          for (int j = 0; j < t; ++j) {
            double lambda = std::exp(eta_tilde(i, j));
            store_llike[collect_count](i, j) = R::dpois(Y(i, j), lambda, true);
            Yfitted2(i, j) = R::rpois(lambda);
          }
        }
      } else if (family == "bernoulli") {
        // Eigen::MatrixXd pr_y1(p, t); // Pr (Y = 1 | parameters)
        for (int i = 0; i < p; ++i) {
          for (int j = 0; j < t; ++j) {
            double pr_y1 = 1.0 - R::pnorm(- thetay_draw(i, j), 0.0, 1.0, true, false);
            store_llike[collect_count](i, j) = R::dbinom(Y(i, j), 1, pr_y1, true);
            Yfitted2(i, j) = R::rbinom(1, pr_y1);
          }
        }
      }
      
      
      if (keepY) {
        YFITTED_out[collect_count] = Yfitted;
      }
      
      // Diagnostics
      Eigen::VectorXd Y_obs(n_obs), Yfitted_obs(n_obs), Yfitted_2_obs(n_obs), mean_obs(n_obs), Var_obs(n_obs);
      int idx = 0;
      
      for (const auto& [i, j] : valid_indices) {
        Y_obs(idx) = Y(i, j);
        Yfitted_obs(idx) = Yfitted(i, j);
        Yfitted_2_obs(idx) = Yfitted2(i, j);
        if (family == "gaussian") {
          mean_obs(idx) = thetay_draw(i, j);
          Var_obs(idx) = s2_err_mis_draw;
        } else if (family == "poisson") {
          mean_obs(idx) = std::exp(eta_tilde(i, j));
          Var_obs(idx) = mean_obs(idx);
        } else if (family == "bernoulli") {
          mean_obs(idx) = 1.0 - R::pnorm(- thetay_draw(i, j), 0.0, 1.0, true, false);
          Var_obs(idx) = mean_obs(idx) * (1.0 - mean_obs(idx));
        }
        ++idx;
      }
      
      Eigen::VectorXd pearson_res = (Y_obs - mean_obs).array() / Var_obs.array().sqrt();
      Eigen::VectorXd pearson_res_fitted = (Yfitted_obs - mean_obs).array() / Var_obs.array().sqrt();
      
      chi_sq_obs_(collect_count) = pearson_res.squaredNorm();
      chi_sq_fitted_(collect_count) = pearson_res_fitted.squaredNorm();
      
      if (std::isfinite(chi_sq_obs_(collect_count)) && std::isfinite(chi_sq_fitted_(collect_count))) {
        if(chi_sq_obs_(collect_count) >= chi_sq_fitted_(collect_count)) {
          chisq_count += 1.0;
        }
      }
      
      for (int i = 0; i < n_obs; ++i) {
        if (std::pow(pearson_res(i), 2.0) >= std::pow(pearson_res_fitted(i), 2.0)) {
          pvalue_ResgrRespred_sum(i) += 1.0;
        }
        if (Y_obs(i) >= Yfitted_obs(i)) {
          pvalue_YgrYhat_sum(i) += 1.0;
        }
      }

      // CRPS components
      store_CRPS_1_sum += (Yfitted_obs - Yfitted_2_obs).array().abs().sum();
      store_CRPS_2_sum += (Yfitted_obs - Y_obs).array().abs().sum();

      
      RMSE_(0, collect_count) = std::sqrt((Y_obs - Yfitted_obs).array().square().mean());
      MAE_(0, collect_count) = (Y_obs - Yfitted_obs).array().abs().mean();
      
      Eigen::VectorXd p95_pred = cpp_prctile(Yfitted, quant);
      
      // Element-wise comparison and accumulation
      for (int i = 0; i < p; ++i) {
        if (p95_obs_(i) >= p95_pred(i)) {
          percentile95_sum(i) += 1.0;
        }
      }
      
      
      // Update averages ('ave' structure)
      Yfitted_mean += Yfitted;
      Yfitted2_mean += Yfitted.array().square().matrix();
      Eta_tilde_mean += eta_tilde;
      thetay_mean += thetay_draw;
      meanZmean += meanZ;
      meanY1mean += current_meanY1;
      
      for (int k = 0; k < ncovx; ++k) {
        B_postmean[k] += Bdraw[k];
        B2_postmean[k] += Bdraw[k].array().square().matrix();
        
        Btime_postmean[k] += Btimedraw[k];
        Btime2_postmean[k] += Btimedraw[k].array().square().matrix();
        
        Bspace_postmean[k] += Bspacedraw[k];
        Bspace2_postmean[k] += Bspacedraw[k].array().square().matrix();
        
        Bspacetime_postmean[k] += Bspacetimedraw[k];
        Bspacetime2_postmean[k] += Bspacetimedraw[k].array().square().matrix();
        
        double Bdrawk = Bdraw_vec(k);
        // B_c_t: 1 x t
        Eigen::MatrixXd B_c_t = Eigen::MatrixXd::Ones(1, t) * Bdrawk + Btimedraw[k];
        B2_c_t[k] += B_c_t.array().square().matrix();
        
        // B_c_s: p x 1
        Eigen::MatrixXd B_c_s = Eigen::MatrixXd::Ones(p, 1) * Bdrawk + Bspacedraw[k];
        B2_c_s[k] += B_c_s.array().square().matrix();
        
        // B_c_t_s_st: p x t
        Eigen::MatrixXd B_c_t_s_st = Eigen::MatrixXd::Ones(p, t) * Bdrawk
					+ Btimedraw[k].colwise().replicate(p)
          + Bspacedraw[k].rowwise().replicate(t)
          + Bspacetimedraw[k];
        B2_c_t_s_st[k] += B_c_t_s_st.array().square().matrix();
        
				// B_c_t_st: p x t
				Eigen::MatrixXd B_c_t_st = Eigen::MatrixXd::Ones(p, t) * Bdrawk 
          + Btimedraw[k].colwise().replicate(p)
				  + Bspacetimedraw[k];
				B2_c_t_st[k] += B_c_t_st.array().square().matrix();
				
				// B_t_st: p x t
				Eigen::MatrixXd B_t_st = Btimedraw[k].colwise().replicate(p) + Bspacetimedraw[k];
				B2_t_st[k] += B_t_st.array().square().matrix();
        
				// B_t_s: p x t
				Eigen::MatrixXd B_t_s = Btimedraw[k].colwise().replicate(p) + Bspacedraw[k].rowwise().replicate(t);
				B2_t_s[k] += B_t_s.array().square().matrix();
        
				// B_t_s_st: p x t
				Eigen::MatrixXd B_t_s_st = Btimedraw[k].colwise().replicate(p)
					+ Bspacedraw[k].rowwise().replicate(t)
					+ Bspacetimedraw[k];
				B2_t_s_st[k] += B_t_s_st.array().square().matrix();
					
				// B_c_t_s: p x t
				Eigen::MatrixXd B_c_t_s = Eigen::MatrixXd::Ones(p, t) * Bdrawk
					+ Btimedraw[k].colwise().replicate(p)
					+ Bspacedraw[k].rowwise().replicate(t);
				B2_c_t_s[k] += B_c_t_s.array().square().matrix();
					
				// B_s_st: p x t
				Eigen::MatrixXd B_s_st = Bspacedraw[k].rowwise().replicate(t) + Bspacetimedraw[k];
				B2_s_st[k] += B_s_st.array().square().matrix();
				
				// B_c_s_st: p x t
				Eigen::MatrixXd B_c_s_st = Eigen::MatrixXd::Ones(p, t) * Bdrawk
				+ Bspacedraw[k].rowwise().replicate(t)
					+ Bspacetimedraw[k];
				B2_c_s_st[k] += B_c_s_st.array().square().matrix();
				
				// Interactions
				E_t_s[k] += (Btimedraw[k].colwise().replicate(p).array() * Bspacedraw[k].rowwise().replicate(t).array()).matrix();
				E_t_st[k] += (Btimedraw[k].colwise().replicate(p).array() * Bspacetimedraw[k].array()).matrix();
				E_s_st[k] += (Bspacedraw[k].rowwise().replicate(t).array() * Bspacetimedraw[k].array()).matrix();
      }
      
      
      collect_count++;
    }
    
    
  } // End MCMC loop
  Rcpp::Rcout << "MCMC finished." << std::endl;
  
  // --- Post-processing and calculating averages ---
  double n_samples_collected = static_cast<double>(collect_count);
  
  // Finalize 'ave' list
  Rcpp::List ave_results;
  ave_results["Yfitted_mean"] = Yfitted_mean / n_samples_collected;
  ave_results["Yfitted2_mean"] = Yfitted2_mean / n_samples_collected;
  ave_results["Eta_tilde_mean"] = Eta_tilde_mean / n_samples_collected;
  ave_results["thetay_mean"] = thetay_mean / n_samples_collected;
  ave_results["meanZmean"] = meanZmean / n_samples_collected;
  ave_results["meanY1mean"] = meanY1mean / n_samples_collected;
  
  // Convert vector of matrices to 3D arrays for R
  ave_results["B_postmean"] = cube_to_array(B_postmean, n_samples_collected);
  ave_results["B2_postmean"] = cube_to_array(B2_postmean, n_samples_collected);
  ave_results["Btime_postmean"] = cube_to_array(Btime_postmean, n_samples_collected);
  ave_results["Btime2_postmean"] = cube_to_array(Btime2_postmean, n_samples_collected);
  ave_results["Bspace_postmean"] = cube_to_array(Bspace_postmean, n_samples_collected);
  ave_results["Bspace2_postmean"] = cube_to_array(Bspace2_postmean, n_samples_collected);
  ave_results["Bspacetime_postmean"] = cube_to_array(Bspacetime_postmean, n_samples_collected);
  ave_results["Bspacetime2_postmean"] = cube_to_array(Bspacetime2_postmean, n_samples_collected);
  ave_results["B2_c_t_s_st"] = cube_to_array(B2_c_t_s_st, n_samples_collected);

  if (X_pred_nullable.isNotNull()) {
    // Predictions
    ave_results["Ypred_mean"] = Ypred_mean / n_samples_collected;
    ave_results["Ypred2_mean"] = Ypred2_mean / n_samples_collected;
    ave_results["Btime_pred_postmean"] = cube_to_array(Btime_pred_postmean, n_samples_collected);
    ave_results["Btime_pred2_postmean"] = cube_to_array(Btime_pred2_postmean, n_samples_collected);
    ave_results["Bspacetime_pred_postmean"] = cube_to_array(Bspacetime_pred_postmean, n_samples_collected);
    ave_results["Bspacetime_pred2_postmean"] = cube_to_array(Bspacetime_pred2_postmean, n_samples_collected);
    ave_results["Bspace_pred_postmean"] = cube_to_array(Bspace_pred_postmean, n_samples_collected);
    ave_results["Bspace_pred2_postmean"] = cube_to_array(Bspace_pred2_postmean, n_samples_collected);
    ave_results["B_pred2_c_t_s_st"] = cube_to_array(B_pred2_c_t_s_st, n_samples_collected);
  }
  
  // DIC calculation (using only non-missing Y)
  Eigen::VectorXd store_llike_vec = Eigen::VectorXd::Zero(collect_count);
  for (int c = 0; c < collect_count; ++c) {
    for (const auto& [i, j] : valid_indices) {
      store_llike_vec(c) += store_llike[c](i, j);
    }
  }
  double mean_llike = store_llike_vec.mean();
  double D_bar = -2.0 * mean_llike;

  // Extract posterior means
  Eigen::MatrixXd theta_hat = thetay_mean / n_samples_collected;;
  Eigen::MatrixXd eta_tilde_hat = Eta_tilde_mean / n_samples_collected;
  double s2_err_hat = S2_err_mis_.row(0).mean();
  
  double log_lik_hat = 0.0;
  if (family == "gaussian") {
    for (const auto& [i, j] : valid_indices) {
      log_lik_hat += R::dnorm(Y(i, j), theta_hat(i, j), std::sqrt(s2_err_hat), true);
    }
  } else if (family == "poisson") {
    for (const auto& [i, j] : valid_indices) {
      double lambda = std::exp(eta_tilde_hat(i, j));
      log_lik_hat += R::dpois(Y(i, j), lambda, true);
    }
  } else if (family == "bernoulli") {
    // Eigen::MatrixXd pr_y1(p, t); // Pr (Y = 1 | parameters)
    for (const auto& [i, j] : valid_indices) {
      double pr_y1 = 1.0 - R::pnorm(- theta_hat(i, j), 0.0, 1.0, true, false);
      log_lik_hat += R::dbinom(Y(i, j), 1, pr_y1, true);
    }
  }


  double D_hat = -2.0 * log_lik_hat;  
	ave_results["Dbar"] = D_bar;
	ave_results["pD"] = D_bar - D_hat;
  ave_results["DIC"] = D_bar + (D_bar - D_hat);  // DIC = D_bar + pD = 2*D_bar - D_hat

  // WAIC calculation (using only non-missing Y)
  Rcpp::List waic_list = computeWAIC(store_llike, valid_indices);
  ave_results["WAIC"] = waic_list["waic"];
  ave_results["se_WAIC"] = waic_list["se_waic"];
  ave_results["pWAIC"] = waic_list["p_waic"];
  ave_results["se_pWAIC"] = waic_list["se_p_waic"];
  ave_results["elpd"] = waic_list["elpd"];
  ave_results["se_elpd"] = waic_list["se_elpd"];
  
  // Calculate other summary stats for 'ave'
  ave_results["pvalue_ResgrReshat"] = pvalue_ResgrRespred_sum.mean() / n_samples_collected;
  ave_results["pvalue_YgrYhat"] = pvalue_YgrYhat_sum.mean() / n_samples_collected;
  ave_results["pvalue_chisquare"] = chisq_count / n_samples_collected;
  ave_results["CRPS"] = (0.5 * store_CRPS_1_sum - store_CRPS_2_sum) / (n_samples_collected * n_obs);
  
  
  // Percentile p-value vector
  ave_results["pvalue_perc95"] = Rcpp::wrap((percentile95_sum / n_samples_collected).eval());
  
  // PMCC: Posterior Mean + Variance (using only non-missing Y)
  Eigen::MatrixXd Yfitted_mean_ave = Yfitted_mean / n_samples_collected;
  Eigen::MatrixXd Yfitted2_mean_ave = Yfitted2_mean / n_samples_collected;
  Eigen::MatrixXd varYPRED = Yfitted2_mean_ave - Yfitted_mean_ave.array().square().matrix();
  
  double pmcc_sum = 0.0;
  for (const auto& [i, j] : valid_indices) {
    double diff = Y(i, j) - Yfitted_mean_ave(i, j);
    pmcc_sum += diff * diff + varYPRED(i, j);
  }
  ave_results["PMCC"] = pmcc_sum;

  if (family == "poisson") {
    ave_results["AccRate"] = pswitch_Y / (n_samples_collected + nburn);
  }
  
  // Finalize 'out' list
  out_results["sigma2"] = S2_err_mis_;
  out_results["rho1_space"] = RHO1_space_;
  out_results["rho1_spacetime"] = RHO1_space_time_;
  out_results["rho2_space"] = RHO2_space_;
  out_results["rho2_spacetime"] = RHO2_space_time_;
  out_results["sigma2_Btime"] = S2_Btime_;
  if (!random_walk) {
    out_results["phi_AR1_time"] = PHI_AR1_time_;
    out_results["phi_AR1_spacetime"] = PHI_AR1_space_time_;
  }
  if (ncovz > 0) {
    out_results["gamma"] = G_out;
  }
  if (keepLogLik) {
    out_results["loglik"] = cube_to_array(store_llike, 1.0);
  }
  if (keepY) {
    out_results["fitted"] = cube_to_array(YFITTED_out, 1.0);
    if (X_pred_nullable.isNotNull()) {
      out_results["Ypred"] = cube_to_array(YPRED_out, 1.0);
    }
  }
  if (family != "bernoulli") {
    out_results["RMSE"] = RMSE_;
    out_results["MAE"] = MAE_;
  }
  out_results["chi_sq_fitted_"] = chi_sq_fitted_;
  out_results["chi_sq_obs_"] = chi_sq_obs_;
  
  // Store final state for potential restart
  out_results["eta_tilde"] = eta_tilde;
  out_results["Bdraw"] = cube_to_array(Bdraw, 1.0);
  out_results["Btimedraw"] = cube_to_array(Btimedraw, 1.0);
  out_results["Bspacedraw"] = cube_to_array(Bspacedraw, 1.0);
  out_results["Bspacetimedraw"] = cube_to_array(Bspacetimedraw, 1.0);
  
  
  
  // Return both lists
  return Rcpp::List::create(Rcpp::Named("ave") = ave_results,
                            Rcpp::Named("out") = out_results);
  
}

