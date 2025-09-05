#' MCMC Algorithm
#'
#' @description
#' This function executes the Markov Chain Monte Carlo (MCMC) algorithm for STDGLMs, i.e. spatio-temporal dynamic (generalized) linear models.
#'
#' @param y A \emph{p}-by-\emph{t} matrix corresponding to the response variable, where \emph{p} is the number of spatial locations and \emph{t} is the number of time points. Missing values (\code{NA}) are allowed.
#' @param family A character string indicating the family of the response variable. Currently, only \code{"gaussian"} (default), \code{"poisson"}, and \code{"bernoulli"} are supported.
#' @param X A \emph{p}-by-\emph{t}-by-\emph{ncovx} array corresponding to the covariates whose effects should vary across space and time.
#' @param Z A \emph{p}-by-\emph{t}-by-\emph{ncovz} array corresponding to the covariates whose effects are constant across space and time. Defaults to \code{NULL}.
#' @param offset A \emph{p}-by-\emph{t} matrix corresponding to the offset term. If \code{NULL}, it is set to zero.
#' @param point.referenced A logical indicating whether the data are point-referenced (TRUE) or areal (FALSE). Default is TRUE. If FALSE, predictions are not performed.
#' @param random.walk A logical indicating whether the temporal dynamic should be modeled as a random walk (TRUE) or a first-order autoregressive process (FALSE). Default is FALSE.
#' @param interaction A logical vector indicating whether to include the spatio-temporal interaction effect. Default is TRUE, meaning all covariates' effects are allowed to interact across space and time. If FALSE, no interactions are included. It is also possible to pass a logical vector of length \emph{ncovx} to select specific interactions.
#' @param blocks_indices A list of integer vectors indicating the indices of the blocks for spatial predictions. Defaults to \code{NULL}, if no predictions are needed. See details.
#' @param W A \emph{p}-by-\emph{p} matrix corresponding to the spatial weights matrix. If \code{point.referenced} is TRUE, the distance matrix among the observed locations should be provided. If \code{point.referenced} is FALSE, the 0/1 adjacency matrix should be provided.
#' @param W_pred A list with \emph{p_b}-by-\emph{p_b} matrices corresponding to the distance matrix for the prediction locations in the b-th block, for \emph{b} in \code{1:length(blocks_indices)}. If \code{NULL}, predictions are not performed.
#' @param W_cross A list with \emph{p}-by-\emph{p_b} matrices corresponding to the cross distances between the observed and prediction locations in the b-th block.
#' @param X_pred A \emph{p_new}-by-\emph{t_new}-by-\emph{ncovx} array corresponding to the covariates with varying coefficients for predictions, where \emph{p_new} is the total number of prediction locations and \emph{t_new=t+h_ahead}, \emph{h_ahead}>=0, is the number of time points for which predictions are to be made.
#' @param Z_pred A \emph{p_new}-by-\emph{t_new}-by-\emph{ncovz} array corresponding to the covariates with constant coefficients for predictions. Defaults to \code{NULL}.
#' @param offset_pred A \emph{p_new}-by-\emph{t_new} matrix corresponding to the offset term for predictions. If \code{NULL}, it is set to zero.
#' @param ncores An integer indicating the number of cores to parallelize the spatial predictions. If \code{NULL}, it defaults to 1.
#' @param nrep An integer indicating the number of iterations to keep after burn-in.
#' @param nburn An integer indicating the number of iterations to discard as burn-in.
#' @param thin An integer indicating the thinning value. Default is 1.
#' @param print.interval An integer indicating the interval at which to print progress messages.
#' @param prior A named list containing the hyperparameters of the model. If \code{NULL}, default non-informative hyperparameters are used. See details.
#' @param keepY A logical indicating whether to keep the response variable in the output. Default is TRUE.
#' @param keepLogLik A logical indicating whether to keep the log-likelihood in the output. Default is TRUE.
#' @param last_run An optional list containing the output from a previous run of the function, which can be used to restore the state of the sampler and continue the MCMC. Default is \code{NULL}.
#' 
#' @details At the moment, only Gaussian outcomes are supported. The fitted model has the following form:
#' \deqn{y_{it} = \boldsymbol{x}_{it}' \boldsymbol{\beta}_{it} + \boldsymbol{z}_{it}' \boldsymbol{\gamma} + \epsilon_{it}, \quad \epsilon_{it} \sim N(0, \sigma_\epsilon^2)}
#' \deqn{\boldsymbol{\beta}_{j, t} = \boldsymbol{F}_{j,t} \boldsymbol{\beta}_{j, t-1} + \boldsymbol{\eta}_{j,t}, \quad \boldsymbol{\eta}_{j,t} \sim N_p(0, \boldsymbol{\Sigma}_{\eta, j}), \quad j=1, \dots, J}
#' where \eqn{\boldsymbol{F}_{j,t} = \phi_j^{(\mathsf{T})} \boldsymbol{I}_p} and \eqn{J=ncovx}. 
#' 
#' The function allows for the decomposition of the state vector into components that can be interpreted as contributions from different sources of variability, that is:
#' \deqn{\beta_{it} = \overline{\beta} + \beta_{i}^{(\mathsf{S})} + \beta_{t}^{(\mathsf{T})} + \beta_{it}^{(\mathsf{ST})}}
#' where:
#' \describe{
#'  \item{\eqn{\overline{\beta}}}{The overall mean effect.}
#'  \item{\eqn{\beta_{i}^{(\mathsf{S})}}}{The spatial effect for location \eqn{i}.}
#'  \item{\eqn{\beta_{t}^{(\mathsf{T})}}}{The temporal effect for time \eqn{t}.}
#'  \item{\eqn{\beta_{it}^{(\mathsf{ST})}}}{The spatio-temporal effect for location \eqn{i} and time \eqn{t}.}
#' }
#' See the package vignette for more details.
#'  
#' \code{prior} must be a named list with these elements:
#' \describe{
#'  \item{V_beta_0}{Either a scalar or numeric vector defining the prior variance for the initial state of the time-varying coefficients, related to the covariates in \code{X}. If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{V_gamma}{A scalar defining the prior variance for the initial state of the constant coefficients, i.e. the constant effects of covariates in \code{X} \emph{and} those related to the covariates in \code{Z}.}
#'  \item{a_inn_time}{Either a scalar or numeric vector defining the inverse-gamma prior shape for the temporal innovation variance of the time-varying coefficients. If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{b_inn_time}{Either a scalar or numeric vector defining the inverse-gamma prior rate for the temporal innovation variance of the time-varying coefficients. If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{a_rho1s}{Either a scalar or numeric vector defining the inverse-gamma prior shape for the partial sill of the spatial effects. If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{b_rho1s}{Either a scalar or numeric vector defining the inverse-gamma prior rate for the partial sill of the spatial effects. If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{a_rho1st}{Either a scalar or numeric vector defining the inverse-gamma prior shape for the partial sill of the spatio-temporal effects (if \code{interaction==TRUE}). If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{b_rho1st}{Either a scalar or numeric vector defining the inverse-gamma prior rate for the partial sill of the spatio-temporal effects (if \code{interaction==TRUE}). If it is a vector, it must be of length equal to \emph{ncovx}, the number of covariates in \code{X}.}
#'  \item{s2_a}{A scalar defining the inverse-gamma prior shape for the measurement error variance (if \code{family!="bernoulli"}).}
#'  \item{s2_b}{A scalar defining the inverse-gamma prior rate for the measurement error variance (if \code{family!="bernoulli"}).}
#'  \item{ctuning}{A scalar defining the tuning parameter for the random walk proposal distribution (if \code{family=="poisson"}).}
#' }
#' 
#' Out-of-sample predictions are performed only if \code{point.referenced} is \code{TRUE} and \code{W_pred} is provided. For spatial interpolation of space-varying coefficients, computations are performed block-wise. \code{blocks_indices} is a list of disjoint sets of indices specifying the block membership of each new spatial location. The b-th element of the list has \emph{p_b} new spatial locations, and the total number of new spatial locations is \emph{p_new}, given by the sum of \emph{p_b} over all blocks. \code{W_pred} and \code{W_cross} must be lists of length equal to \code{length(blocks_indices)}. To perform computations in parallel, \code{ncores} must be greater than 1. \cr
#' For temporal predictions of time-varying coefficients, it suffices that \emph{t_new}>\emph{t}.
#' 
#' @return A named list containing \code{ave}, which stores posterior summaries, and \code{out}, which stores the MCMC samples.
#' 
#' The posterior summaries in \code{ave} include:
#' \describe{
#'  \item{Yfitted_mean, Yfitted2_mean}{First two moments of draws from the posterior predictive distribution for the observed data points.}
#'  \item{Ypred_mean, Ypred2_mean}{\emph{p_new}-by-\emph{t_new} matrices with first two moments of draws from the posterior predictive distribution for the new data points (only if out-of-sample predictions are required).}
#'  \item{B_postmean, B2_postmean}{First two moments of the overall effect of varying coefficients.}
#'  \item{Btime_postmean, Btime2_postmean}{First two moments of the temporal effect of varying coefficients.}
#'  \item{Bspace_postmean, Bspace2_postmean}{First two moments of the spatial effect of varying coefficients.}
#'  \item{Bspacetime_postmean, Bspacetime2_postmean}{First two moments of the spatio-temporal effect of varying coefficients.}
#'  \item{B2_c_t_s_st}{2nd moment of the varying coefficients, \eqn{\beta_{it}}.}
#'  \item{Btime_pred_postmean, Btime_pred2_postmean}{First two moments of the temporal effect of varying coefficients at the predicted time points.}
#'  \item{Bspace_pred_postmean, Bspace_pred2_postmean}{First two moments of the spatial effect of varying coefficients at the predicted spatial locations.}
#'  \item{Bspacetime_pred_postmean, Bspacetime_pred2_postmean}{First two moments of the spatio-temporal effect of varying coefficients at the predicted spatial locations and all time points.}
#'  \item{B_pred2_c_t_s_st}{2nd moment of the varying coefficients, \eqn{\beta_{it}}, at the predicted spatial locations and all time points.}
#'  \item{meanY1mean}{Contribution of covariates with varying coefficients.}
#'  \item{meanZmean}{Contribution of covariates with non-varying effects.}
#'  \item{thetay_mean}{It is defined as \code{thetay_mean = meanY1mean + meanZmean + offset}.}
#'  \item{Eta_tilde_mean}{Posterior mean of the linear predictor (for non-gaussian outcomes). For Poisson outcomes, it is defined as \code{Eta_tilde_mean = thetay_mean + epsilon}, where \code{epsilon} is a Gaussian error term. For Bernoulli outcomes, it is obtained by drawing from a truncated normal distribution with mean \code{thetay_mean}.}
#'  \item{DIC, Dbar, pD}{Deviance Information Criterion, \eqn{DIC = \bar{D} + pD}.}
#'  \item{WAIC, se_WAIC, pWAIC, se_pWAIC, elpd, se_elpd}{Widely Applicable Information Criterion, the penalty term, and expected log pointwise predictive density. The prefix \code{se_} denotes standard errors. See Gelman et al. (2014).}
#'  \item{CRPS}{Continuous Ranked Probability Score, as defined in Gschlößl & Czado (2007).}
#'  \item{PMCC}{Predictive model choice criterion proposed by Gelfand & Ghosh (1998).}
#'  \item{pvalue_*}{Bayesian p-values, see \url{https://czaccard.github.io/STDGLM/articles/model_output.html}.}
#'  \item{AccRate}{Point-wise acceptance rate for the random-walk Metropolis-Hastings step (if \code{family=="poisson"}).}
#' }
#' 
#' Note that the criteria above (DIC, WAIC, etc.) are computed only for the non-missing values of the response variable.
#'
#' The MCMC chains included in \code{out} are:
#' \describe{
#' \item{sigma2}{Measurement error variance. For Bernoulli outcomes, it is fixed to 1.}
#' \item{sigma2_Btime}{A \emph{J}-by-\code{nrep} matrix with the variance of the innovation of the temporal effects for j=1,...,J. The j-th row corresponds to the temporal effect of the j-th varying coefficient.}
#' \item{rho1_space, rho2_space}{\emph{J}-by-\code{nrep} matrices with spatial correlation parameters. The j-th row corresponds to the spatial effect of the j-th varying coefficient.}
#' \item{rho1_spacetime, rho2_spacetime}{\emph{J}-by-\code{nrep} matrices with the correlation parameters of spatially-structured innovations in the spatio-temporal effects. The j-th row corresponds to the spatio-temporal effect of the j-th varying coefficient.}
#' \item{phi_AR1_time, phi_AR1_spacetime}{\emph{J}-by-\code{nrep} matrices with the AR(1) coefficients for the temporal effects for j=1,...,J. The j-th row corresponds to the temporal effect of the j-th varying coefficient. If \code{random.walk==TRUE}, it is \code{NULL}.}
#' \item{gamma}{Regression coefficients related to the covariates with non-varying effects.}
#' \item{loglik}{Pointwise log-likelihood, if \code{keepLogLik = TRUE}.}
#' \item{fitted}{Draws from the posterior predictive distribution for the observed data points, if \code{keepY = TRUE}.}
#' \item{Ypred}{Draws from the posterior predictive distribution for the out-of-sample data points, if \code{keepY = TRUE}.}
#' \item{RMSE}{In-sample Root Mean Squared Error, if \code{family != "bernoulli"}.}
#' \item{MAE}{In-sample Mean Absolute Error, if \code{family != "bernoulli"}.}
#' \item{chi_sq_pred_, chi_sq_fitted_}{Chi-square statistics (i.e., sum of squared Pearson residuals) for predicted and fitted values.}
#' }
#' \code{out} contains also other elements needed for restarting.
#' 
#' @examples
#' \dontrun{
#' data(ApuliaAQ, package = "STDGLM")
#' p = length(unique(ApuliaAQ$AirQualityStation)) # 51
#' t = length(unique(ApuliaAQ$time))              # 365
#' 
#' # distance matrix
#' W = as.matrix(dist(cbind(ApuliaAQ$Longitude[1:p], ApuliaAQ$Latitude[1:p])))
#' 
#' # response variable: temperature
#' y = matrix(ApuliaAQ$CL_t2m, p, t)
#' # covariates with spacetime-varying coefficients: intercept + altitude
#' X = array(1, dim = c(p, t, 2))
#' X[,,2] = matrix(ApuliaAQ$Altitude, p, t)
#' 
#' mod <- stdglm(y=y, X=X, W=W)
#' 
#' # Model with spacetime-varying intercept, but fixed altitude effect
#' mod2 <- stdglm(y=y, X=X[,,1,drop=FALSE], Z=X[,,2,drop=FALSE], W=W)
#' }
#'
#' @seealso \code{vignette("STDGLM")}.
#' @references
#' Gelfand, A. E., & Ghosh, S. K. (1998). Model choice: a minimum posterior predictive loss approach. Biometrika, 85(1), 1-11. \cr
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2014). Bayesian data analysis (3rd ed.). Chapman and Hall/CRC. \cr
#' Gschlößl, S., & Czado, C. (2007). Spatial modelling of claim frequency and claim size in non-life insurance. Scandinavian Actuarial Journal, 2007(3), 202–225. \doi{10.1080/03461230701414764}

#' @export
stdglm = function(y, family = "gaussian", X, Z = NULL, offset = NULL, 
                  point.referenced = TRUE, random.walk = FALSE, 
                  interaction = TRUE,
                  blocks_indices = NULL, W, W_pred = NULL, W_cross = NULL, 
                  X_pred = NULL, Z_pred = NULL, offset_pred = NULL,
                  ncores = NULL,
                  nrep = 100, nburn = 100, thin = 1, 
                  print.interval = 10, prior = NULL, keepY = TRUE,
                  keepLogLik = TRUE, last_run = NULL) {
  
  stopifnot("Please specify nburn>1" = nburn>1)
  stopifnot("Please specify nrep>1" = nrep>1)
  stopifnot("NA not allowed in X" = !any(is.na(X)))

  if (is.null(prior)) {
    prior = list(
      V_beta_0 = 1e4,       # Prior variance of initial state
      V_gamma = 1e6,        # Prior variance of constant coefficients
      a_inn_time = 0.01,    # Prior shape for temporal variance
      b_inn_time = 0.01,    # Prior rate for temporal variance
      a_rho1s = 0.01,       # Prior shape for partial sill of spatial effects
      b_rho1s = 0.01,       # Prior rate for partial sill of spatial effects
      a_rho1st = 0.01,      # Prior shape for partial sill of spatio-temporal effects
      b_rho1st = 0.01,      # Prior rate for partial sill of spatio-temporal effects
      s2_a = 0.01,          # Prior shape for measurement error variance
      s2_b = 0.01,          # Prior rate for measurement error variance
      ctuning = 0           # Tuning parameter for the random walk proposal distribution (if family=="poisson")
    )
  }
  V_beta_0 = prior$V_beta_0
  if (length(V_beta_0) == 1) V_beta_0 = rep(V_beta_0, dim(X)[3])
  V_gamma = prior$V_gamma
  a_inn_time = prior$a_inn_time
  if (length(a_inn_time) == 1) a_inn_time = rep(a_inn_time, dim(X)[3])
  b_inn_time = prior$b_inn_time
  if (length(b_inn_time) == 1) b_inn_time = rep(b_inn_time, dim(X)[3])
  a_rho1s = prior$a_rho1s
  if (length(a_rho1s) == 1) a_rho1s = rep(a_rho1s, dim(X)[3])
  b_rho1s = prior$b_rho1s
  if (length(b_rho1s) == 1) b_rho1s = rep(b_rho1s, dim(X)[3])
  if (any(interaction)) {
    a_rho1st = prior$a_rho1st
    b_rho1st = prior$b_rho1st
  } else {
    a_rho1st = 0.01
    b_rho1st = 0.01
  }
  if (length(a_rho1st) == 1) a_rho1st = rep(a_rho1st, dim(X)[3])
  if (length(b_rho1st) == 1) b_rho1st = rep(b_rho1st, dim(X)[3])

  if (family == "poisson" || family == "gaussian") {
    s2_a = prior$s2_a
    s2_b = prior$s2_b
  } else {
    s2_a = 0.01
    s2_b = 0.01
  }
  if (family == "poisson") {
    ctuning = prior$ctuning
    if (is.null(ctuning) || !(ctuning > 0)) {
      stop("ctuning must be a positive number")
    }
  } else {
    ctuning = 0
  }
  
  ncovx = dim(X)[3]
  stopifnot("'interaction' has wrong length." = length(interaction) == 1 || length(interaction) == ncovx)
  X = lapply(1:ncovx, \(i) X[,,i])
  if (!is.null(Z)) {
    stopifnot("NA not allowed in Z" = !any(is.na(Z)))
    ncovz = dim(Z)[3]
    Z = lapply(1:ncovz, \(i) Z[,,i])
  }
  if (is.null(offset)) offset = matrix(0, nrow(X[[1]]), ncol(X[[1]]))

  if (!point.referenced) W_pred = W_cross = NULL

  if (!is.null(W_pred)) {
    stopifnot("Please specify W_cross" = !is.null(W_cross))
    stopifnot("NA not allowed in X_pred" = !any(is.na(X_pred)))
    stopifnot("blocks_indices must be a list" = is.list(blocks_indices))
    stopifnot("W_pred must be a list" = is.list(W_pred))
    stopifnot("W_cross must be a list" = is.list(W_cross))

    blocks_indices = lapply(blocks_indices, \(x) as.integer(x-1)) # Adjust for zero-based indexing in C++

    X_pred = lapply(1:ncovx, \(i) X_pred[,,i])
    if (!is.null(Z_pred)) {
      stopifnot("NA not allowed in Z_pred" = !any(is.na(Z_pred)))
      Z_pred = lapply(1:ncovz, \(i) Z_pred[,,i])
    }
    if (is.null(offset_pred)) offset_pred = matrix(0, nrow(X_pred[[1]]), ncol(X_pred[[1]]))
  } else {
    Z_pred = NULL
    W_pred = W_cross = list(matrix(0, 1, 1))
    offset_pred = matrix(0, 1, 1)
    blocks_indices = list(0)
  }
  
  if (!(is.null(last_run) || identical(last_run, list()))) {
    out_prev = last_run$out
    stopifnot("'last_run' not specified correctly." = !is.null(out_prev))
  } else {
    out_prev = NULL
  }

  if (length(interaction) == 1) {
    interaction = rep(interaction, ncovx)
  }
  interaction = as.numeric(interaction)

  if (is.null(ncores)) ncores = 1
  RcppParallel::setThreadOptions(numThreads = ncores)

  re = dlm_cpp(y, family, X, Z, offset,
               point.referenced, random.walk, 
               interaction,
               blocks_indices,
               W, W_pred, W_cross, 
               X_pred, Z_pred, offset_pred,
               nrep, nburn, thin, print.interval,
               V_beta_0, V_gamma, 
               a_inn_time, b_inn_time, a_rho1s, b_rho1s, a_rho1st, b_rho1st,
               s2_a, s2_b, ctuning,
               keepY, keepLogLik, out_prev)
  class(re) = c("stdglm", class(re))
  return(re)
}
