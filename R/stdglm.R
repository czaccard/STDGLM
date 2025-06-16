#' MCMC Algorithm
#'
#' @description
#' This function executes the Markov Chain Monte Carlo (MCMC) algorithm for STDGLM.
#'
#' @param y Either a \emph{p}-by-\emph{t} matrix corresponding to the response variable, where \emph{p} is the number of spatial locations and \emph{t} is the number of time points.
#' @param X A \emph{p}-by-\emph{t}-by-\emph{ncovx} array corresponding to the covariates whose effects should vary across space and time.
#' @param Z A \emph{p}-by-\emph{t}-by-\emph{ncovz} array corresponding to the covariates whose effects are constant across space and time. Defaults to \code{NULL}.
#' @param offset A \emph{p}-by-\emph{t} matrix corresponding to the offset term. If \code{NULL}, it is set to zero.
#' @param point.referenced A logical indicating whether the data are point-referenced (TRUE) or areal (FALSE). Default is TRUE. If FALSE, predictions are not performed.
#' @param random.walk A logical indicating whether the temporal dynamic should be modeled as a random walk (TRUE) or a first-order autoregressive process (FALSE). Default is FALSE.
#' @param W A \emph{p}-by-\emph{p} matrix corresponding to the spatial weights matrix. If \code{point.referenced} is TRUE, the distance matrix among the observed locations should be provided. If \code{point.referenced} is FALSE, the 0/1 adjacency matrix should be provided.
#' @param W_pred A \emph{p}-by-\emph{p} matrix corresponding to the spatial weights matrix for the prediction locations. If \code{NULL}, predictions are not performed.
#' @param W_cross A \emph{p}-by-\emph{p_new} matrix corresponding to the cross spatial weights matrix between the observed and prediction locations.
#' @param X_pred A \emph{p_new}-by-\emph{t_new}-by-\emph{ncovx} array corresponding to the covariates with varying coefficients for predictions, where \emph{p_new} is the number of prediction locations and \emph{t_new=t+h_ahead}, \emph{h_ahead}>=0, is the number of time points for which predictions are to be made.
#' @param Z_pred A \emph{p_new}-by-\emph{t_new}-by-\emph{ncovz} array corresponding to the covariates with constant coefficients for predictions. Defaults to \code{NULL}.
#' @param offset_pred A \emph{p_new}-by-\emph{t_new} matrix corresponding to the offset term for predictions. If \code{NULL}, it is set to zero.
#' @param nrep An integer indicating the number of iterations to keep after burn-in.
#' @param nburn An integer indicating the number of iterations to discard as burn-in.
#' @param thin An integer indicating the thinning value. Default is 1.
#' @param print.interval An integer indicating the interval at which to print progress messages.
#' @param prior A named list containing the hyperparameters of the model. If \code{NULL}, default non-informative hyperparameters are used. See details below.
#' @param keepY A logical indicating whether to keep the response variable in the output. Default is TRUE.
#' @param keepLogLik A logical indicating whether to keep the log-likelihood in the output. Default is TRUE.
#' @param last_run An optional list containing the output from a previous run of the function, which can be used to restore the state of the sampler. Default is \code{NULL}.
#' 
#' @details \code{prior} must be a named list with these elements:
#' \describe{
#'  \item{V_beta_0}{A scalar defining the prior variance for the initial state of the time-varying coefficients, related to the covariates in \code{X}.}
#'  \item{V_gamma}{A scalar defining the prior variance for the initial state of the constant coefficients, related to the covariates in \code{Z}. Assign a value even if \code{Z} is \code{NULL}.}
#'  \item{a1}{A scalar defining the inverse-gamma prior shape for the temporal variance of the time-varying coefficients.}
#'  \item{b1}{A scalar defining the inverse-gamma prior rate for the temporal variance of the time-varying coefficients.}
#'  \item{s2_a}{A scalar defining the inverse-gamma prior shape for the measurement error variance.}
#'  \item{s2_b}{A scalar defining the inverse-gamma prior rate for the measurement error variance.}
#' }
#' 
#' @return A named list containing \code{out} and \code{ave}. \code{ave} stores posterior summaries, while \code{out} stores the MCMC samples.
#'
#' @references insert here.
#' @export
#'
stdglm = function(y, X, Z = NULL, offset = NULL, 
                  point.referenced = TRUE, random.walk = FALSE, 
                  W, W_pred = NULL, W_cross = NULL, 
                  X_pred = NULL, Z_pred = NULL, offset_pred = NULL,
                  nrep = 100, nburn = 100, thin = 1, 
                  print.interval = 100, prior = NULL, keepY = TRUE,
                  keepLogLik = TRUE, last_run = NULL) {
  
  stopifnot("Please specify nburn>1" = nburn>1)
  stopifnot("Please specify nrep>1" = nrep>1)
  stopifnot("NA not allowed in X" = !any(is.na(X)))

  if (is.null(prior)) {
    prior = list(
      V_beta_0 = 1e4, # Prior variance of initial state
      V_gamma = 1e6,  # Prior variance of constant coefficients
      a1 = 0.01,      # Prior shape for temporal variance
      b1 = 0.01,      # Prior rate for temporal variance
      s2_a = 0.01,    # Prior shape for measurement error variance
      s2_b = 0.01     # Prior rate for measurement error variance
    )
  }
  V_beta_0 = prior$V_beta_0
  V_gamma = prior$V_gamma
  a1 = prior$a1
  b1 = prior$b1
  s2_a = prior$s2_a
  s2_b = prior$s2_b
  
  ncovx = dim(X)[3]
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
    X_pred = lapply(1:ncovx, \(i) X_pred[,,i])
    if (!is.null(Z_pred)) {
      stopifnot("NA not allowed in Z_pred" = !any(is.na(Z_pred)))
      Z_pred = lapply(1:ncovz, \(i) Z_pred[,,i])
    }
  } else {
    Z_pred = NULL
    W_pred = W_cross = matrix(0, 1, 1)
  }
  if (is.null(offset_pred)) offset_pred = matrix(0, nrow(X_pred[[1]]), ncol(X_pred[[1]]))
  
  if (!(is.null(last_run) | identical(last_run, list()))) {
    out_prev = last_run$out
    stopifnot("'last_run' not specified correctly." = !is.null(out_prev))
  } else {
    out_prev = NULL
  }
  
  re = dlm_cpp(y, X, Z, offset,
               point.referenced, random.walk, 
               W, W_pred, W_cross, 
               X_pred, Z_pred, offset_pred,
               nrep, nburn, thin, print.interval,
               V_beta_0, V_gamma, 
               a1, b1, s2_a, s2_b, keepY, keepLogLik, out_prev)
  return(re)
}
