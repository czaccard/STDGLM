# MCMC Algorithm

This function executes the Markov Chain Monte Carlo (MCMC) algorithm for
STDGLMs, i.e. spatio-temporal dynamic (generalized) linear models.

## Usage

``` r
stdglm(
  y,
  family = "gaussian",
  X,
  Z = NULL,
  offset = NULL,
  point.referenced = TRUE,
  random.walk = FALSE,
  interaction = TRUE,
  n.harmonics = 0,
  period = 0,
  blocks_indices = NULL,
  W,
  W_pred = NULL,
  W_cross = NULL,
  X_pred = NULL,
  Z_pred = NULL,
  offset_pred = NULL,
  ncores = NULL,
  nrep = 100,
  nburn = 100,
  thin = 1,
  print.interval = 10,
  prior = NULL,
  keepY = TRUE,
  keepLogLik = TRUE,
  last_run = NULL
)
```

## Arguments

- y:

  A *p*-by-*t* matrix corresponding to the response variable, where *p*
  is the number of spatial locations and *t* is the number of time
  points. Missing values (`NA`) are allowed.

- family:

  A character string indicating the family of the response variable.
  Currently, only `"gaussian"` (default), `"poisson"`, and `"bernoulli"`
  are supported.

- X:

  A *p*-by-*t*-by-*ncovx* array corresponding to the covariates whose
  effects should vary across space and time.

- Z:

  A *p*-by-*t*-by-*ncovz* array corresponding to the covariates whose
  effects are constant across space and time. Defaults to `NULL`.

- offset:

  A *p*-by-*t* matrix corresponding to the offset term. If `NULL`, it is
  set to zero.

- point.referenced:

  A logical indicating whether the data are point-referenced (TRUE) or
  areal (FALSE). Default is TRUE. If FALSE, predictions are not
  performed.

- random.walk:

  A logical indicating whether the temporal dynamic should be modeled as
  a random walk (TRUE) or a first-order autoregressive process (FALSE).
  Default is FALSE.

- interaction:

  A logical vector indicating whether to include the spatio-temporal
  interaction effect. Default is TRUE, meaning all covariates' effects
  are allowed to interact across space and time. If FALSE, no
  interactions are included. It is also possible to pass a logical
  vector of length *ncovx* to select specific interactions.

- n.harmonics:

  A scalar or a numeric vector of length *ncovx* with non-negative
  values for modeling the cyclic behavior of the temporal effects. It
  stands for the number of harmonics of the Fourier representation. If
  scalar, 0 indicates that the seasonal behavior should not be modeled.
  If positive integer, the seasonal behavior is modeled for all temporal
  effects, and the input `random.walk` is ignored. Using the same logic,
  a numeric vector can be provided to select specific effects. Defaults
  to 0.

- period:

  A scalar indicating the period of the data, required for modeling the
  cyclic behavior. It is ignored whenever `n.harmonics=0`. Defaults to
  0.

- blocks_indices:

  A list of integer vectors indicating the indices of the blocks for
  spatial predictions. Defaults to `NULL`, if no predictions are needed.
  See details.

- W:

  A *p*-by-*p* matrix corresponding to the spatial weights matrix. If
  `point.referenced` is TRUE, the distance matrix among the observed
  locations should be provided. If `point.referenced` is FALSE, the 0/1
  adjacency matrix should be provided.

- W_pred:

  A list with *p_b*-by-*p_b* matrices corresponding to the distance
  matrix for the prediction locations in the b-th block, for *b* in
  `1:length(blocks_indices)`. If `NULL`, predictions are not performed.

- W_cross:

  A list with *p*-by-*p_b* matrices corresponding to the cross distances
  between the observed and prediction locations in the b-th block.

- X_pred:

  A *p_new*-by-*t_new*-by-*ncovx* array corresponding to the covariates
  with varying coefficients for predictions, where *p_new* is the total
  number of prediction locations and *t_new=t+h_ahead*, *h_ahead*\>=0,
  is the number of time points for which predictions are to be made.

- Z_pred:

  A *p_new*-by-*t_new*-by-*ncovz* array corresponding to the covariates
  with constant coefficients for predictions. Defaults to `NULL`.

- offset_pred:

  A *p_new*-by-*t_new* matrix corresponding to the offset term for
  predictions. If `NULL`, it is set to zero.

- ncores:

  An integer indicating the number of cores to parallelize the spatial
  predictions. If `NULL`, it defaults to 1.

- nrep:

  An integer indicating the number of iterations to keep after burn-in.

- nburn:

  An integer indicating the number of iterations to discard as burn-in.

- thin:

  An integer indicating the thinning value. Default is 1.

- print.interval:

  An integer indicating the interval at which to print progress
  messages.

- prior:

  A named list containing the hyperparameters of the model. If `NULL`,
  default non-informative hyperparameters are used. See details.

- keepY:

  A logical indicating whether to keep the response variable in the
  output. Default is TRUE.

- keepLogLik:

  A logical indicating whether to keep the log-likelihood in the output.
  Default is TRUE.

- last_run:

  An optional list containing the output from a previous run of the
  function, which can be used to restore the state of the sampler and
  continue the MCMC. Default is `NULL`.

## Value

A named list containing `ave`, which stores posterior summaries, and
`out`, which stores the MCMC samples.

The posterior summaries in `ave` include:

- Yfitted_mean, Yfitted2_mean:

  First two moments of draws from the posterior predictive distribution
  for the observed data points.

- Ypred_mean, Ypred2_mean:

  *p_new*-by-*t_new* matrices with first two moments of draws from the
  posterior predictive distribution for the new data points (only if
  out-of-sample predictions are required).

- B_postmean, B2_postmean:

  First two moments of the overall effect of varying coefficients.

- Btime_postmean, Btime2_postmean:

  First two moments of the temporal effect of varying coefficients.

- Bspace_postmean, Bspace2_postmean:

  First two moments of the spatial effect of varying coefficients.

- Bspacetime_postmean, Bspacetime2_postmean:

  First two moments of the spatio-temporal effect of varying
  coefficients.

- B2_c_t_s_st:

  2nd moment of the varying coefficients, \\\beta\_{it}\\.

- Btime_pred_postmean, Btime_pred2_postmean:

  First two moments of the temporal effect of varying coefficients at
  the predicted time points.

- Bspace_pred_postmean, Bspace_pred2_postmean:

  First two moments of the spatial effect of varying coefficients at the
  predicted spatial locations.

- Bspacetime_pred_postmean, Bspacetime_pred2_postmean:

  First two moments of the spatio-temporal effect of varying
  coefficients at the predicted spatial locations and all time points.

- B_pred2_c_t_s_st:

  2nd moment of the varying coefficients, \\\beta\_{it}\\, at the
  predicted spatial locations and all time points.

- meanY1mean:

  Contribution of covariates with varying coefficients.

- meanZmean:

  Contribution of covariates with non-varying effects, if `Z` is not
  NULL.

- thetay_mean:

  It is defined as `thetay_mean = meanY1mean + meanZmean + offset`.

- Eta_tilde_mean:

  Posterior mean of the linear predictor (for non-gaussian outcomes).
  For Poisson outcomes, it is defined as
  `Eta_tilde_mean = thetay_mean + epsilon`, where `epsilon` is a
  Gaussian error term. For Bernoulli outcomes, it is obtained by drawing
  from a truncated normal distribution with mean `thetay_mean`.

- DIC, Dbar, pD:

  Deviance Information Criterion, \\DIC = \bar{D} + pD\\.

- WAIC, se_WAIC, pWAIC, se_pWAIC, elpd, se_elpd:

  Widely Applicable Information Criterion, the penalty term, and
  expected log pointwise predictive density. The prefix `se_` denotes
  standard errors. See Gelman et al. (2014).

- CRPS:

  Continuous Ranked Probability Score, as defined in Gschlößl & Czado
  (2007).

- PMCC:

  Predictive model choice criterion proposed by Gelfand & Ghosh (1998).

- pvalue\_\*:

  Bayesian p-values, see
  <https://czaccard.github.io/STDGLM/articles/model_output.html>.

- AccRate:

  Point-wise acceptance rate for the random-walk Metropolis-Hastings
  step (if `family=="poisson"`).

Note that the criteria above (DIC, WAIC, etc.) are computed only for the
non-missing values of the response variable.

The MCMC chains included in `out` are:

- sigma2:

  Measurement error variance. For Bernoulli outcomes, it is fixed to 1.

- sigma2_Btime:

  A *J*-by-`nrep` matrix with the variance of the innovation of the
  temporal effects for j=1,...,J. The j-th row corresponds to the
  temporal effect of the j-th varying coefficient.

- rho1_space, rho2_space:

  *J*-by-`nrep` matrices with spatial correlation parameters. The j-th
  row corresponds to the spatial effect of the j-th varying coefficient.

- rho1_spacetime, rho2_spacetime:

  *J*-by-`nrep` matrices with the correlation parameters of
  spatially-structured innovations in the spatio-temporal effects. The
  j-th row corresponds to the spatio-temporal effect of the j-th varying
  coefficient.

- phi_AR1_time, phi_AR1_spacetime:

  *J*-by-`nrep` matrices with the AR(1) coefficients for the temporal
  effects for j=1,...,J. The j-th row corresponds to the temporal effect
  of the j-th varying coefficient. If `random.walk==TRUE`, it is `NULL`.

- gamma:

  Regression coefficients related to the covariates with non-varying
  effects.

- loglik:

  Pointwise log-likelihood, if `keepLogLik = TRUE`.

- fitted:

  Draws from the posterior predictive distribution for the observed data
  points, if `keepY = TRUE`.

- Ypred:

  Draws from the posterior predictive distribution for the out-of-sample
  data points, if `keepY = TRUE`.

- RMSE:

  In-sample Root Mean Squared Error, if `family != "bernoulli"`.

- MAE:

  In-sample Mean Absolute Error, if `family != "bernoulli"`.

- chi_sq_pred\_, chi_sq_fitted\_:

  Chi-square statistics (i.e., sum of squared Pearson residuals) for
  predicted and fitted values.

`out` contains also other elements needed for restarting.

## Details

The fitted model has the following form: \$\$y\_{it} \sim F\$\$
\$\$\eta\_{it} = g(E( y\_{it} )) = \boldsymbol{x}\_{it}'
\boldsymbol{\beta}\_{it} + \boldsymbol{z}\_{it}' \boldsymbol{\gamma} +
\epsilon\_{it}, \quad \epsilon\_{it} \sim N(0, \sigma\_\epsilon^2)\$\$
\$\$\boldsymbol{\beta}\_{j, t} = \boldsymbol{G}\_{j,t}
\boldsymbol{\beta}\_{j, t-1} + \boldsymbol{\eta}\_{j,t}, \quad
\boldsymbol{\eta}\_{j,t} \sim N_p(0, \boldsymbol{\Sigma}\_{\eta, j}),
\quad j=1, \dots, J\$\$ where either \\\boldsymbol{G}\_{j,t} =
\phi_j^{(\mathsf{T})} \boldsymbol{I}\_p\\ or \\\boldsymbol{G}\_{j,t}\\
is block diagonal with harmonic matrices, and \\J=ncovx\\.

The function allows for the decomposition of the state vector into
components that can be interpreted as contributions from different
sources of variability, that is: \$\$\beta\_{it} = \overline{\beta} +
\beta\_{i}^{(\mathsf{S})} + \beta\_{t}^{(\mathsf{T})} +
\beta\_{it}^{(\mathsf{ST})}\$\$ where:

- \\\overline{\beta}\\:

  The overall mean effect.

- \\\beta\_{i}^{(\mathsf{S})}\\:

  The spatial effect for location \\i\\.

- \\\beta\_{t}^{(\mathsf{T})}\\:

  The temporal effect for time \\t\\.

- \\\beta\_{it}^{(\mathsf{ST})}\\:

  The spatio-temporal effect for location \\i\\ and time \\t\\.

See the package vignette for more details.

`prior` must be a named list with these elements:

- V_beta_0:

  Either a scalar or numeric vector defining the prior variance for the
  initial state of the time-varying coefficients, related to the
  covariates in `X`. If it is a vector, it must be of length equal to
  *ncovx*, the number of covariates in `X`.

- V_gamma:

  A scalar defining the prior variance for the initial state of the
  constant coefficients, i.e. the constant effects of covariates in `X`
  *and* those related to the covariates in `Z`.

- a_inn_time:

  Either a scalar or numeric vector defining the inverse-gamma prior
  shape for the temporal innovation variance of the time-varying
  coefficients. If it is a vector, it must be of length equal to
  *ncovx*, the number of covariates in `X`.

- b_inn_time:

  Either a scalar or numeric vector defining the inverse-gamma prior
  rate for the temporal innovation variance of the time-varying
  coefficients. If it is a vector, it must be of length equal to
  *ncovx*, the number of covariates in `X`.

- a_rho1s:

  Either a scalar or numeric vector defining the inverse-gamma prior
  shape for the partial sill of the spatial effects. If it is a vector,
  it must be of length equal to *ncovx*, the number of covariates in
  `X`.

- b_rho1s:

  Either a scalar or numeric vector defining the inverse-gamma prior
  rate for the partial sill of the spatial effects. If it is a vector,
  it must be of length equal to *ncovx*, the number of covariates in
  `X`.

- a_rho1st:

  Either a scalar or numeric vector defining the inverse-gamma prior
  shape for the partial sill of the spatio-temporal effects (if
  `interaction==TRUE`). If it is a vector, it must be of length equal to
  *ncovx*, the number of covariates in `X`.

- b_rho1st:

  Either a scalar or numeric vector defining the inverse-gamma prior
  rate for the partial sill of the spatio-temporal effects (if
  `interaction==TRUE`). If it is a vector, it must be of length equal to
  *ncovx*, the number of covariates in `X`.

- s2_a:

  A scalar defining the inverse-gamma prior shape for the measurement
  error variance (if `family!="bernoulli"`).

- s2_b:

  A scalar defining the inverse-gamma prior rate for the measurement
  error variance (if `family!="bernoulli"`).

- ctuning:

  A scalar defining the tuning parameter for the random walk proposal
  distribution (if `family=="poisson"`).

Out-of-sample predictions are performed only if `point.referenced` is
`TRUE` and `W_pred` is provided. For spatial interpolation of
space-varying coefficients, computations are performed block-wise.
`blocks_indices` is a list of disjoint sets of indices specifying the
block membership of each new spatial location. The b-th element of the
list has *p_b* new spatial locations, and the total number of new
spatial locations is *p_new*, given by the sum of *p_b* over all blocks.
`W_pred` and `W_cross` must be lists of length equal to
`length(blocks_indices)`. To perform computations in parallel, `ncores`
must be greater than 1.  
For temporal predictions of time-varying coefficients, it suffices that
*t_new*\>*t*.

## References

Gelfand, A. E., & Ghosh, S. K. (1998). Model choice: a minimum posterior
predictive loss approach. Biometrika, 85(1), 1-11.  
Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2014). Bayesian
data analysis (3rd ed.). Chapman and Hall/CRC.  
Gschlößl, S., & Czado, C. (2007). Spatial modelling of claim frequency
and claim size in non-life insurance. Scandinavian Actuarial Journal,
2007(3), 202–225.
[doi:10.1080/03461230701414764](https://doi.org/10.1080/03461230701414764)

## See also

[`vignette("STDGLM")`](https://czaccard.github.io/STDGLM/articles/STDGLM.md).

## Examples

``` r
if (FALSE) { # \dontrun{
data(ApuliaAQ, package = "STDGLM")
p = length(unique(ApuliaAQ$AirQualityStation)) # 51
t = length(unique(ApuliaAQ$time))              # 365

# distance matrix
W = as.matrix(dist(cbind(ApuliaAQ$Longitude[1:p], ApuliaAQ$Latitude[1:p])))

# response variable: temperature
y = matrix(ApuliaAQ$CL_t2m, p, t)
# covariates with spacetime-varying coefficients: intercept + altitude
X = array(1, dim = c(p, t, 2))
X[,,2] = matrix(ApuliaAQ$Altitude, p, t)

mod <- stdglm(y=y, X=X, W=W, interaction = FALSE)

# Model with spacetime-varying intercept, but fixed altitude effect
mod2 <- stdglm(y=y, X=X[,,1,drop=FALSE], Z=X[,,2,drop=FALSE], W=W, interaction = FALSE)
} # }
```
