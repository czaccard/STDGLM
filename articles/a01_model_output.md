# Details on Model Output

This article dives into the details of the output of the function
[`stdglm()`](https://czaccard.github.io/STDGLM/reference/stdglm.md).

## Model Output

The output is an object of class `stdglm` which is a list with elements
`ave` and `out`. The element `ave` contains the posterior means of the
model parameters, while `out` contains the full MCMC output.

`ave` and `out` are lists as well, and their elements change depending
on the model fitted.

### `ave` list

Generally speaking, the list `ave` contains the following elements:

- `Yfitted_mean`, `Yfitted2_mean`: p-by-T matrices with first two
  moments of draws from the posterior predictive distribution for the
  observed data points.
- `Ypred_mean`, `Ypred2_mean`: p\_{new}-by-T\_{new} matrices with first
  two moments of draws from the posterior predictive distribution for
  the new data points (only if out-of-sample predictions are required).
- `B_postmean`, `B2_postmean`: First two moments of the overall effect
  of varying coefficients.
- `Btime_postmean`, `Btime2_postmean`: First two moments of the temporal
  effect of varying coefficients.
- `Bspace_postmean`, `Bspace2_postmean`: First two moments of the
  spatial effect of varying coefficients.
- `Bspacetime_postmean`, `Bspacetime2_postmean`: First two moments of
  the spatio-temporal effect of varying coefficients.
- `B2_c_t_s_st`: 2nd moment of the varying coefficients, \beta\_{it}.
- `Btime_pred_postmean`, `Btime_pred2_postmean`: First two moments of
  the temporal effect of varying coefficients at the predicted time
  points (only if out-of-sample predictions are required).
- `Bspace_pred_postmean`, `Bspace_pred2_postmean`: First two moments of
  the spatial effect of varying coefficients at the predicted spatial
  locations (only if out-of-sample predictions are required).
- `Bspacetime_pred_postmean`, `Bspacetime_pred2_postmean`: First two
  moments of the spatio-temporal effect of varying coefficients at the
  predicted spatial locations and all time points (only if out-of-sample
  predictions are required).
- `B_pred2_c_t_s_st`: 2nd moment of the varying coefficients,
  \beta\_{it}, at the predicted spatial locations and all time points
  (only if out-of-sample predictions are required).
- `meanY1mean`: Contribution of covariates with varying coefficients,
  i.e. \boldsymbol{x}\_{it} multiplied by its effects.
- `meanZmean`: Contribution of covariates with non-varying effects,
  i.e. \boldsymbol{z}\_{it}' \boldsymbol{\gamma} (only if `Z` is
  specified).
- `thetay_mean`: It is defined as `meanY1mean + meanZmean + offset`.
- `Eta_tilde_mean`: Posterior mean of the linear predictor (for
  non-Gaussian outcomes). For Poisson outcomes, it is defined as
  `Eta_tilde_mean = thetay_mean + epsilon`, where `epsilon` is a
  Gaussian error term. For Bernoulli outcomes, it is obtained by drawing
  from a truncated normal distribution with mean `thetay_mean`.
- `DIC`, `Dbar`, `pD`: Deviance Information Criterion, DIC = \bar{D} +
  pD.
- `WAIC`, `se_WAIC`, `pWAIC`, `se_pWAIC`, `elpd`, `se_elpd`: Widely
  Applicable Information Criterion, the penalty term, and expected log
  pointwise predictive density. The prefix se\_ denotes standard errors.
  See Gelman et al. (2014) for details.
- `CRPS`: Continuous Ranked Probability Score. It is *positively
  oriented*, i.e. the model with the highest mean score is favoured
  (Gschlößl and Czado 2007):
  \mathrm{CRPS}\left(y\_{it}\right)=\frac{1}{2} E\left\|y\_{r e p,
  {it}}-\tilde{y}\_{r e p, {it}}\right\|-E\left\|y\_{r e p,
  {it}}-y\_{it}\right\| where y\_{r e p, {it}} and \tilde{y}\_{r e p,
  {it}} are independent replicates from the posterior predictive
  distribution.
- `PMCC`: Predictive model choice criterion. It is *negatively
  oriented*, i.e. the model with the lowest score is favoured (Gelfand
  and Ghosh 1998): \mathrm{PMCC}=\sum\_{i=1}^p \sum\_{t=1}^T
  \left\\y\_{it}-E\left(y\_{r e p, {it}} \mid
  \mathbf{y}\right)\right\\^2+\sum\_{i=1}^p \sum\_{t=1}^T
  \operatorname{Var}\left(y\_{r e p, {it}} \mid \mathbf{y}\right) .
- **Bayesian p-values**: following Gelman et al. (2014), the *posterior
  predictive p-value* is defined as:
  p_B=\operatorname{Pr}\left(T\left(y, \theta\right) \geq T(y\_{rep},
  \theta) \mid y\right), for some test quantity T(y, \theta) and some
  parameter vector \theta. The *p-values* returned by
  [`stdglm()`](https://czaccard.github.io/STDGLM/reference/stdglm.md)
  are based on the following functions:
  - `pvalue_YgrYhat`: T(y\_{it}, \theta) = y\_{it}.
  - `pvalue_ResgrReshat`: T(y\_{it}, \theta) = r, where
    r\_{it}=\frac{y\_{it} - E\left( y\_{it} \mid \theta
    \right)}{\operatorname{Var}\left( y\_{it} \mid \theta \right)} are
    the Pearson residuals.
  - `pvalue_chisquare`: T(y, \theta) = \sum\_{i=1}^p \sum\_{t=1}^T
    r\_{it}^2
  - `pvalue_perc95`: T(y, \theta) is the 95-th percentile of the
    distribution of the outcome for each spatial location.
- `AccRate`: Point-wise acceptance rate for the random-walk
  Metropolis-Hastings step (only for Poisson outcome).

Note that the criteria above (DIC, WAIC, p-values, etc.) are computed
*only* using the non-missing values of the response variable.

### `out` list

The list `out` contains the following elements:

- please see the documentation of the function
  [here](https://czaccard.github.io/STDGLM/reference/stdglm.html).

Note that the function
[`stdglm()`](https://czaccard.github.io/STDGLM/reference/stdglm.md) does
not return all the posterior draws for the varying coefficients, but
only their posterior summaries (i.e., first two moments). This is done
to save memory, as the storing matrices can be very large.

## References

Gelfand, Alan E, and Sujit K Ghosh. 1998. “Model Choice: A Minimum
Posterior Predictive Loss Approach.” *Biometrika* 85 (1): 1–11.

Gelman, Andrew, John B Carlin, Hal S Stern, and Donald B Rubin. 2014.
*Bayesian Data Analysis*. 3rd ed. Chapman; Hall/CRC.

Gschlößl, Susanne, and Claudia Czado. 2007. “Spatial Modelling of Claim
Frequency and Claim Size in Non-Life Insurance.” *Scandinavian Actuarial
Journal* 2007 (3): 202–25.
