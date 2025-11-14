# Introduction to STDGLM

## Introduction

The `STDGLM` package provides a framework for fitting spatio-temporal
dynamic generalized linear models. These models are useful for analyzing
data that varies over both space and time, allowing for the
incorporation of spatial and temporal dependencies in the modeling
process. The package provides functions for fitting these models, as
well as tools for visualizing and interpreting the results.

## Installation

You can install the package from GitHub using the following command:

``` r
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
  require(devtools)
}
devtools::install_github("czaccard/STDGLM")
```

Run the following command to load the package:

``` r
library(STDGLM)
```

## Quick Usage Example

``` r
data(ApuliaAQ)
p = length(unique(ApuliaAQ$AirQualityStation)) # 51
t = length(unique(ApuliaAQ$time))              # 365

# distance matrix
W = as.matrix(dist(cbind(ApuliaAQ$Longitude[1:p], ApuliaAQ$Latitude[1:p])))

# response variable: temperature
y = matrix(ApuliaAQ$CL_t2m, p, t)
# covariates (intercept + altitude)
X = array(1, dim = c(p, t, 2))
X[,,2] = matrix(ApuliaAQ$Altitude, p, t)

mod <- stdglm(y=y, X=X, W=W)
```

## Detailed Explanation on supported STDGLMs

Let p denote the number of *spatial units* (either georeferenced
locations or areal units) where data is collected, and let T denote the
number of *time points*. Generalized dynamic linear models (GDLMs) with
the following **observation equation** can be handled: y\_{it} \sim F \\
\eta\_{it} = g\left(E\left( y\_{it} \right)\right) =
\boldsymbol{x}\_{it}' \boldsymbol{\beta}\_{it} + \boldsymbol{z}\_{it}'
\boldsymbol{\gamma} + \epsilon\_{it}, \quad \epsilon\_{it} \sim N(0,
\sigma\_\epsilon^2) where:

- y\_{it} is the response variable at spatial unit i=1, \dots, p and
  time t=1, \dots, T, following an exponential family distribution F,

- \eta\_{it} is the corresponding linear predictor, defined using a
  non-linear link function g(\cdot),

- \boldsymbol{x}\_{it} = (x\_{1,it}, \dots, x\_{J,it})' is a
  J-dimensional (J\ge 1) vector of covariates at spatial unit i at time
  t (an intercept may or may not be included here),

- \boldsymbol{\beta}\_{it} = (\beta\_{1,it}, \dots, \beta\_{J,it})' is
  the state vector at time t at spatial unit i,

- \boldsymbol{z}\_{it} is a q-dimensional vector of covariates whose
  effects are constant (an intercept may or may not be included here),

- \boldsymbol{\gamma} is a vector of non-varying coefficients,

- \epsilon\_{it} is the observation error at time t at spatial unit i.

The evolution of the state vector is described by a **state equation**,
which follows from the ANOVA decomposition of the state vector, as
described in a later section.

### Supported Distributions

As for the current version, the following distributions for the outcome
are supported: Gaussian, Poisson, Bernoulli.

### ANOVA Decomposition of the State Vector

The function `stdglm` allows for the decomposition of the state vector
into components that can be interpreted as contributions from different
sources of variability. Dropping the subscript j for the sake of
simplicity, the state vector \beta\_{it} is decomposed as follows:
\beta\_{it} = \overline{\beta} + \beta\_{i}^{(\mathsf{S})} +
\beta\_{t}^{(\mathsf{T})} + \beta\_{it}^{(\mathsf{ST})} where:

- \overline{\beta} is the overall mean effect,

- \beta\_{i}^{(\mathsf{S})} is the spatial effect at spatial unit i,

- \beta\_{t}^{(\mathsf{T})} is the temporal effect at time t,

- \beta\_{it}^{(\mathsf{ST})} is the interaction effect between space
  and time at spatial unit i and time t.

#### Spatial Effects

The spatial effects \beta\_{1}^{(\mathsf{S})}, \dots,
\beta\_{p}^{(\mathsf{S})} are structured to reflect *spatial
relationships*. This is achieved by assuming a zero-mean Gaussian
process with an **exponential covariance function** if the data are
point-referenced, denoted as GP(0, \rho\_{1}, \rho\_{2}; exp):
Cov(\beta\_{i}^{(\mathsf{S})}, \beta\_{\ell}^{(\mathsf{S})}) = \rho\_{1}
\exp\left(-\frac{d\_{i\ell}}{\rho\_{2}}\right), \quad d\_{i\ell} = \\
\boldsymbol{s}\_i - \boldsymbol{s}\_\ell \\ where \rho\_{1} is the
partial sill, \rho\_{2} is the range parameter, and d\_{i\ell} is the
Euclidean distance between locations \boldsymbol{s}\_i and
\boldsymbol{s}\_\ell.

If the data are areal, a **proper conditional autoregressive (PCAR)
covariance structure** is assumed: Var(\beta\_{1}^{(\mathsf{S})}, \dots,
\beta\_{p}^{(\mathsf{S})}) = \rho\_{1} \left( \boldsymbol{D}\_w -
\rho\_{2} \boldsymbol{W} \right)^{-1} where \boldsymbol{W} is a binary
adjacency matrix, \boldsymbol{D}\_w is a diagonal matrix with row sums
of \boldsymbol{W} on the diagonal, and \rho\_{1} and \rho\_{2} are the
conditional variance and autocorrelation parameters, respectively. The
zero-mean PCAR process is denoted as PCAR(0, \rho\_{1}, \rho\_{2}).

#### Temporal Effects

The temporal effects can be specified in three ways:

- **autoregressive process** of order 1, denoted as AR(1;
  \phi^{(\mathsf{T})}): \beta\_{t}^{(\mathsf{T})} = \phi^{(\mathsf{T})}
  \beta\_{t-1}^{(\mathsf{T})} + \eta\_{t}^{(\mathsf{T})}, \quad
  \eta\_{t}^{(\mathsf{T})} \sim N(0, V\_\beta^{(\mathsf{T})}) where
  \phi^{(\mathsf{T})} is the temporal autocorrelation parameter and
  V\_\beta^{(\mathsf{T})} is the innovation variance.

- **random walk**, denoted as RW, which is a special case of the AR(1)
  process with \phi^{(\mathsf{T})} = 1

- **seasonal** component, denoted as SEAS(q), where q is the number of
  harmonics to include. The seasonal component is modeled as:
  \beta\_{t}^{(\mathsf{T})} = \boldsymbol{F} \boldsymbol{\theta}\_t \\
  \boldsymbol{\theta}\_t = \boldsymbol{G} \boldsymbol{\theta}\_{t-1} +
  \boldsymbol{\eta}\_t, \quad \boldsymbol{\eta}\_t \sim N(0,
  V\_\beta^{(\mathsf{T})} \boldsymbol{I}\_{2q}) where \boldsymbol{F} =
  (1, 0, 1, 0, \dots, 1, 0) is a 1 \times 2q matrix, \boldsymbol{G} is a
  2q \times 2q block diagonal matrix with the k-th block of the form
  \begin{pmatrix} \cos(2\pi k / R) & \sin(2\pi k / R) \\ -\sin(2\pi k
  / R) & \cos(2\pi k / R) \end{pmatrix}, \quad k=1, \dots, q and R is
  the period of the seasonal component (e.g., R=12 for monthly data with
  yearly seasonality).

### Bayesian Hierarchical Structure

For Gaussian outcomes, the Bayesian model is as follows, for t=1, \dots,
T and j=1, \dots, J (generalization to other distributions is
straightforward): \begin{align\*} y\_{it} &\sim N(\boldsymbol{x}\_{it}'
\boldsymbol{\beta}\_{it} + \boldsymbol{z}\_{it}' \boldsymbol{\gamma},
\sigma\_\epsilon^2) \\ \boldsymbol{\beta}\_{j,t} &= \boldsymbol{1}\_p
\overline{\beta}\_j + \boldsymbol{\beta}\_j^{(\mathsf{S})} +
\boldsymbol{1}\_p \beta\_{j,t}^{(\mathsf{T})} +
\boldsymbol{\beta}\_{j,t}^{(\mathsf{ST})} \\
\boldsymbol{\beta_j}^{(\mathsf{S})} &= (\beta\_{j,1}^{(\mathsf{S})},
\dots, \beta\_{j,p}^{(\mathsf{S})})' \sim GP(0,
\rho\_{j,1}^{(\mathsf{S})}, \rho\_{j,2}^{(\mathsf{S})}; exp) \quad or
\quad PCAR(0, \rho\_{j,1}^{(\mathsf{S})}, \rho\_{j,2}^{(\mathsf{S})}) \\
\beta\_{j,t}^{(\mathsf{T})} &\sim AR(1; \phi_j^{(\mathsf{T})}) \quad or
\quad RW \quad or \quad SEAS(q_j) \\
\boldsymbol{\beta}\_{j,t}^{(\mathsf{ST})} &=
(\beta\_{j,1t}^{(\mathsf{ST})}, \dots, \beta\_{j,pt}^{(\mathsf{ST})})'
\sim GP(\phi_j^{(\mathsf{ST})}
\boldsymbol{\beta}\_{j,t-1}^{(\mathsf{ST})},
\rho\_{j,1}^{(\mathsf{ST})}, \rho\_{j,2}^{(\mathsf{ST})}; exp) \quad or
\quad PCAR(\phi_j^{(\mathsf{ST})}
\boldsymbol{\beta}\_{j,t-1}^{(\mathsf{ST})},
\rho\_{j,1}^{(\mathsf{ST})}, \rho\_{j,2}^{(\mathsf{ST})}) \end{align\*}

The model is completed with the following priors (again, dropping the
subscript j for simplicity): \begin{align\*} \overline{\beta} &\sim N(0,
V\_\gamma) \\ \beta\_{0}^{(\mathsf{T})} &\sim N(0, V\_{\beta_0}) \\
\boldsymbol{\beta}\_{0}^{(\mathsf{ST})} &\sim N_p(0, V\_{\beta_0}
\boldsymbol{I}\_p) \\ \boldsymbol{\gamma} &\sim N_q(0, V\_\gamma) \\
\sigma\_\epsilon^2 &\sim IG(a\_\epsilon, b\_\epsilon) \\
\rho_1^{(\mathsf{S})} &\sim IG(a\_{\rho, S}, b\_{\rho, S}) \\
\rho_2^{(\mathsf{S})} &\sim U(min\_\rho, max\_\rho) \\
\rho_1^{(\mathsf{ST})} &\sim IG(a\_{\rho, ST}, b\_{\rho, ST}) \\
\rho_2^{(\mathsf{ST})} &\sim U(min\_\rho, max\_\rho) \\
\phi^{(\mathsf{T})} &\sim TN\_{(-1, 1)}(0, 1) \\ \phi^{(\mathsf{ST})}
&\sim TN\_{(-1, 1)}(0, 1) \\ V\_\beta^{(\mathsf{T})} &\sim
IG(a^{(\mathsf{T})}, b^{(\mathsf{T})}) \end{align\*} where TN\_{(q,
r)}(\mu, \sigma^2) denotes a normal distribution with mean \mu and
variance \sigma^2 truncated to the interval (q, r). The hyperparameters
min\_\rho and max\_\rho depend on the type of spatial data. If the data
are point-referenced, they are set to the minimum and maximum distances
between points divided by 3, respectively. If the data are areal,
min\_\rho= 0.1 and max\_\rho \rightarrow 1.

Note that the spacetime-varying coefficients are assumed independent a
priori across j=1,\dots,J.

### Efficient Inference and Identifiability

To build an efficient sampler, the algorithm proposed by Chan and
Jeliazkov (2009) is used in conjuction with sparse matrix techniques.

To make the model identifiable, some **constraints** are imposed on the
varying parameters at each MCMC iteration:

- Set \sum\_{t=1}^{T} \beta\_{t}^{(\mathsf{T})} = 0.

- Set \sum\_{i=1}^{p} \beta\_{i}^{(\mathsf{S})} = 0.

- Set \sum\_{i=1}^{p} \beta\_{i,t}^{(\mathsf{ST})} = 0 for each
  t=1,\dots,T.

- Set \sum\_{t=1}^{T} \beta\_{i,t}^{(\mathsf{ST})} = 0 for each
  i=1,\dots,p.

## References

Chan, Joshua CC, and Ivan Jeliazkov. 2009. “Efficient Simulation and
Integrated Likelihood Estimation in State Space Models.” *International
Journal of Mathematical Modelling and Numerical Optimisation* 1 (1-2):
101–20.
