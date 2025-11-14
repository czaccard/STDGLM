
<!-- README.md is generated from README.Rmd. Please edit that file -->

# STDGLM <a href="https://czaccard.github.io/STDGLM/"><img src="man/figures/logo.png" align="right" height="139" alt="STDGLM website" /></a>

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/STDGLM)](https://CRAN.R-project.org/package=STDGLM)
<!-- [![Codecov test coverage](https://codecov.io/gh/czaccard/STDGLM/graph/badge.svg)](https://app.codecov.io/gh/czaccard/STDGLM) -->
[![R-CMD-check](https://github.com/czaccard/STDGLM/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/czaccard/STDGLM/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

R package: Spatio-Temporal Dynamic Generalized Linear Models (STDGLM) by
Carlo Zaccardi

The `STDGLM` package provides a framework for fitting spatio-temporal
dynamic generalized linear models. These models are useful for analyzing
data that varies over both space and time, allowing for the
incorporation of spatial and temporal dependencies in the modeling
process. The package provides functions for fitting these models, as
well as tools for visualizing and interpreting the results.

This package has been developed with funding from [**High-resolution
Data Fusion for AIR Quality MAPping
(HAIRQ-MAP)**](https://github.com/GRINS-Spoke0-WP2/HAIRQ-MAP), a
research project led by a team at the University “G. d’Annunzio” of
Chieti-Pescara (UdA).

> **Funding acknowledgement**  
> Financial support from the National Recovery and Resilience Plan
> **GRINS - PE0000018 - BAC “High-Resolution Data Fusion for Air Quality
> Mapping - HAIRQ-MAP”**, **SPOKE 0 e 2 - CUP J33C22002910001**.

# Installation

``` r
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
  require(devtools)
}
devtools::install_github("czaccard/STDGLM")
```

# Quick Usage Example

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

## TO-DO

- [ ] Allow for different prior distributions for variances and ranges
- [ ] Add bool to decide whether to save VC draws

## How to Cite / Acknowledge

If you use the `STDGLM` package in published research, please run the
following R code to generate a citation:

``` r
citation("STDGLM")
```

We acknowledge the funding:

> HAIRQ-MAP — High-resolution Data Fusion for AIR Quality MAPping.
> University “G. d’Annunzio” of Chieti-Pescara. Funded by the National
> Recovery and Resilience Plan **GRINS - PE0000018 - BAC
> “High-Resolution Data Fusion for Air Quality Mapping - HAIRQ-MAP”**,
> **SPOKE 0 e 2 - CUP J33C22002910001**.

<p align="center">

<img src="man/figures/logo_BAC.png" alt="NextGenerationEU • MUR • Italia Domani • UdA" />
</p>
