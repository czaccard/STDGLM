<!-- badges: start -->
[![R-CMD-check](https://github.com/czaccard/STDGLM/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/czaccard/STDGLM/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->


# STDGLM
R package: Spatio-Temporal Dynamic Generalized Linear Models (STDGLM) by Carlo Zaccardi

The `STDGLM` package provides a framework for fitting spatio-temporal dynamic generalized linear models. These models are useful for analyzing data that varies over both space and time, allowing for the incorporation of spatial and temporal dependencies in the modeling process. The package provides functions for fitting these models, as well as tools for visualizing and interpreting the results.

# Installation
```r
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
  require(devtools)
}
devtools::install_github("czaccard/STDGLM")
```

# Quick Usage Example

```r
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
- [ ] Add non-Gaussian responses
- [ ] Add seasonal temporal effects
- [ ] Allow for different priors for variances and ranges
- [ ] Add bool to decide whether to save VC draws



## Changelog
### 0.0.0.9000
- Initial release
