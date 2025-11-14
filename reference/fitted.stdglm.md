# Fitted values for `stdglm` objects

Extracts the fitted values from an `stdglm` object.

## Usage

``` r
# S3 method for class 'stdglm'
fitted(object, ...)
```

## Arguments

- object:

  An `stdglm` object.

- ...:

  Additional arguments (currently ignored).

## Value

A matrix of fitted values.

## Details

Extracts the posterior mean of the predictive distribution for the
observed data points.

## See also

[`stdglm`](https://czaccard.github.io/STDGLM/reference/stdglm.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `mod` is a fitted stdglm object
fitted_values <- fitted(mod)
summary(t(fitted_values[1:5,])) # Extract temporal summaries for the first 5 locations
} # }
```
