# Retrieve predictions for `stdglm` objects

Extracts out-of-sample predictions from an `stdglm` object.

## Usage

``` r
# S3 method for class 'stdglm'
predict(object, type = "response_mat", Coo_sf_pred = NULL, ...)
```

## Arguments

- object:

  An `stdglm` object.

- type:

  A character string indicating the type of coefficients. Options are:

  - `response_mat`: Posterior mean of the predictive distribution for
    the response variable. Returns a *p_new*-by-*t_new* matrix
    (default).

  - `response_df`: Mean, standard deviation and 95 % credible interval
    for the response variable (default). Returns a dataframe.

  - `tvc`: Mean, standard deviation and 95 % credible interval for the
    temporal effects of varying coefficients.

  - `svc`: Mean, standard deviation and 95 % credible interval for the
    spatial effects of varying coefficients.

  - `stvc`: Mean, standard deviation and 95 % credible interval for the
    spatio-temporal effects of varying coefficients.

- Coo_sf_pred:

  A simple feature object from package `sf` with the prediction points,
  whose geometry is used for spatial effects (optional).

- ...:

  Additional arguments (currently ignored).

## Value

Either a matrix or a dataframe or an `sf` object with posterior mean and
95% credible interval bounds. The function returns `NULL` if predictions
are not available in the `stdglm` object.

## Details

Returns the posterior mean of the predictive distribution and the
associated 95% credible intervals for the out-of-sample data points, and
for the random variable specified in the input `type`.

## See also

[`stdglm`](https://czaccard.github.io/STDGLM/reference/stdglm.md),
[`sf`](https://r-spatial.github.io/sf/reference/sf.html)

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `mod` is a fitted stdglm object
predictions <- predict(mod)
colMeans(predictions) # Get the average predictions across all locations
pred_df <- predict(mod, type = 'response_df')
head(pred_df)
} # }
```
