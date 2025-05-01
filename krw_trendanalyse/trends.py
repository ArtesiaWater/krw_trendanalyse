from numba import njit
import numpy as np
from scipy.stats import norm
import pastas as ps
import pandas as pd


# @njit
def _compute_residual_stats(args, iref=0, z_score=1.96):
    """Compute mean, variance, and confidence interval of residuals per period.

    Parameters
    ----------
    args : tuple
        tuple of arrays containing the residuals for each period.
    iref : int
        index of the reference period (default is 0).
    z_score : float
        z-score for the confidence interval (default is 1.96 for 95% CI).

    Returns
    -------
    mean_res : np.ndarray
        array of mean residuals for each period.
    var_res : np.ndarray
        array of variance of residuals for each period.
    delta_var_res : np.ndarray
        array of variance relative to the reference period.
    ci : np.ndarray
        array of confidence intervals for each period.

    Notes
    -----
    Gebruikte methode en Matlab-script in 2012 (aangeleverd door W. Swierstra,
    RHDHV, 9-5-2017). Matlab-script omgezet naar Python-script.
    Wiskundige formules:
    - Broers, H.P., P. Schipper, R. Stuurman, F.C. van Geer en G. van Oyen
      (2005) Opzet van het KRW-meetprogramma grondwater voor het stroomgebied
      Maas. NITG-05-176-A.
    """
    num_periods = len(args)

    # Initialize arrays
    mean_res = np.zeros(num_periods, dtype=float)
    var_res = np.zeros(num_periods, dtype=float)

    # Process each period
    for k, res in enumerate(args):
        res_notnull = res[~np.isnan(res)]
        # Calculate MeanRes
        mean_res[k] = np.mean(res_notnull)
        # Length of residuals
        n = res_notnull.size
        # Calculate autocorrelation of residuals
        corr = np.correlate(res_notnull, res_notnull, "full")
        corr = corr[len(res_notnull) :] / corr[len(res_notnull) - 1]
        # Calculate variance of residuals
        ivar = np.var(res_notnull)
        var_res[k] = (ivar / n) * (1 + (2 / n) * (np.arange(n - 1, 0, -1) * corr).sum())

    # Calculate variance relative to reference period
    delta_var_res = var_res + var_res[iref]
    delta_var_res[iref] = 0.0  # Set the variance of the reference period to 0

    # Calculate 95% confidence interval
    ci = z_score * np.sqrt(delta_var_res)

    return mean_res, var_res, delta_var_res, ci


def compute_residual_stats(*args, reference=0, alpha=0.95):
    z_score = norm.ppf(1 - (1 - alpha) / 2)
    mean, var, dvar, ci = _compute_residual_stats(args, iref=reference, z_score=z_score)
    # Create a DataFrame to store the results
    data = {
        "mean": mean,
        "var": var,
        "dvar": dvar,
        "ci": ci,
    }
    return pd.DataFrame(data)


def model_residual_stats(ml: ps.Model, periods, iref=0, alpha=0.95):
    """Compute mean, variance, and confidence interval of residuals per period.

    Parameters
    ----------
    ml : pastas.Model
        Model object from the pastas library.
    periods : list of tuples with str or pd.Timestamp
        List of periods (tmin, tmax) to split the residuals.
    iref : int
        Index of the reference period (default is 0).
    alpha : float
        Significance level for the confidence interval (default is 0.95).

    Returns
    -------
    mean_res : np.ndarray
        Array of mean residuals for each period.
    var_res : np.ndarray
        Array of variance of residuals for each period.
    delta_var_res : np.ndarray
        Array of variance relative to the reference period.
    ci : np.ndarray
        Array of confidence intervals for each period.
    """
    # Get residuals for each period
    res = ml.residuals()
    res_per_period = [res.loc[tmin:tmax].values for tmin, tmax in periods]
    df = compute_residual_stats(*res_per_period, reference=iref, alpha=alpha)
    df["start"] = [pd.Timestamp(tmin) for tmin, _ in periods]
    df["end"] = [pd.Timestamp(tmax) for _, tmax in periods]
    df.index.name = ml.name
    return df.loc[:, ["start", "end", "mean", "var", "dvar", "ci"]]


def aggregate_trends():

    # TODO: convert this code:
    VarRes = pd.DataFrame(data02)

    num_periods = MeanRes.shape[1]
    # Initialize arrays
    NormMean = np.zeros(MeanRes.shape)
    Stdev_1 = np.zeros(MeanRes.shape)
    Sqrt_VarRes = np.zeros(MeanRes.shape)
    SUM_NormMean = np.zeros(num_periods)
    SUM_Stdev = np.zeros(num_periods)
    MEAN = np.zeros(num_periods)
    MEANREF = np.zeros(num_periods)
    MEAN_Sqrt_VarRes = np.zeros(num_periods)
    MEAN_STDEV = np.zeros(num_periods)
    confidence_interval = np.zeros(num_periods)
    lower_bound = np.zeros(num_periods)
    upper_bound = np.zeros(num_periods)


    for k in range(MeanRes.shape[1]):
        for m in range(MeanRes.shape[0]):
            # Normalize mean
            NormMean[m,k] = MeanRes.iloc[m,k] / np.sqrt(VarRes.iloc[m,k])
            # Calculate standard deviation
            Stdev_1[m,k] = 1 / np.sqrt(VarRes.iloc[m,k])
            Sqrt_VarRes[m,k] = np.sqrt(VarRes.iloc[m,k])
            SUM_NormMean[k] = np.sum(NormMean[:,k])
            SUM_Stdev[k] = np.sum(Stdev_1[:,k])

        # Calculate MEAN
        n = NormMean.shape[0]
        MEAN[k] = SUM_NormMean[k] / SUM_Stdev[k]
        MEANREF[k] = MEAN[k] - MEAN[0]
        MEAN_Sqrt_VarRes[k] = np.mean(Sqrt_VarRes[:,k])

        # Calculate MEAN_STDEV
        MEAN_STDEV[k] = MEAN_Sqrt_VarRes[k] / np.sqrt(n)

        # Calculate 95% Confidence Interval
        confidence_interval[k] = 1.96 * MEAN_STDEV[k]
        lower_bound[k] = MEANREF[k] - confidence_interval[k]
        upper_bound[k] = MEANREF[k] + confidence_interval[k]
