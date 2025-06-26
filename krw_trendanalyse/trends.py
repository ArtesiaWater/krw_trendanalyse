import numpy as np
from scipy.stats import norm
import pastas as ps
import pandas as pd


# @njit
def _compute_mean_per_series(args, iref=0, z_score=1.96):
    """Compute mean, variance, and confidence interval of a time series per period.

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


def compute_mean_per_series(*args, reference=0, alpha=0.95):
    z_score = norm.ppf(1 - (1 - alpha) / 2)
    mean, var, dvar, ci = _compute_mean_per_series(
        args, iref=reference, z_score=z_score
    )
    # Create a DataFrame to store the results
    data = {
        "mean": mean,
        "var": var,
        "Δvar": dvar,
        "ci": ci,
    }
    return pd.DataFrame(data)


def mean_per_period(s: pd.Series, periods, iref=0, alpha=0.95):
    """Compute mean, variance, and confidence interval of a time series per period.

    Parameters
    ----------
    s : pandas.Series
        times series
    periods : list of tuples with str or pd.Timestamp
        List of periods (tmin, tmax) to split the residuals.
    iref : int
        Index of the reference period (default is 0).
    alpha : float
        Significance level for the confidence interval (default is 0.95).

    Returns
    -------
    mean : np.ndarray
        Array of mean for each period.
    var : np.ndarray
        Array of variance for each period.
    dvar : np.ndarray
        Array of variance relative to the reference period.
    ci : np.ndarray
        Array of confidence intervals for mean estimate each period.
    """
    # Get residuals for each period
    series_per_period = []
    starts = []
    ends = []
    for tmin, tmax in periods:
        if isinstance(tmin, str):
            tmin = pd.Timestamp(tmin)
        if isinstance(tmax, str):
            if len(tmax) == 4:  # year only
                tmax = pd.Timestamp(tmax) + pd.offsets.YearEnd(1)
            else:
                tmax = pd.Timestamp(tmax)
        res_period = s.loc[tmin:tmax]
        starts.append(tmin)
        ends.append(tmax)
        if res_period.empty:
            raise ValueError(f"No residuals found for period {tmin} to {tmax}.")
        series_per_period.append(res_period.values)
    df = compute_mean_per_series(*series_per_period, reference=iref, alpha=alpha)
    df["Δmean"] = df["mean"] - df["mean"].iloc[iref]
    df["start"] = starts
    df["end"] = ends
    df["reference"] = ""
    df["reference"].values[iref] = "*"
    df.index.name = s.name
    return df.loc[
        :, ["reference", "start", "end", "mean", "var", "Δmean", "Δvar", "ci"]
    ]


def model_residual_period_stats(
    ml: ps.Model,
    periods,
    iref=0,
    alpha=0.95,
    add_contributions=None,
    tmin=None,
    tmax=None,
):
    """Compute mean, variance, and confidence interval of model residuals per period.

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
    add_contributions : list of str
        List of contributions to add to the residuals (default is None).
        If None, only the residuals are used.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with mean, variance, delta variance and confidence interval
        per period.
         - mean_res: mean of residuals for each period.
         - var_res: variance of residuals for each period.
         - dvar: variance relative to the reference period.
         - ci: confidence intervals for each period.
    """
    if tmin is None:
        tmin = pd.Timestamp(periods[0][0])
    if tmax is None:
        tmax = pd.Timestamp(periods[-1][1])
    res = ml.residuals(tmin=tmin, tmax=tmax)
    if add_contributions is not None:
        for contribution in add_contributions:
            c = ml.get_contribution(contribution, tmin=tmin, tmax=tmax)
            # interpolate to match the residuals index
            c = c.reindex(c.index.union(res.index)).interpolate().loc[res.index]
            res += c

    df = mean_per_period(res, periods, iref=iref, alpha=alpha)
    df.index.name = ml.name
    return df


def aggregate_trends(trends, iref=0):
    """Aggregate trends from multiple time series.

    Parameters
    ----------
    trends : list of pandas.DataFrame
        List of DataFrames with columns 'mean' and 'var' for each time series.
        Each DataFrame should have a datetime index.
    iref : int
        Index of the reference period (default is 0).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with aggregated mean, variance, standard deviation, confidence interval,
        lower and upper bounds for each period. Columns are:
        - agg_mean: aggregated mean for each period.
        - Δagg_mean: change in aggregated mean relative to the reference period.
        - σ: standard deviation of the aggregated mean.
        - ci: confidence interval for the aggregated mean.
        - lower_bound: lower bound of the confidence interval.
        - upper_bound: upper bound of the confidence interval.
    """
    # collect means and variances, different series as rows, periods as columns
    means = pd.concat(
        [t["mean"] for t in trends], axis=1, keys=[t.index.name for t in trends]
    ).T
    variances = pd.concat(
        [t["var"] for t in trends], axis=1, keys=[t.index.name for t in trends]
    ).T
    # deal with 0 variance
    variances[variances == 0.0] = np.nan
    stdev = np.sqrt(variances)
    norm_mean = means / stdev
    mean = norm_mean.sum(axis=0) / (1 / stdev).sum(axis=0)
    mean_ref = mean - mean.iloc[iref]  # reference to first period
    # mean of std devs, corrected for NaNs
    mean_std = stdev.mean(axis=0) / np.sqrt((~stdev.isna()).sum(axis=0))
    ci = 1.96 * mean_std  # 95% confidence interval
    lb = mean_ref - ci
    ub = mean_ref + ci

    df = pd.concat(
        [mean, mean_ref, mean_std, ci, lb, ub],
        axis=1,
        keys=["agg_mean", "Δagg_mean", "σ", "ci", "lower_bound", "upper_bound"],
    )
    df.index.name = "period"
    return df


def _aggregate_trends_original(trends, iref=0):
    """Aggregate trends from multiple time series.

    Original implementation to compare with the Python-style aggregate_trends.

    Parameters
    ----------
    trends : list of pandas.DataFrame
        List of DataFrames with columns 'mean' and 'var' for each time series.
        Each DataFrame should have a datetime index.
    iref : int
        Index of the reference period (default is 0).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with aggregated mean, variance, standard deviation, confidence interval,
        lower and upper bounds for each period. Columns are:
        - agg_mean: aggregated mean for each period.
        - Δagg_mean: change in aggregated mean relative to the reference period.
        - σ: standard deviation of the aggregated mean.
        - ci: confidence interval for the aggregated mean.
        - lower_bound: lower bound of the confidence interval.
        - upper_bound: upper bound of the confidence interval.
    """
    # collect means and variances, different series as rows, periods as columns
    means = pd.concat(
        [t["mean"] for t in trends], axis=1, keys=[t.index.name for t in trends]
    ).T
    variances = pd.concat(
        [t["var"] for t in trends], axis=1, keys=[t.index.name for t in trends]
    ).T
    n_periods = means.columns.size
    n_series = means.index.size
    # Initialize arrays
    norm_mean = np.zeros(means.shape)
    stdev_1 = np.zeros(means.shape)
    stdev = np.zeros(means.shape)
    sum_norm_mean = np.zeros(n_periods)
    sum_stdev = np.zeros(n_periods)
    mean = np.zeros(n_periods)
    meanref = np.zeros(n_periods)
    mean_sqrt_var = np.zeros(n_periods)
    mean_stdev = np.zeros(n_periods)
    confidence_interval = np.zeros(n_periods)
    lower_bound = np.zeros(n_periods)
    upper_bound = np.zeros(n_periods)

    for k in range(n_periods):
        for m in range(n_series):
            # Normalize mean
            norm_mean[m, k] = means.iloc[m, k] / np.sqrt(variances.iloc[m, k])
            # Calculate standard deviation
            stdev_1[m, k] = 1 / np.sqrt(variances.iloc[m, k])
            stdev[m, k] = np.sqrt(variances.iloc[m, k])
            sum_norm_mean[k] = np.sum(norm_mean[:, k])
            sum_stdev[k] = np.sum(stdev_1[:, k])

        # Calculate MEAN
        n = norm_mean.shape[0]
        mean[k] = sum_norm_mean[k] / sum_stdev[k]
        meanref[k] = mean[k] - mean[0]
        mean_sqrt_var[k] = np.mean(stdev[:, k])

        # Calculate MEAN_STDEV
        mean_stdev[k] = mean_sqrt_var[k] / np.sqrt(n)

        # Calculate 95% Confidence Interval
        confidence_interval[k] = 1.96 * mean_stdev[k]
        lower_bound[k] = meanref[k] - confidence_interval[k]
        upper_bound[k] = meanref[k] + confidence_interval[k]

    df = pd.DataFrame(index=range(n_periods))
    df["mean"] = mean
    df["Δmean"] = meanref
    df["σ"] = mean_stdev
    df["ci"] = confidence_interval
    df["lower_bound"] = lower_bound
    df["upper_bound"] = upper_bound
    df.index.name = "period"
    return df
