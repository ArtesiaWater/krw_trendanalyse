import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import CenteredNorm


def plot_mean_per_period(
    series,
    df,
    color_method="significance",
    plot_series=True,
    plot_mean=True,
    plot_ci=True,
    ax=None,
):
    """Plot mean per period and confidence intervals.

    Parameters
    ----------
    series : pandas.Series
        Time series to plot.
    df : pandas.DataFrame
        DataFrame with mean, variance, and confidence interval per period.
    color_method : str
        Color method to use. Options are "significance" or "scaled".

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.
    """
    tmin = df["start"].min()
    tmax = df["end"].max()

    if color_method == "scaled":
        cmap = plt.get_cmap("RdYlGn")
        norm = CenteredNorm(vcenter=0.0, halfrange=df["Δmean"].abs().max())

    meanref = df["mean"].loc[df["reference"] == "*"].item()
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 3))
    else:
        f = ax.figure
    if plot_series:
        ax.plot(series.index, series.values, label=series.name, color="k", lw=1.0)

    add_to_legend = True
    for i, irow in df.iterrows():
        if irow["reference"] == "*":
            if plot_mean:
                ax.plot(
                    [irow["start"], irow["end"]],
                    [irow["mean"], irow["mean"]],
                    color="k",
                    label="reference period",
                    lw=2.0,
                )
        else:
            if color_method == "scaled":
                color = cmap(norm(irow["Δmean"]))
            else:
                if (irow["mean"] + irow["ci"]) < meanref:
                    color = "C3"
                elif (irow["mean"] - irow["ci"]) > meanref:
                    color = "C2"
                else:
                    color = "C0"
            if plot_mean:
                ax.plot(
                    [irow["start"], irow["end"]],
                    [irow["mean"], irow["mean"]],
                    color=color,
                    label="mean" if irow["reference"] == "*" else None,
                    lw=2.0,
                )
            if plot_ci:
                ax.fill_between(
                    [irow["start"], irow["end"]],
                    irow["mean"] - irow["ci"],
                    irow["mean"] + irow["ci"],
                    alpha=0.2,
                    label="confidence interval (95%)" if add_to_legend else None,
                    color=color,
                )
                # ax.plot(
                #     [irow["start"], irow["end"]],
                #     [irow["mean"] - irow["ci"]] * 2,
                #     color=color,
                #     linestyle="dashed",
                #     label="confidence interval (95%)" if add_to_legend else None,
                # )
                # ax.plot(
                #     [irow["start"], irow["end"]],
                #     [irow["mean"] + irow["ci"]] * 2,
                #     color=color,
                #     linestyle="dashed",
                # )
            add_to_legend = False

    ax.set_xlim(tmin, tmax + pd.offsets.Day())
    ax.grid(True)
    for t in df["start"]:
        ax.axvline(t, color="k", ls="dashed", lw=1.0)
    ax.axvline(df["end"].iloc[-1], color="k", ls="dashed", lw=1.0)
    ax.legend(loc=(0, 1), frameon=False, ncol=3, fontsize="small")
    ax.set_ylabel("[m NAP]")
    return ax


def plot_dmean_per_period(
    df,
    color_method="significance",
    plot_ci=True,
    ax=None,
):
    """Plot mean per period and confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with mean, variance, and confidence interval per period.
    color_method : str
        Color method to use. Options are "significance" or "scaled".

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.
    """
    tmin = df["start"].min()
    tmax = df["end"].max()

    if color_method == "scaled":
        cmap = plt.get_cmap("RdYlGn")
        norm = CenteredNorm(vcenter=0.0, halfrange=df["Δmean"].abs().max())

    meanref = df["mean"].loc[df["reference"] == "*"].item()
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(10, 3))
    else:
        f = ax.figure

    add_to_legend = True
    for i, irow in df.iterrows():
        if irow["reference"] == "*":
            ax.plot(
                [irow["start"], irow["end"]],
                [irow["Δmean"], irow["Δmean"]],
                color="k",
                label="reference period",
                lw=2.0,
            )
        else:
            if color_method == "scaled":
                color = cmap(norm(irow["Δmean"]))
            else:
                if (irow["mean"] + irow["ci"]) < meanref:
                    color = "C3"
                elif (irow["mean"] - irow["ci"]) > meanref:
                    color = "C2"
                else:
                    color = "C0"

            ax.plot(
                [irow["start"], irow["end"]],
                [irow["Δmean"], irow["Δmean"]],
                color=color,
                label="Δmean" if irow["reference"] == "*" else None,
                lw=2.0,
            )
            if plot_ci:
                ax.fill_between(
                    [irow["start"], irow["end"]],
                    irow["Δmean"] - irow["ci"],
                    irow["Δmean"] + irow["ci"],
                    alpha=0.2,
                    label="confidence interval (95%)" if add_to_legend else None,
                    color=color,
                )
            add_to_legend = False

    ax.set_xlim(tmin, tmax + pd.offsets.Day())
    ax.grid(True)
    for t in df["start"]:
        ax.axvline(t, color="k", ls="dashed", lw=1.0)
    ax.axvline(df["end"].iloc[-1], color="k", ls="dashed", lw=1.0)
    ax.legend(loc=(0, 1), frameon=False, ncol=4, fontsize="small")
    return ax


def plot_model_residuals_summary(ml, df, add_contributions=None, axes=None, color=None):
    tmin = df["start"].min()
    tmax = df["end"].max()

    if axes is None:
        f, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    else:
        f = axes[0].figure
        assert len(axes) == 3, "There must be 3 axes to plot the data."

    obs = ml.observations(tmin=tmin, tmax=tmax)
    sim = ml.simulate(tmin=tmin, tmax=tmax)
    axes[0].plot(obs.index, obs.values, label="observations", color="k")
    axes[0].plot(
        sim.index,
        sim.values,
        label=f"simulation (R$^2$={ml.stats.rsq():.3f})",
        color=color,
    )
    axes[0].set_ylabel("[m NAP]")

    if color is None:
        color = "C0"

    res = ml.residuals(tmin=tmin, tmax=tmax)
    axes[1].hlines(res.mean(), res.index[0], res.index[-1], color="k", label="mean")
    if add_contributions is not None:
        for contribution in add_contributions:
            c = ml.get_contribution(contribution)
            # interpolate to match the residuals index
            c = c.reindex(c.index.union(res.index)).interpolate().loc[res.index]
            res += c
        # axes[1].plot(res.index, res.values, label="residuals+contributions", color="C3")
    axes[1].plot(res.index, res.values, label="residuals", color=color)
    axes[1].set_ylabel("[m]")

    add_to_legend = True
    for i, irow in df.iterrows():
        axes[2].plot(
            [irow["start"], irow["end"]],
            [irow["Δmean"], irow["Δmean"]],
            color=color,
            label="trend" if i == 0 else None,
            lw=2.0,
        )
        if irow["reference"] != "*":
            # axes[2].plot(
            #     [irow["start"], irow["end"]],
            #     [irow["Δmean"] - irow["ci"]] * 2,
            #     color="C3",
            #     linestyle="dashed",
            #     label="confidence interval (95%)" if add_to_legend else None,
            # )
            # axes[2].plot(
            #     [irow["start"], irow["end"]],
            #     [irow["Δmean"] + irow["ci"]] * 2,
            #     color="C3",
            #     linestyle="dashed",
            # )
            axes[2].fill_between(
                [irow["start"], irow["end"]],
                irow["Δmean"] - irow["ci"],
                irow["Δmean"] + irow["ci"],
                alpha=0.2,
                label="confidence interval (95%)" if add_to_legend else None,
                color=color,
            )
            add_to_legend = False
    axes[2].set_ylabel("[m]")

    for iax in axes:
        for t in df["start"]:
            iax.axvline(t, color="k", ls="dashed", lw=1.0)
        iax.axvline(df["end"].iloc[-1], color="k", ls="dashed", lw=1.0)
        iax.legend(loc=(0, 1), frameon=False, ncol=3, fontsize="small")
        iax.grid(True)
    axes[0].set_xlim(tmin, tmax + pd.offsets.Day())
    f.align_ylabels()
    return axes


def histogram(
    trends,
    method: str,
    bins: int | np.ndarray,
    iper=1,
    cmap=None,
    title=None,
    figsize=(10, 6),
    **kwargs,
):
    """Plot histogram of trends.

    Parameters
    ----------
    trends : list of pandas.DataFrame
        List of DataFrames with trends for each location.
    method : str
        Method used for trend analysis, e.g., "A", "B", or "C, used in title.
    bins : int or np.ndarray
        Number of bins or array of bin edges for the histogram.
    iper : int, optional
        Index of the period to plot (default is 1).
    cmap : str or matplotlib colormap, optional
        Colormap to use for the histogram. If None, "RdYlBu" is used.
    title : str, optional
        Title for the plot. If None, an empty string is used. The method and number
        of series are appended to the title.
    figsize : tuple, optional
        Size of the figure (default is (10, 6)).
    **kwargs : dict, optional


    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the histogram plot.
    """
    dmeans = pd.concat(
        [t.loc[[iper], "Δmean"] for t in trends],
        axis=1,
        keys=[t.index.name for t in trends],
    ).T

    _, ax = plt.subplots(figsize=figsize)

    if isinstance(bins, int):
        v = dmeans.squeeze().abs().max()
        bins = np.linspace(-v, v, bins + 1)

    if cmap is None:
        cmap = plt.get_cmap("RdYlBu")
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors = cmap(np.linspace(0, 1, len(bins) - 1))
    _, _, patches = ax.hist(
        dmeans,
        bins=bins,
        rwidth=kwargs.pop("rwidth", 0.9),
        edgecolor=kwargs.pop("edgecolor", "k"),
        align=kwargs.pop("align", "mid"),
        **kwargs,
    )
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    ax.grid(True)
    ax.set_ylabel("Number of series [-]")
    ax.set_xlabel("Trend [m]")
    if title is None:
        title = ""
    title = f"{title}(methode {method}, n={dmeans.index.size})"
    ax.set_title(title)
    return ax


def map_trends(
    x,
    y,
    trends,
    bins,
    method,
    iper=1,
    title=None,
    annotate=True,
    figsize=(10, 6),
    **kwargs,
):
    """Plot trends on a map.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates of the locations (in meters RD).
    y : np.ndarray
        Y-coordinates of the locations (in meters RD).
    trends : list of pandas.DataFrame
        List of DataFrames with trends for each location.
    bins : int or np.ndarray
        Number of bins or array of bin edges for the color mapping.
    method : str
        Method used for trend analysis, e.g., "A", "B", or "C
    iper : int, optional
        Index of the period to plot (default is 1).
    title : str, optional
        Title for the plot. If None, an empty string is used. The method and number
        of series are appended to the title.
    annotate : bool, optional
        Whether to annotate the points with trend values (default is True).
    figsize : tuple, optional
        Size of the figure (default is (10, 6)).
    **kwargs : dict, optional
        Additional keyword arguments passed to the scatter plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the map plot.
    """
    dmean = np.array([tr_i.loc[iper, "Δmean"] for tr_i in trends])
    if isinstance(bins, int):
        v = dmean.squeeze().abs().max() * 100
        bins = np.linspace(-v, v, bins + 1)
    cmap = plt.get_cmap("RdYlBu")
    norm = mpl.colors.BoundaryNorm(bins, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    sm = ax.scatter(
        x,
        y,
        c=dmean * 100.0,
        cmap=cmap,
        norm=norm,
        s=kwargs.pop("s", 70),
        edgecolors=kwargs.pop("edgecolors", "k"),
        linewidths=kwargs.pop("linewidths", 0.5),
        **kwargs,
    )
    if annotate:
        for ix, iy, idmean in zip(x, y, dmean):
            value = idmean * 100  # convert to cm
            color = cmap(norm(value))
            # Calculate luminance to decide text color
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.annotate(
                f"{value:.0f}",
                (ix, iy),
                color=text_color,
                fontsize=5,
                ha="center",
                va="center",
            )
    fig.colorbar(sm, ax=ax, pad=0.02, label="Trend [cm]")
    plt.yticks(rotation=90, va="center")
    ax.set_xlabel("X [m RD]")
    ax.set_ylabel("Y [m RD]")
    ax.grid(color="k", linestyle=":", alpha=0.5)
    if title is None:
        title = ""
    ax.set_title(
        f"{title} (methode {method}, " f"n={len(trends)})",
        fontsize=10,
    )
    return ax
