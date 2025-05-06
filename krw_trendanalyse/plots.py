import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import CenteredNorm


def plot_mean_per_period(
    series,
    df,
    color_method="significance",
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
    f, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(series.index, series.values, label=series.name, color="k", lw=1.0)

    add_to_legend = True
    for i, irow in df.iterrows():
        if irow["reference"] == "*":
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

            ax.plot(
                [irow["start"], irow["end"]],
                [irow["mean"], irow["mean"]],
                color=color,
                label="mean" if irow["reference"] == "*" else None,
                lw=2.0,
            )
            ax.plot(
                [irow["start"], irow["end"]],
                [irow["mean"] - irow["ci"]] * 2,
                color=color,
                linestyle="dashed",
                label="confidence interval (95%)" if add_to_legend else None,
            )
            ax.plot(
                [irow["start"], irow["end"]],
                [irow["mean"] + irow["ci"]] * 2,
                color=color,
                linestyle="dashed",
            )
            add_to_legend = False

    ax.set_xlim(tmin, tmax + pd.offsets.Day())
    ax.grid(True)
    for t in df["start"]:
        ax.axvline(t, color="k", ls="dashed", lw=1.0)
    ax.axvline(df["end"].iloc[-1], color="k", ls="dashed", lw=1.0)
    ax.legend(loc=(0, 1), frameon=False, ncol=4, fontsize="small")
    return ax


def plot_model_residuals_summary(ml, df, add_contributions=None):
    tmin = df["start"].min()
    tmax = df["end"].max()

    f, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

    obs = ml.observations(tmin=tmin, tmax=tmax)
    sim = ml.simulate(tmin=tmin, tmax=tmax)
    axes[0].plot(obs.index, obs.values, label="observations", color="k")
    axes[0].plot(
        sim.index, sim.values, label=f"simulation (R$^2$={ml.stats.rsq():.3f})"
    )
    axes[0].set_ylabel("[m NAP]")

    res = ml.residuals(tmin=tmin, tmax=tmax)
    axes[1].plot(res.index, res.values, label="residuals")
    axes[1].hlines(res.mean(), res.index[0], res.index[-1], color="k", label="mean")
    if add_contributions is not None:
        for contribution in add_contributions:
            c = ml.get_contribution(contribution)
            res += c
        axes[1].plot(res.index, res.values, label="residuals+contributions", color="C3")
    axes[1].set_ylabel("[m]")

    add_to_legend = True
    for i, irow in df.iterrows():
        axes[2].plot(
            [irow["start"], irow["end"]],
            [irow["Δmean"], irow["Δmean"]],
            color="C0",
            label="trend" if i == 0 else None,
            lw=2.0,
        )
        if irow["reference"] != "*":
            axes[2].plot(
                [irow["start"], irow["end"]],
                [irow["Δmean"] - irow["ci"]] * 2,
                color="C3",
                linestyle="dashed",
                label="confidence interval (95%)" if add_to_legend else None,
            )
            axes[2].plot(
                [irow["start"], irow["end"]],
                [irow["Δmean"] + irow["ci"]] * 2,
                color="C3",
                linestyle="dashed",
            )
            add_to_legend = False
    axes[2].set_ylabel("[m]")

    for iax in axes:
        for t in df["start"]:
            iax.axvline(t, color="k", ls="dashed", lw=1.0)
        iax.axvline(df["end"].iloc[-1], color="k", ls="dashed", lw=1.0)
        iax.legend(loc=(0, 1), frameon=False, ncol=4, fontsize="small")
        iax.grid(True)
    axes[0].set_xlim(tmin, tmax + pd.offsets.Day())
    f.align_ylabels()
    return axes
