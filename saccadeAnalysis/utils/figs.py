"""
fmEphys/plot/axs.py

Functions for plotting on axes.

Functions
---------
plot_tuning
    Plot the tuning curve for one cell.
plot_columns
    Categorical column scatter plot.
plot_PSTH_heatmap
    Plot a heatmap of normalized PSTHs.
plot_regression
    Plot a linear regression.
plot_running_median
    Plot median of a dataset along a set of horizontal bins.
    
    
Written by DMM, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import saccadeAnalysis as sacc
import fmEphys as fme


def plot_tuning(ax, bins, tuning, error,
              label=None, unum=None, ylim=None):
    """ Plot the tuning curve for one cell.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    bins : np.array
        The bin centers for the tuning curve.
    tuning : np.array
        The tuning curve values. Should be 1D and have the same
        length as `bins`.
    error : np.array
        Error bars for the tuning curve. Should be 1D and have
        the same length as `bins`.
    label : str, optional
        The label for the x-axis.
    ylim : float, optional
        The y-axis limit. If None, the limit is set to 1.2 times
        the maximum value of `tuning`.
    
    Example use
    -----------
    fme.plot.tuning_ax(axs[i], bins, tuning[i,:], error[i,:])

    """

    ax.errorbar(bins, tuning, yerr=error)

    if ylim is None:
        try:
            ax.set_ylim(0, np.nanmax(tuning*1.2))
        except ValueError:
            ax.set_ylim(0,1)
    elif ylim is not None:
        ax.set_ylim([0,ylim])

    # Set x-axis limits
    bin_sz = np.nanmedian(np.diff(bins))
    llim = bins[0]-(bin_sz/2)
    ulim = bins[-1]+(bin_sz/2)

    ax.set_xlim([llim, ulim])
    ax.set_ylabel('sp/sec')

    if label is not None:
        ax.set_xlabel(label)


def plot_columns(ax, df, prop, cat=None, cats=None,
                colors=None, use_median=False):
    """ Categorical column scatter plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.

    
    """

    if cats is None and cat is None:

        cats = ['early','late','biphasic','negative']
        cat = 'gazecluster'

        _props = fme.props()
        colors = _props['colors']

    for c_i, c in enumerate(cats):

        cdata = df[prop][df[cat]==c]

        x_jitter = sacc.jitter(c_i, np.size(cdata,0))

        ax.plot(x_jitter, cdata-1,
                '.', color=colors[c], markersize=2)
        
        # Either use median or mean of the data
        if use_median:
            hline = np.nanmedian(cdata)
        elif not use_median:
            hline = np.nanmean(cdata)
        
        ax.hlines(hline, c_i-0.2, c_i+0.2,
                  color='k', linewidth=2)

        err = fme.stderr(cdata)
        
        ax.vlines(c_i, hline-err, hline+err,
                  color='k', linewidth=2)
        
    ax.set_xticks(range(len(cats)), cats)


def plot_PSTH_heatmap(ax, tseq,
                cscale=0.75):
    """ Plot a heatmap of normalized PSTHs.

    with shape (n_cells, n_timepoints)
    
    """

    ax.set_xlabel('time (msec)')
    ax.set_ylabel('cells')
    ax.set_ylim([np.size(tseq,0), 0])

    img = ax.imshow(tseq, cmap='coolwarm', vmin=-cscale,
                    vmax=cscale)
    
    if np.size(tseq,1)==2001:
        psth_bins = np.arange(-1., 1., 1/1000)
        winStart = 800 # 1000-200
        winEnd = 1400 # 1000+400
        cent = 1000

    elif np.size(tseq,1)==3001:
        psth_bins = np.arange(-1.5, 1.5, 1/1000)
        winStart = 1300 # 1500-200
        winEnd = 1900 # 1500+400
        cent = 1500

    ax.set_xlim([winStart,winEnd])

    ax.set_xticks(np.linspace(winStart, winEnd, 4),
                  labels=np.linspace(-200, 400, 4).astype(int))
    
    ax.vlines(cent, 0, np.size(tseq,0), color='k',
              linestyle='dashed', linewidth=1)

    return img


def plot_regression(ax, x_in, y_in):
    """ Plot a linear regression.
    
    """

    use_inds = (~np.isnan(x_in)) * (~np.isnan(y_in))

    x = x_in[use_inds]
    y = y_in[use_inds]

    res = scipy.stats.linregress(x, y)

    minval = np.min(x)
    maxval = np.max(x)

    plotx = np.linspace(0, maxval, 2)

    ax.plot(plotx, (res.slope*plotx) + res.intercept,
            'k--', linewidth=1)

    return res


def plot_running_median(ax, x, y, n_bins=7):
    """ Plot median of a dataset along a set of horizontal bins.
    
    """

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
               bin_means,
               '-', color='k')
    
    ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                       bin_means-tuning_err,
                       bin_means+tuning_err,
                       color='k', alpha=0.2)






# def plot_tempseq(panel, tseq, return_img=False, freev=None):
#     panel.set_xlabel('time (msec)')
#     panel.set_ylim([np.size(tseq,0),0])
#     vmin = -0.75; vmax = 0.75
#     if freev is not None:
#         vmin = -freev
#         vmax = freev
#     img = panel.imshow(tseq, cmap='coolwarm', vmin=vmin, vmax=vmax)
#     panel.set_xlim([800,1400])
#     panel.set_xticks(np.linspace(800,1400,4), labels=np.linspace(-200,400,4).astype(int))
#     panel.vlines(1000, 0, np.size(tseq,0), color='k', linestyle='dashed', linewidth=1)
#     if return_img:
#         return img
#
# def plot_cprop_scatter(panel, data, prop_name, use_median=False):
#     cmap = fms.make_colors()
#     for c, cluster in enumerate(['early','late','biphasic','negative']):
#         cluster_data = data[prop_name][data['gazecluster']==cluster]
#         x_jitter = np.random.uniform(c-0.2, c+0.2, np.size(cluster_data,0))
#         panel.plot(x_jitter, cluster_data, '.', color=cmap[cluster], markersize=2)
#         if use_median:
#             hline = np.nanmedian(cluster_data)
#         elif not use_median:
#             hline = np.nanmean(cluster_data)
#         panel.hlines(hline, c-0.2, c+0.2, color='k', linewidth=2)
#         err = np.std(cluster_data) / np.sqrt(np.size(cluster_data))
#         panel.vlines(c, hline-err, hline+err, color='k', linewidth=2)
#         panel.set_xticks(range(4), ['early','late','biphasic','negative'])
