
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.stats


def set_plt_params():
    
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams.update({'font.size':10})


def to_color(r,g,b):
    return (r/255, g/255, b/255)


def make_colors():

    plasma_map = plt.cm.plasma(np.linspace(0,1,15))

    cmap = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey',
        'gaze': 'firebrick',
        'comp': 'mediumblue',
        'rc': 'indigo'
    }

    return cmap


def jitter_ax(center, size):

    return np.ones(size)+np.random.uniform(center-0.2, center+0.2, size)


def plot_linregress1(ax, x_in, y_in):

    x = x_in[(~np.isnan(x_in)) * (~np.isnan(y_in))]
    y = y_in[(~np.isnan(x_in)) * (~np.isnan(y_in))]

    res = scipy.stats.linregress(x, y)

    minval = np.min(x)
    maxval = np.max(x)

    plotx = np.linspace(0, maxval, 2)

    ax.plot(plotx,
            (res.slope*plotx) + res.intercept,
            'k--', linewidth=1)
            
    return res


def running_median(ax, x, y, n_bins=7, color='k'):
    """
    ax is the mpl panel
    """
    # Drop any NaNs (important!)
    usex = x[~np.isnan(x) & ~np.isnan(y)].copy()
    usey = y[~np.isnan(x) & ~np.isnan(y)].copy()

    # Set up binning
    bins = np.linspace(np.min(x), np.max(x), n_bins)

    # Calculate median
    binmed, edges, _ = scipy.stats.binned_statistic(usex, usey,
                                                    statistic=np.median, bins=bins)

    # Calculate the standrd deviation along the same bins
    binstd, _, _ = scipy.stats.binned_statistic(usex, usey,
                                                 statistic=np.nanstd, bins=bins)

    # Need the number of samples in each bin
    hist, _ = np.histogram(usex, bins=bins)

    # Calculate the error for each bin
    tuning_err = binstd / np.sqrt(hist)

    # Now, plot the value for each bin
    # and put error bars in the form of a filling with an alpha value
    use_xplot_vals = edges[:-1] + (np.median(np.diff(bins))/2)

    ax.plot(use_xplot_vals, binmed, '-', color=color)
    ax.fill_between(use_xplot_vals,
                    binmed-tuning_err, binmed+tuning_err,
                    color=color, alpha=0.2)
