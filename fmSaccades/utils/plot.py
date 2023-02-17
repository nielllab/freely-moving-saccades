
import os
import sys
import json
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import scipy.stats

from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size':10})


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



subdirs = list_subdirs(new_dir, givepath=True)
usable_recordings = ['fm1','fm1_dark','fm_dark','hf1_wn','hf2_sparsenoiseflash','hf3_gratings','hf4_revchecker']
subdirs = [p for p in subdirs if any(s in p for s in usable_recordings)]
print(subdirs)

df = pd.DataFrame([])
for path in subdirs:
    ephys_path = find('*_ephys_props.h5',path)[0]
    print(ephys_path)
    rec_data = pd.read_hdf(ephys_path)
    rec_type = '_'.join(([col for col in rec_data.columns.values if 'contrast_tuning_bins' in col][0]).split('_')[:-3])
    rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT',
                                        'spikeTraw':rec_type+'_spikeTraw',
                                        'rate':rec_type+'_rate',
                                        'n_spikes':rec_type+'_n_spikes'})
    # get column names
    column_names = list(df.columns.values) + list(rec_data.columns.values)
    # new columns for same unit within a session
    df = pd.concat([df, rec_data],axis=1,ignore_index=True)
    # add the list of column names from all sessions plus the current recording
    df.columns = column_names
    # remove duplicate columns (i.e. shared metadata)
    df = df.loc[:,~df.columns.duplicated()]

    ellipse_json_path = find('*fm_eyecameracalc_props.json', new_dir)[0]
print(ellipse_json_path)
with open(ellipse_json_path) as f:
    ellipse_fit_params = json.load(f)
df['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
df['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']

df['original_session_path'] = new_dir
df['probe_name'] = probe

df['index'] = df.index.values
df.reset_index(inplace=True)