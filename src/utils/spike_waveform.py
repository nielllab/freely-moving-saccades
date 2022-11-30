"""

"""


import os
import pickle
import numpy as np
from sklearn.cluster import KMeans

import fmEphys


def label_spike_waveform(data, savedir):
    """
    data is a dataframe

    Use ephys spike waveforms to cluster excitatory and inhibitory cell types
    """

    data['norm_waveform'] = data['waveform'].copy()

    for ind, row in data.iterrows():
        if type(row['waveform']) == list:

            # Subtract baseline value
            basesub_waveform = [i-np.mean(row['waveform'][:6]) for i in row['waveform']]

            # Normalize relative to negative deflection
            norm_waveform = basesub_waveform / -np.min(basesub_waveform)

            # Time gap between falling and rising voltage
            data.at[ind, 'waveform_trough_width'] = len(norm_waveform[norm_waveform < -0.2])

            # Afterhyperpolarization
            data.at[ind, 'waveform_AHP'] = norm_waveform[27]
            # Peak value
            data.at[ind, 'waveform_peak'] = norm_waveform[18]
            # Normalized waveform
            data.at[ind, 'norm_waveform'] = norm_waveform

    # Cluster waveforms using KMeans (k=2) to split narrow and broad spiking waveforms
    # apart.
    km_input = list(data['norm_waveform'][data['waveform_peak'] < 0].to_numpy())
    km_model = KMeans(n_clusters=2).fit(km_input)

    # Save the model out
    date, time = fmEphys.fmt_now()
    savename = 'spike_waveform_KMeans_{}_{}.pickle'.format(date, time)
    savepath = os.path.join(savedir, savename)
    with open(savepath, 'wb') as f:
        pickle.dump(km_model, f)

    # Cells are labeled 0 or 1, but the cluster label is assigned randomly between narrow
    # and broad.
    km_labels = km_model.labels_

    # Set inhibitory cluster to always be group 0. To do this, we will compare the mean
    # waveform trough width. Excitatory waveforms should always have a smaller mean
    # waveform trough, so if cluster 0 has a smaller average, flip the KMeans labels.
    cluster0_trough = np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==0])
    cluster1_trough = np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==1])
    if cluster0_trough > cluster1_trough:
        km_labels = [0 if i==1 else 1 for i in km_labels]

    # Now, add the KMeans properties to the dataset.
    data['waveform_km'] = np.nan
    _use = (data['waveform_peak'] < 0) * (data['waveform_AHP'] < 0.7)
    for i, ind in enumerate(data[_use].index.values):
        data.at[ind, 'waveform_km'] = km_labels[i]

    # Make new column of strings for excitatory vs inhibitory putative cell types
    data['putative_celltype'] = np.nan
    for ind, row in data.iterrows():
        if row['waveform_km'] == 0:
            data.at[ind, 'putative_celltype'] = 'inh'
        elif row['waveform_km'] == 1:
            data.at[ind, 'putative_celltype'] = 'exc'

    return data