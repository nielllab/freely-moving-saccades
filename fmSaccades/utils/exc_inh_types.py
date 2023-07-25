


import numpy as np
import sklearn.cluster


def putative_cell_type(data):
    """
    
    """

    data['norm_waveform'] = data['waveform'].copy()

    for ind, row in data.iterrows():

        if type(row['waveform']) == list:

            starting_val = np.mean(row['waveform'][:6])

            center_waveform = [i-starting_val for i in row['waveform']]
            norm_waveform = center_waveform / -np.min(center_waveform)

            data.at[ind, 'waveform_trough_width'] = len(norm_waveform[norm_waveform < -0.2])
            data.at[ind, 'AHP'] = norm_waveform[27]
            data.at[ind, 'waveform_peak'] = norm_waveform[18]
            data.at[ind, 'norm_waveform'] = norm_waveform

    # Cluster into two groups
    km = sklearn.cluster.KMeans(n_clusters=2)
    km.fit(list(data['norm_waveform'][data['waveform_peak'] < 0].to_numpy()))
    km_labels = km.labels_

    # Make inhibitory is always group 0. Excitatory should always
    # have a smaller mean waveform trough. If it's larger, flip
    # the kmeans labels.
    km0_width = np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==0])
    km1_width = np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==1])
    if km0_width > km1_width:
        km_labels = [0 if i==1 else 1 for i in km_labels]

    # Add labels of 0 or 1 to data
    data['waveform_km_label'] = np.nan
    count = 0
    for ind, row in data.iterrows():
        if row['waveform_peak'] < 0 and row['AHP'] < 0.7:
            data.at[ind, 'waveform_km_label'] = km_labels[count]
            count = count+1

    # Also add the data as a column of strings, either 'exc'
    # or 'inh'.
    data['exc_or_inh'] = np.nan
    for ind, row in data.iterrows():
        if row['waveform_km_label'] == 0:
            data.at[ind, 'exc_or_inh'] = 'inh'
        elif row['waveform_km_label'] == 1:
            data.at[ind, 'exc_or_inh'] = 'exc'

    return data