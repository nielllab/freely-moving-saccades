


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import saccadeAnalysis as sacc


def main(savepath):

    data = pd.DataFrame()

    # Determine gazeshift responsive cells
    for ind, row in data.iterrows():

        # firing rate
        sec = row['FmLt_eyeT'][-1].astype(float) - row['FmLt_eyeT'][0].astype(float)
        sp = len(row['FmLt_spikeT'])
        fm_fr = sp/sec
        data.at[ind, 'Fm_fr'] = fm_fr
        
        raw_psth = row['pref_gazeshift_psth_raw']
        data.at[ind, 'raw_mod_at_pref_peak'] = sacc.calc_PSTH_modind(raw_psth)
        
        norm_psth = row['pref_gazeshift_psth']
        data.at[ind, 'norm_mod_at_pref_peak'] = sacc.calc_PSTH_modind(norm_psth)

    data['gazeshift_responsive'] = False
    for ind, row in data.iterrows():

        if (row['raw_mod_at_pref_peak'] > 1) and (row['norm_mod_at_pref_peak'] > 0.1):
            data.at[ind, 'gazeshift_responsive'] = True

    # Format inputs for clustering
    pca_input = sacc.make_cluster_model_input(data)

    # Cluster on the data
    labels, _km_opts = sacc.make_clusters(pca_input, savepath)

    # proj = _km_opts['proj']
    # plt.scatter(proj[:,0], proj[:,1], c=labels)

    # Add labels to the dataset, which will open a figure and requires
    # user input in a series of popup windows to determine the string
    # labels for clusters.
    data = sacc.add_labels_to_dataset(data, labels)















## gratings
for ind, row in data.iterrows():

    sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
    sp = len(row['Gt_spikeT'])
    data.at[ind, 'Gt_fr'] = sp/sec
    
    data.at[ind, 'raw_mod_for_Gt'] = gt_modind(row['Gt_grating_psth'])
    
    data.at[ind, 'norm_mod_for_Gt'] = gt_modind(row['norm_gratings_psth'])

# gratings responsive
data['Gt_responsive'] = False
for ind, row in data.iterrows():
    if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):
        data.at[ind, 'Gt_responsive'] = True