

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fmEphys as fme
import saccadeAnalysis as sacc


def HfFm(session_dict, savepath):

    _saveas = os.path.join(savepath,
                           'HfFm_plain_dataset_{}.h5'.format(fme.fmt_now(c=True)))
    
    print('Creating HfFm dataset.')
    data = sacc.create_dataset(session_dict, _saveas)

    # Normalize PSTHs for the light condition.
    data = sacc.get_norm_FmLt_PSTHs(data)

    # Normalize for head-fixed data.
    data = sacc.get_norm_Hf_PSTHs(data)


    # Gazeshift latency
    for ind, row in data.iterrows():
        raw_psth = row['pref_gazeshift_psth_raw']
        norm_psth = row['pref_gazeshift_psth']
        
        peakT, peak_val = sacc.calc_PSTH_latency(norm_psth)
        
        data.at[ind, 'FmLt_gazeshift_baseline'] = raw_psth[0:800].astype(object)
        data.at[ind, 'FmLt_gazeshift_med_baseline'] = np.median(raw_psth[0:800])
        data.at[ind, 'FmLt_gazeshift_peak_val'] = peak_val
        data.at[ind, 'FmLt_gazeshift_peakT'] = peakT

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