"""
saccadeAnalysis/utils/dark.py

Dark analysis.

Written by DMM 2022
Last modified March 2024
"""


import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

import fmEphys as fme
import saccadeAnalysis as sacc


def make_ltdk_dataset(savepath, session_dict=None, ltdk_path=None,
                      km_model=None, pca_model=None):
    """ Make light/dark dataset.

    Parameters
    ----------
    savepath : str
        Path to save the dataset.
    session_dict : dict
        Dictionary containing the info for individual recording HDF
        files to be combined into a single dataset. For expected
        key:val pairs, see the function `sacc.stack_dataset()`.
    ltdk_path : str
        Path to a pre-existing LtDk dataset to read in (instead of
        creating a new one with the recordings itemized in `session_dict`).
    km_model : str
        Filepath to a saved KMeans clustering model (which must have been
        written using the HfFm dataset and saved to disk as a pickle file).
    pca_model : str
        Same as `km_model` but for the PCA model.
    
    Returns
    -------
    data: pd.DataFrame
        LtDk dataset in whcih each index is a cell and each column is a
        properties of cells, either as values or arrays inserted as objects.
        A plain version of this (read in data without any subsequent analyses)
        is saved to disk as an HDF file. The returned data, including the results
        of subsequenta analyses is NOT saved by the function, and should be
        saved as an HDF file if desired.
    out: dict
        Dictionary containing various arrays and values for plotting and which
        summarize data across cells. These are mostly related to temopral sequences
        of PSTHs (that can be plotted as heatmaps). Keys are: 'light_pref', 'light_nonpref',
        'dark_pref', 'dark_nonpref', 'tseq_dark_by_dark', 'tseq_pref_light_by_light',
        'tseq_pref_dark_by_light', 'tseq_nonpref_dark_by_light', 'tseq_comp_dark_by_light',
        'tseq_pref_light_by_light_w_unresp', 'tseq_pref_dark_by_light_w_unresp',
        'tseq_nonpref_dark_by_light_w_unresp', 'tseq_comp_dark_by_light_w_unresp',
        'tseq_legend', 'tseq_legend_w_unresp'.
    """
    
    _saveas = os.path.join(savepath,
                           'LtDk_plain_dataset_{}.h5'.format(fme.fmt_now(c=True)))
    
    # Read individual HDF files and make a new LtDk dataset.
    if ltdk_path is None and session_dict is not None:
        print('Creating LtDk dataset.')
        data = sacc.create_dataset(session_dict, _saveas)

    # OR: Read a pre-existing LtDk dataset.
    if ltdk_path is not None and session_dict is None:
        print('Reading LtDk dataset.')
        data = fme.read_group_h5(ltdk_path)

    # First, run analysis on freely moving light condition
    data = sacc.get_norm_FmLt_PSTHs(data)

    # Normalize PSTHs
    data = sacc.get_norm_FmDk_PSTHs(data)

    # Calculate latencies
    data = sacc.FmLtDk_peak_time(data)

    # Plotting props
    sacc.set_plt_params()
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']

    ###  Gazeshift clusters

    # Create input array of data
    pca_input = sacc.make_cluster_model_input(data)

    # Apply the EXISTING clustering model (calculated on hffm dataset)
    clustering_labels, _ = sacc.apply_saved_cluster_models(pca_input, km_model, pca_model)

    # Add labels to the dataset
    data = sacc.add_labels_to_dataset(data, clustering_labels, savepath)

    # Dark modulation
    for ind, row in data.iterrows():
        norm_psth = row['pref_dark_gazeshift_psth'].copy().astype(float)
        data.at[ind, 'norm_dark_modulation'] = sacc.calc_PSTH_modind(norm_psth)
        
        raw_psth = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])].copy().astype(float)
        data.at[ind, 'dark_modulation'] = sacc.calc_PSTH_modind(raw_psth)

    plotvals = data[data['gazeshift_responsive']][data['gazecluster_ind']!=4].copy()

    light_pref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
    light_nonpref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]

    dark_pref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
    dark_nonpref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]

    for ind in plotvals.index.values:
        
        if (plotvals.loc[ind,'dark_modulation']>1) and                      \
                    (plotvals.loc[ind,'norm_dark_modulation']>0.1) and      \
                    (plotvals.loc[ind,'FmDk_gazeshift_peakT']<=0.035):
            
            data.at[ind, 'dark_responsive'] = True

    # Some light/dark locomotion calcs
    model_dt = 0.025

    for ind, row in data.iterrows():

        modelT = np.arange(0, np.nanmax(row['FmLt_eyeT']), model_dt)
        
        # timing is off sometimes... using eyeT instead of worldT to get maximum length
        # and they can be different by a few frames
        diff = len(modelT) - len(row['FmLt_rate'])
        if diff>0: # modelT is longer
            modelT = modelT[:-diff]
        elif diff<0: # modelT is shorted
            for i in range(np.abs(diff)):
                modelT = np.append(modelT, modelT[-1]+model_dt)
        
        model_gz = interp1d(row['FmLt_imuT'], row['FmLt_gyro_z'], bounds_error=False)(modelT)
        model_active = np.convolve(np.abs(model_gz),
                                   np.ones(int(1/model_dt)),
                                   'same') / len(np.ones(int(1/model_dt)))
        data.at[ind, 'FmLt_model_active'] = model_active.astype(object)
        data.at[ind, 'FmLt_modelT'] = modelT.astype(object)
        
        modelT = np.arange(0, np.nanmax(row['FmDk_eyeT']), model_dt)

        diff = len(modelT) - len(row['FmDk_rate'])
        if diff>0: # modelT is longer
            modelT = modelT[:-diff]
        elif diff<0: # modelT is shorted
            for i in range(np.abs(diff)):
                modelT = np.append(modelT, modelT[-1]+model_dt)

        model_gz = interp1d(row['FmDk_imuT'], row['FmDk_gyro_z'], bounds_error=False)(modelT)

        model_active = np.convolve(np.abs(model_gz),
                                   np.ones(int(1/model_dt)),
                                   'same') / len(np.ones(int(1/model_dt)))
        data.at[ind, 'FmDk_model_active'] = model_active.astype(object)
        data.at[ind, 'FmDk_modelT'] = modelT.astype(object)
    
    data['FmLt_fr'] = ((data['FmLt_rate'].apply(np.sum)*0.025) / data['FmLt_eyeT'].apply(np.nanmax)).to_numpy()
    data['FmDk_fr'] = ((data['FmDk_rate'].apply(np.sum)*0.025) / data['FmDk_eyeT'].apply(np.nanmax)).to_numpy()

    # FmLt_fr = np.zeros([len(data.index.values)])
    # FmDk_fr = np.zeros([len(data.index.values)])
    for ind, row in data.iterrows():
        data.at[ind,'FmLt_active_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']>40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']>40])
        data.at[ind,'FmLt_inactive_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']<40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']<40])
        data.at[ind,'FmDk_active_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']>40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']>40])
        data.at[ind,'FmDk_inactive_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']<40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']<40])
    
    # light/dark correlation
    for ind, row in data[['pref_dark_gazeshift_psth','pref_gazeshift_psth']].iterrows():
        r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['pref_dark_gazeshift_psth'].astype(float)[1000:1250])
        data.at[ind, 'gaze_ltdk_maxcc'] = r[0,1]
    
    # Head-fixed vs. gaze correlation
    # for ind, row in ltdk[['norm_Rc_psth','norm_Sn_psth','pref_gazeshift_psth']].iterrows():
        
    #     if (np.sum(~np.isnan(row['norm_Rc_psth'].astype(float)[1000]))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
    #         r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Rc_psth'].astype(float)[1000:1250])
    #         ltdk.at[ind, 'gaze_rc_maxcc'] = r[0,1]
        
    #     if (np.sum(~np.isnan(row['norm_Sn_psth'].astype(float)))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
    #         r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Sn_psth'].astype(float)[1000:1250])
    #         ltdk.at[ind, 'gaze_sn_maxcc'] = r[0,1]

    for ind in data.index.values:
        Lt_peakT, Lt_peakVal = sacc.calc_PSTH_latency(data.loc[ind,'pref_gazeshift_psth'])
        
        data.at[ind, 'FmLt_gazeshift_peakT'] = Lt_peakT

    for ind in data.index.values:
        sorted_df = data[['FmLt_gazeshift_peakT','FmDk_gazeshift_peakT','FmLt_gazeshift_troughT','FmDk_gazeshift_troughT','gazecluster',
                                'pref_gazeshift_psth','pref_dark_gazeshift_psth','nonpref_dark_gazeshift_psth','gazeshift_responsive',
                                'pref_dark_comp_psth']].copy()

    # shuffle unresponsive cells
    tseq_unresp = sorted_df.copy()
    tseq_unresp = tseq_unresp[tseq_unresp['gazecluster']=='unresponsive'].sample(frac=1).reset_index(drop=True)
    tseq_l_unresp = fme.flatten_series(tseq_unresp['pref_gazeshift_psth'].copy())
    tseq_d_unresp = fme.flatten_series(tseq_unresp['pref_dark_gazeshift_psth'].copy())
        
    # sort dark by dark times
    tseq_dark_sort = sorted_df.copy()
    tseq_dark_sort = tseq_dark_sort[tseq_dark_sort['gazecluster']!='unresponsive']
    tseq_dark_sort.sort_values(by='FmDk_gazeshift_peakT', inplace=True)

    tseq_dark_by_dark = np.vstack([fme.flatten_series(tseq_dark_sort['pref_dark_gazeshift_psth'].copy()), tseq_d_unresp])

    # sort light/dark by light times
    sort_by_light = sorted_df.copy()
    sort_by_light = sort_by_light[sort_by_light['gazecluster']!='unresponsive']
    sort_by_light.sort_values(by='FmLt_gazeshift_peakT', inplace=True)

    tseq_light = fme.flatten_series(sort_by_light['pref_gazeshift_psth'].copy())
    tseq_dark_pref = fme.flatten_series(sort_by_light['pref_dark_gazeshift_psth'].copy())
    tseq_dark_nonpref = fme.flatten_series(sort_by_light['nonpref_dark_gazeshift_psth'].copy())
    tseq_dark_comp = fme.flatten_series(sort_by_light['pref_dark_comp_psth'].copy())

    tseq_light1 = np.vstack([fme.flatten_series(sort_by_light['pref_gazeshift_psth'].copy()), tseq_l_unresp])
    tseq_dark_pref1 = np.vstack([fme.flatten_series(sort_by_light['pref_dark_gazeshift_psth'].copy()), tseq_d_unresp])
    tseq_dark_nonpref1 = np.vstack([fme.flatten_series(sort_by_light['nonpref_dark_gazeshift_psth'].copy()), tseq_d_unresp])
    tseq_dark_comp1 = np.vstack([fme.flatten_series(sort_by_light['pref_dark_comp_psth'].copy()), tseq_d_unresp])


    tseq_legend_col = sort_by_light['gazecluster'].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = mpl.colors.to_rgba(colors[n])

    ucmap = mpl.colors.to_rgba(colors['unresponsive'])
    u = np.zeros([np.size(tseq_l_unresp,0), 1, 4])
    for x in range(4):
        u[:,:,x] = ucmap[x]
    tseq_legend1 = np.vstack([tseq_legend, u])


    for ind, row in data.iterrows():

        sec = row['FmDk_eyeT'][-1].astype(float) - row['FmDk_eyeT'][0].astype(float)
        sp = len(row['FmDk_spikeT'])
        fm_fr = sp/sec

        data.at[ind, 'FmDk_fr'] = fm_fr
        
        data.at[ind, 'norm_mod_at_pref_peak_dark'] = sacc.calc_PSTH_modind(row['pref_dark_gazeshift_psth'])
        
        data.at[ind, 'raw_mod_at_pref_peak_dark'] = sacc.calc_PSTH_modind(row['pref_dark_gazeshift_psth_raw'])
        
        data.at[ind, 'norm_mod_at_pref_peak'] = sacc.calc_PSTH_modind(row['pref_gazeshift_psth'])
        
        data.at[ind, 'raw_mod_at_pref_peak'] = sacc.calc_PSTH_modind(row['pref_gazeshift_psth_raw'])
        
        peakT, peak_val = sacc.calc_PSTH_latency(row['pref_dark_gazeshift_psth'])
        data.at[ind, 'dark_peak_val'] = peak_val
        data.at[ind, 'dark_peakT'] = peakT

    vals = data[data['norm_mod_at_pref_peak_dark']>0.1][data['raw_mod_at_pref_peak_dark']>1][data['dark_peak_val']>0.5][data['dark_peakT']<.1]

    data['dark_responsive'] = False
    for ind in vals.index.values:
        data.at[ind, 'dark_responsive'] = True

    out = {
        'light_pref': light_pref,
        'light_nonpref': light_nonpref,
        'dark_pref': dark_pref,
        'dark_nonpref': dark_nonpref,

        'tseq_dark_by_dark': tseq_dark_by_dark,

        'tseq_pref_light_by_light': tseq_light,
        'tseq_pref_dark_by_light': tseq_dark_pref,
        'tseq_nonpref_dark_by_light': tseq_dark_nonpref,
        'tseq_comp_dark_by_light': tseq_dark_comp,

        'tseq_pref_light_by_light_w_unresp': tseq_light1,
        'tseq_pref_dark_by_light_w_unresp': tseq_dark_pref1,
        'tseq_nonpref_dark_by_light_w_unresp': tseq_dark_nonpref1,
        'tseq_comp_dark_by_light_w_unresp': tseq_dark_comp1,

        'tseq_legend': tseq_legend,
        'tseq_legend_w_unresp': tseq_legend1

    }

    return data, out