

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import fmEphys as fme
import saccadeAnalysis as sacc


def make_hffm_dataset(session_dict, savepath):

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


    # Determine gratings-responsive cells
    for ind, row in data.iterrows():

        sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        sp = len(row['Gt_spikeT'])
        data.at[ind, 'Gt_fr'] = sp/sec
        
        data.at[ind, 'raw_mod_for_Gt'] = sacc.calc_PSTH_modind(
                                                row['Gt_grating_psth'],
                                                trange='gt')
        
        data.at[ind, 'norm_mod_for_Gt'] = sacc.calc_PSTH_modind(
                                                row['norm_gratings_psth'],
                                                trange='gt')

    data['Gt_responsive'] = False

    for ind, row in data.iterrows():
        if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):

            data.at[ind, 'Gt_responsive'] = True

    # Reversing checkerboard responsive
    for ind, row in data.iterrows():

        _end = row['Rc_eyeT'][-1].astype(float)
        _start = row['Rc_eyeT'][0].astype(float)

        sp = len(row['Rc_spikeT'])
        data.at[ind, 'Rc_fr'] = sp / (_end - _start)

        data.at[ind, 'raw_mod_for_Rc'] = sacc.calc_PSTH_modind(
                                                    row['Rc_psth'],
                                                    trange='fm')

        data.at[ind, 'norm_mod_for_Rc'] = sacc.calc_PSTH_modind(row['norm_Rc_psth'])
    
        # Latency
        Rc_peakT, Rc_peak_val = sacc.calc_PSTH_latency(row['norm_Rc_psth'])
        data.at[ind, 'rc_peakT'] = Rc_peakT
        data.at[ind, 'rc_peak_val'] = Rc_peak_val

    data['Rc_responsive'] = False
    for ind, row in data.iterrows():
        if (row['raw_mod_for_Rc']>1) and (row['norm_mod_for_Rc']>0.1):
            data.at[ind, 'Rc_responsive'] = True

    # Sparse noise responsive
    for ind, row in data.iterrows():
        sec = row['Sn_eyeT'][-1].astype(float) - row['Sn_eyeT'][0].astype(float)
        sp = len(row['Sn_spikeT'])
        data.at[ind, 'Sn_fr'] = sp/sec
        
        data.at[ind, 'raw_mod_for_Sn'] = sacc.calc_PSTH_modind(
                                                row['Sn_on_background_psth'],
                                                baseval='sn')

        data.at[ind, 'norm_mod_for_Sn'] = sacc.calc_PSTH_modind(
                                                row['norm_Sn_psth'],
                                                trange='sn')
        
    data['Sn_responsive'] = False
    for ind, row in data.iterrows():
        if (row['raw_mod_for_Sn']>1) and (row['norm_mod_for_Sn']>0.1):
            data.at[ind, 'Sn_responsive'] = True

    # Sparse noise latency
    for ind, row in data.iterrows():
        Sn_peakT, Sn_peak_val = sacc.calc_PSTH_latency(row['norm_Sn_psth'])
        data.at[ind, 'sn_peakT'] = Sn_peakT
        data.at[ind, 'sn_peak_val'] = Sn_peak_val

    # Temporal sequences
    data.at[ind, 'rc_peakT'] = Rc_peakT
    data.at[ind, 'sn_peakT'] = Sn_peakT

    use_cols = ['FmLt_gazeshift_peakT','gazecluster','pref_gazeshift_psth',
                'nonpref_gazeshift_psth','Rc_responsive','Sn_responsive',
                'pref_comp_psth','Gt_responsive','nonpref_comp_psth',
                'norm_Rc_psth','norm_Sn_psth','tf_pref_cps',
                'sf_pref_cpd','gazeshift_responsive']
        
    sorted_df = data[use_cols].copy()
    tseq_unresp = sorted_df.copy()
    tseq_unresp = sorted_df[sorted_df['gazecluster']=='unresponsive'][sorted_df['gazeshift_responsive']==False].sample(frac=1).reset_index(drop=True)
    tseq_unresp_pref = fme.flatten_series(tseq_unresp['pref_gazeshift_psth'])
    tseq_unresp_nonpref = fme.flatten_series(tseq_unresp['nonpref_gazeshift_psth'])
    tseq_unresp_comp = fme.flatten_series(tseq_unresp['pref_comp_psth'])
    tseq_unresp_rc = fme.flatten_series(tseq_unresp['norm_Rc_psth'][sorted_df['Rc_responsive']])
    tseq_unresp_sn = fme.flatten_series(tseq_unresp['norm_Sn_psth'][sorted_df['Sn_responsive']])

    sorted_df.sort_values(by='FmLt_gazeshift_peakT', inplace=True)
    sorted_df = sorted_df[sorted_df['gazecluster']!='unresponsive'][sorted_df['gazeshift_responsive']==True].reset_index()
    tseq_pref = fme.flatten_series(sorted_df['pref_gazeshift_psth'].copy())
    tseq_nonpref = fme.flatten_series(sorted_df['nonpref_gazeshift_psth'].copy())
    tseq_comp = fme.flatten_series(sorted_df['pref_comp_psth'].copy())
    tseq_rc = fme.flatten_series(sorted_df['norm_Rc_psth'][sorted_df['Rc_responsive']].copy())
    tseq_sn = fme.flatten_series(sorted_df['norm_Sn_psth'][sorted_df['Sn_responsive']].copy())
    tseq_grat_tf = sorted_df['tf_pref_cps'][sorted_df['Gt_responsive']].copy().to_numpy()
    tseq_grat_sf = sorted_df['sf_pref_cpd'][sorted_df['Gt_responsive']].copy().to_numpy()

    # Stack with unresponsive cells.
    tseq_pref1 = np.vstack([tseq_pref, tseq_unresp_pref])
    tseq_nonpref1 = np.vstack([tseq_nonpref, tseq_unresp_nonpref])
    tseq_comp1 = np.vstack([tseq_comp, tseq_unresp_comp])
    tseq_rc1 = np.vstack([tseq_rc, tseq_unresp_rc])
    tseq_sn1 = np.vstack([tseq_sn, tseq_unresp_sn])

    props = sacc.propsdict()
    colors = props['colors']

    tseq_legend_col = sorted_df['gazecluster'].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = colors[n]
    ucmap = mpl.colors.to_rgba(colors['unresponsive'])
    u = np.zeros([np.size(tseq_unresp_pref,0), 1, 4])
    for x in range(4):
        u[:,:,x] = ucmap[x]
    tseq_legend1 = np.vstack([tseq_legend, u])

    