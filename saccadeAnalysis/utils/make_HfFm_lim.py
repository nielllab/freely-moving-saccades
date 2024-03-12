

import os
import numpy as np

import fmEphys as fme
import saccadeAnalysis as sacc


def make_hffm_dataset_onlyFmRc(savepath, session_dict=None, hffm_path=None,
                      km_model=None, pca_model=None):

    _saveas = os.path.join(savepath,
                           'HfFm_plain_dataset_{}.h5'.format(fme.fmt_now(c=True)))
    
    if hffm_path is None and session_dict is not None:
        print('Creating HfFm dataset.')
        data = sacc.create_dataset(session_dict, _saveas)

    if hffm_path is not None and session_dict is None:
        print('Reading HfFm dataset.')
        data = fme.read_group_h5(hffm_path)

    # Normalize PSTHs for the light condition.
    data = sacc.get_norm_FmLt_PSTHs(data)

    # Normalize for head-fixed data.
    #data = sacc.get_norm_Hf_PSTHs(data, onlyRc=True)


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
    if km_model is None or pca_model is None:
        labels, _km_opts = sacc.make_clusters(pca_input, savepath)

    elif km_model is not None and pca_model is not None:
        # apply existing clustering
        labels, _ = sacc.apply_saved_cluster_models(
            pca_input,
            km_model,
            pca_model
        )

    # Add labels to the dataset, which will open a figure and requires
    # user input in a series of popup windows to determine the string
    # labels for clusters.
    data = sacc.add_labels_to_dataset(data, labels, savepath)

    # Reversing checkerboard responsive
    #for ind, row in data.iterrows():

        #_end = row['Rc_eyeT'][-1].astype(float)
        #_start = row['Rc_eyeT'][0].astype(float)

        #sp = len(row['Rc_spikeT'])
        #data.at[ind, 'Rc_fr'] = sp / (_end - _start)

        #data.at[ind, 'raw_mod_for_Rc'] = sacc.calc_PSTH_modind(
                                                    row['Rc_stim_PSTH'],
                                                    trange='fm')

        #data.at[ind, 'norm_mod_for_Rc'] = sacc.calc_PSTH_modind(row['norm_Rc_psth'])

        # Latency
        #Rc_peakT, Rc_peak_val = sacc.calc_PSTH_latency(row['norm_Rc_psth'])
        #data.at[ind, 'rc_peakT'] = Rc_peakT
        #data.at[ind, 'rc_peak_val'] = Rc_peak_val

    #data['Rc_responsive'] = False
    #for ind, row in data.iterrows():
        #if (row['raw_mod_for_Rc']>1) and (row['norm_mod_for_Rc']>0.1):
            #data.at[ind, 'Rc_responsive'] = True

    # Temporal sequences
    #data.at[ind, 'rc_peakT'] = Rc_peakT

    #use_cols = ['FmLt_gazeshift_peakT','gazecluster','pref_gazeshift_psth',
                #'nonpref_gazeshift_psth','Rc_responsive',
                #'pref_comp_psth','nonpref_comp_psth',
                #'norm_Rc_psth','gazeshift_responsive']
        
    sorted_df = data[use_cols].copy()

    sorted_df = data[use_cols].copy()
    sorted_df.sort_values(by='FmLt_gazeshift_peakT', inplace=True)
    sorted_df = sorted_df[sorted_df['gazecluster']!='unresponsive'][sorted_df['gazeshift_responsive']==True].reset_index()
    tseq_pref = fme.flatten_series(sorted_df['pref_gazeshift_psth'].copy())
    tseq_nonpref = fme.flatten_series(sorted_df['nonpref_gazeshift_psth'].copy())
    tseq_comp = fme.flatten_series(sorted_df['pref_comp_psth'].copy())
    #tseq_rc = fme.flatten_series(sorted_df['norm_Rc_psth'][sorted_df['Rc_responsive']].copy())
    
    unsort_df = data[use_cols].copy()
    gsrespinds_unsort = unsort_df[unsort_df['gazecluster']!='unresponsive'][unsort_df['gazeshift_responsive']==True].copy().index.values
    #rcrespinds_unsort = unsort_df[unsort_df['Rc_responsive']].copy().index.values
    #rcseq_unsort = fme.flatten_series(unsort_df['norm_Rc_psth'].copy())
    #gsseq_unsort = fme.flatten_series(unsort_df['pref_gazeshift_psth'].copy())
    #gsnonprefseq_unsort = fme.flatten_series(unsort_df['nonpref_gazeshift_psth'].copy())
    compseq_unsort = fme.flatten_series(unsort_df['pref_comp_psth'].copy())
    latency_unsort = unsort_df['FmLt_gazeshift_peakT'].copy().to_numpy()
    gazecluster_unsort = unsort_df['gazecluster'].copy().to_numpy()

    props = sacc.propsdict()
    colors = props['colors']

    tseq_legend_col = sorted_df['gazecluster'].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = colors[n]

    out = {
        #'Rc_temseq': tseq_rc,
        'Fm_pref_temseq': tseq_pref,
        'Fm_nonpref_temseq': tseq_nonpref,
        'Fm_comp_temseq': tseq_comp,
        'temseq_legend': tseq_legend,
        'Fm_pref_useinds_unsort': gsrespinds_unsort,
        #'Rc_useinds_unsort': rcrespinds_unsort,
        #'Rc_temseq_unsort': rcseq_unsort,
        'Fm_pref_temseq_unsort': gsseq_unsort,
        'Fm_nonpref_temseq_unsort': gsnonprefseq_unsort,
        'Fm_comp_temseq_unsort': compseq_unsort,
        'Fm_latency_unsort': latency_unsort,
        'Fm_gazecluster_unsort': gazecluster_unsort,
    }

    return data, out


