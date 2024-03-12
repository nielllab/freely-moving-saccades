

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import fmEphys as fme
import saccadeAnalysis as sacc


def make_hffm_dataset(savepath, session_dict=None, hffm_path=None,
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
    if km_model is None or pca_model is None:
        labels, _km_opts = sacc.make_clusters(pca_input, savepath)

    elif km_model is not None and pca_model is not None:
        # apply existing clustering
        labels, _km_opts = sacc.apply_saved_cluster_models(
            pca_input,
            km_model,
            pca_model
        )

    # proj = _km_opts['proj']
    # plt.scatter(proj[:,0], proj[:,1], c=labels)

    # Add labels to the dataset, which will open a figure and requires
    # user input in a series of popup windows to determine the string
    # labels for clusters.
    data = sacc.add_labels_to_dataset(data, labels, savepath)

    # Determine gratings-responsive cells
    #for ind, row in data.iterrows():

        #sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        #sp = len(row['Gt_spikeT'])
        #data.at[ind, 'Gt_fr'] = sp/sec
        
        #data.at[ind, 'raw_mod_for_Gt'] = sacc.calc_PSTH_modind(
                                                #row['Gt_stim_PSTH'],
                                                #trange='gt')
        
        #data.at[ind, 'norm_mod_for_Gt'] = sacc.calc_PSTH_modind(
                                                #row['norm_gratings_psth'],
                                                #trange='gt')

    #data['Gt_responsive'] = False
    #for ind, row in data.iterrows():
        #if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):

            #data.at[ind, 'Gt_responsive'] = True


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

    # Sparse noise responsive
    #for ind, row in data.iterrows():
        #sec = row['Sn_eyeT'][-1].astype(float) - row['Sn_eyeT'][0].astype(float)
        #sp = len(row['Sn_spikeT'])
        #data.at[ind, 'Sn_fr'] = sp/sec
        
        #data.at[ind, 'raw_mod_for_Sn'] = sacc.calc_PSTH_modind(
                                                row['Sn_stim_PSTH_onSub_bckgndRF'],
                                                trange='sn')

        #data.at[ind, 'norm_mod_for_Sn'] = sacc.calc_PSTH_modind(
                                                row['norm_Sn_psth'],
                                                trange='sn')
        
    #data['Sn_responsive'] = False
    #for ind, row in data.iterrows():
        #if (row['raw_mod_for_Sn']>1) and (row['norm_mod_for_Sn']>0.1):
            #data.at[ind, 'Sn_responsive'] = True

    # Sparse noise latency
    #for ind, row in data.iterrows():
        #Sn_peakT, Sn_peak_val = sacc.calc_PSTH_latency(row['norm_Sn_psth'])
        #data.at[ind, 'sn_peakT'] = Sn_peakT
        #data.at[ind, 'sn_peak_val'] = Sn_peak_val

    # Temporal sequences
    #data.at[ind, 'rc_peakT'] = Rc_peakT
    #data.at[ind, 'sn_peakT'] = Sn_peakT

    
    # Only for HFFM
    ## Gratings
    #for sf in ['low','mid','high']:
        #data['norm_ori_tuning_'+sf] = data['Gt_ori_tuning_tf'].copy().astype(object)
    #for ind, row in data.iterrows():
        #orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
        #for sfnum in range(3):
            #sf = ['low','mid','high'][sfnum]
            #data.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
        #mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
        #mean_for_sf[mean_for_sf<0] = 0
        #data.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)


        #osi = np.zeros([3])
        #dsi = np.zeros([3])

        #raw_tuning = np.mean(row['Gt_ori_tuning_tf'],2)
        #drift_spont = row['Gt_drift_spont']
        #tuning = raw_tuning - drift_spont # subtract off spont rate
        #tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
        #th_pref = np.nanargmax(tuning,0) # get position of highest firing rate

        #for sf in range(3):
            
            # get that firing rate (avg between peaks)
            #R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5
            
            # get ortho position
            #th_ortho = (th_pref[sf]+2) % 8
            
            # ortho firing rate (average between two peaks)
            #R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5
            
            # orientaiton selectivity index
            #osi[sf] = (R_pref - R_ortho) / (R_pref + R_ortho)
            
            # direction selectivity index
            # get other direction of same orientation
            #th_null = (th_pref[sf]+4) % 8
            
            # tuning value at that peak
            #R_null = tuning[th_null, sf]
            
            # direction selectivity index
            #dsi[sf] = (R_pref - R_null) / (R_pref + R_null)

        #data.at[ind, 'Gt_osi_low'] = osi[0]
        #data.at[ind, 'Gt_osi_mid'] = osi[1]
        #data.at[ind, 'Gt_osi_high'] = osi[2]

        #data.at[ind, 'Gt_dsi_low'] = dsi[0]
        #data.at[ind, 'Gt_dsi_mid'] = dsi[1]
        #data.at[ind, 'Gt_dsi_high'] = dsi[2]


    #data['osi_for_sf_pref'] = np.nan
    #data['dsi_for_sf_pref'] = np.nan
    #for ind, row in data.iterrows():
        #if ~np.isnan(row['sf_pref']):
            #best_sf_pref = int(np.round(row['sf_pref']))
            #data.at[ind, 'osi_for_sf_pref'] = row[(['Gt_osi_low','Gt_osi_mid','Gt_osi_high'][best_sf_pref-1])]
            #data.at[ind, 'dsi_for_sf_pref'] = row[(['Gt_dsi_low','Gt_dsi_mid','Gt_dsi_high'][best_sf_pref-1])]

    #data['osi_for_sf_pref'][data['osi_for_sf_pref']<0] = 0
    #data['dsi_for_sf_pref'][data['dsi_for_sf_pref']<0] = 0
    #for ind, row in data.iterrows():
        #try:
            #mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
            #mean_for_sf[mean_for_sf<0] = 0
            #data.at[ind, 'Gt_evoked_rate'] = np.max(mean_for_sf) - row['Gt_drift_spont']
        #except:
            #pass

    #for ind, row in data.iterrows():
        #if type(row['Gt_ori_tuning_tf']) != float:
            #tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
            #tuning = tuning - row['Gt_drift_spont']
            #tuning[tuning < 0] = 0
            #mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            #tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
            #data.at[ind, 'tf_pref'] = tf_pref

    #for ind, row in data.iterrows():
        #tf = 2 + (6 * (row['tf_pref']-1))
        #sf = 0.02 * 4 ** (row['sf_pref']-1)
        #data.at[ind,'tf_pref_cps'] = tf
        #data.at[ind,'sf_pref_cpd'] = sf
        #data.at[ind,'grat_speed_dps'] = tf / sf


    use_cols = ['FmLt_gazeshift_peakT','gazecluster','pref_gazeshift_psth',
                'nonpref_gazeshift_psth',#'Rc_responsive','Sn_responsive',
                'pref_comp_psth','Gt_responsive','nonpref_comp_psth',
                #'norm_Rc_psth',#'norm_Sn_psth','tf_pref_cps',
                'sf_pref_cpd','gazeshift_responsive']
        
    sorted_df = data[use_cols].copy()
    # tseq_unresp = sorted_df.copy()
    # tseq_unresp = sorted_df[sorted_df['gazecluster']=='unresponsive'][sorted_df['gazeshift_responsive']==False].sample(frac=1).reset_index(drop=True)
    # tseq_unresp_pref = fme.flatten_series(tseq_unresp['pref_gazeshift_psth'])
    # tseq_unresp_nonpref = fme.flatten_series(tseq_unresp['nonpref_gazeshift_psth'])
    # tseq_unresp_comp = fme.flatten_series(tseq_unresp['pref_comp_psth'])
    # tseq_unresp_rc = fme.flatten_series(tseq_unresp['norm_Rc_psth'][sorted_df['Rc_responsive']])
    # tseq_unresp_sn = fme.flatten_series(tseq_unresp['norm_Sn_psth'][sorted_df['Sn_responsive']])

    sorted_df = data[use_cols].copy()
    sorted_df.sort_values(by='FmLt_gazeshift_peakT', inplace=True)
    sorted_df = sorted_df[sorted_df['gazecluster']!='unresponsive'][sorted_df['gazeshift_responsive']==True].reset_index()
    tseq_pref = fme.flatten_series(sorted_df['pref_gazeshift_psth'].copy())
    tseq_nonpref = fme.flatten_series(sorted_df['nonpref_gazeshift_psth'].copy())
    tseq_comp = fme.flatten_series(sorted_df['pref_comp_psth'].copy())
    tseq_rc = fme.flatten_series(sorted_df['norm_Rc_psth'][sorted_df['Rc_responsive']].copy())
    tseq_sn = fme.flatten_series(sorted_df['norm_Sn_psth'][sorted_df['Sn_responsive']].copy())
    tseq_grat_tf = sorted_df['tf_pref_cps'][sorted_df['Gt_responsive']].copy().to_numpy()
    tseq_grat_sf = sorted_df['sf_pref_cpd'][sorted_df['Gt_responsive']].copy().to_numpy()
    
    unsort_df = data[use_cols].copy()
    #gsrespinds_unsort = unsort_df[unsort_df['gazecluster']!='unresponsive'][unsort_df['gazeshift_responsive']==True].copy().index.values
    #rcrespinds_unsort = unsort_df[unsort_df['Rc_responsive']].copy().index.values
    #snrespinds_unsort = unsort_df[unsort_df['Sn_responsive']].copy().index.values
    #rcseq_unsort = fme.flatten_series(unsort_df['norm_Rc_psth'].copy())
    #snseq_unsort = fme.flatten_series(unsort_df['norm_Sn_psth'].copy())
    gsseq_unsort = fme.flatten_series(unsort_df['pref_gazeshift_psth'].copy())
    gsnonprefseq_unsort = fme.flatten_series(unsort_df['nonpref_gazeshift_psth'].copy())
    compseq_unsort = fme.flatten_series(unsort_df['pref_comp_psth'].copy())
    latency_unsort = unsort_df['FmLt_gazeshift_peakT'].copy().to_numpy()
    gazecluster_unsort = unsort_df['gazecluster'].copy().to_numpy()

    props = sacc.propsdict()
    colors = props['colors']

    tseq_legend_col = sorted_df['gazecluster'].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = colors[n]
    # ucmap = mpl.colors.to_rgba(colors['unresponsive'])
    # # u = np.zeros([np.size(tseq_unresp_pref,0), 1, 4])
    # for x in range(4):
    #     u[:,:,x] = ucmap[x]
    # tseq_legend1 = np.vstack([tseq_legend, u])

    out = {
        #'Rc_temseq': tseq_rc,
        #'Sn_temseq': tseq_sn,
        'Fm_pref_temseq': tseq_pref,
        'Fm_nonpref_temseq': tseq_nonpref,
        'Fm_comp_temseq': tseq_comp,
        'temseq_legend': tseq_legend,
        'Fm_pref_useinds_unsort': gsrespinds_unsort,
        #'Rc_useinds_unsort': rcrespinds_unsort,
        #'Sn_useinds_unsort': snrespinds_unsort,
        #'Rc_temseq_unsort': rcseq_unsort,
        #'Sn_temseq_unsort': snseq_unsort,
        'Fm_pref_temseq_unsort': gsseq_unsort,
        'Fm_nonpref_temseq_unsort': gsnonprefseq_unsort,
        'Fm_comp_temseq_unsort': compseq_unsort,
        'Fm_latency_unsort': latency_unsort,
        'Fm_gazecluster_unsort': gazecluster_unsort,
    }

    return data, out


