


def gratings_responsive(data):

    for ind, row in data.iterrows():
        sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        sp = len(row['Gt_spikeT'])
        data.at[ind, 'Gt_fr'] = sp/sec
        
        data.at[ind, 'raw_mod_for_Gt'] = gt_modind(row['Gt_grating_psth'])
        
        data.at[ind, 'norm_mod_for_Gt'] = gt_modind(row['norm_gratings_psth'])
    
    data['Gt_responsive'] = False
    for ind, row in data.iterrows():
        if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):
            data.at[ind, 'Gt_responsive'] = True





def gratings_tuning(data):

    cluster_to_cell_type = {}
    
    for l in range(5):
        med_response = np.median(flatten_series(data['pref_gazeshift_psth'][data['gazecluster_ind']==l]), axis=0)
        cluster_to_cell_type[l] = label_movcluster(med_response)
    data['gazecluster'] = 'unresponsive'
    for ind, row in data.iterrows():
        if row['gazeshift_responsive']:
            data.at[ind, 'gazecluster'] = cluster_to_cell_type[row['gazecluster_ind']]
    data['gazecluster'].value_counts()
    vc = data['gazecluster'].value_counts()
    vc/np.sum(vc)
    # Only for HFFM
    ## Gratings
    for sf in ['low','mid','high']:
        data['norm_ori_tuning_'+sf] = data['Gt_ori_tuning_tf'].copy().astype(object)
    for ind, row in data.iterrows():
        orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
        for sfnum in range(3):
            sf = ['low','mid','high'][sfnum]
            data.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
        mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
        mean_for_sf[mean_for_sf<0] = 0
        data.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)

    data['osi_for_sf_pref'] = np.nan
    data['dsi_for_sf_pref'] = np.nan
    for ind, row in data.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            data.at[ind, 'osi_for_sf_pref'] = row[(['Gt_osi_low','Gt_osi_mid','Gt_osi_high'][best_sf_pref-1])]
            data.at[ind, 'dsi_for_sf_pref'] = row[(['Gt_dsi_low','Gt_dsi_mid','Gt_dsi_high'][best_sf_pref-1])]

    data['osi_for_sf_pref'][data['osi_for_sf_pref']<0] = 0
    data['dsi_for_sf_pref'][data['dsi_for_sf_pref']<0] = 0
    for ind, row in data.iterrows():
        try:
            mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
            mean_for_sf[mean_for_sf<0] = 0
            data.at[ind, 'Gt_evoked_rate'] = np.max(mean_for_sf) - row['Gt_drift_spont']
        except:
            pass

    for ind, row in data.iterrows():
        if type(row['Gt_ori_tuning_tf']) != float:
            tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
            tuning = tuning - row['Gt_drift_spont']
            tuning[tuning < 0] = 0
            mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
            data.at[ind, 'tf_pref'] = tf_pref

    for ind, row in data.iterrows():
        tf = 2 + (6 * (row['tf_pref']-1))
        sf = 0.02 * 4 ** (row['sf_pref']-1)
        data.at[ind,'tf_pref_cps'] = tf
        data.at[ind,'sf_pref_cpd'] = sf
        data.at[ind,'grat_speed_dps'] = tf / sf