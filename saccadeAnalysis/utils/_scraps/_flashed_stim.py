

def flashed_responsive()



    ### revchecker responsive
    for ind, row in hffm.iterrows():
        sec = row['Rc_eyeT'][-1].astype(float) - row['Rc_eyeT'][0].astype(float)
        sp = len(row['Rc_spikeT'])
        hffm.at[ind, 'Rc_fr'] = sp/sec

        hffm.at[ind, 'raw_mod_for_Rc'] = psth_modind(row['Rc_psth'])

        hffm.at[ind, 'norm_mod_for_Rc'] = psth_modind(row['norm_Rc_psth'])
        
    hffm['Rc_responsive'] = False
    for ind, row in hffm.iterrows():
        if (row['raw_mod_for_Rc']>1) and (row['norm_mod_for_Rc']>0.1):
            hffm.at[ind, 'Rc_responsive'] = True
    # print(hffm['Rc_responsive'].sum())

    ### sparse noise responsive
    for ind, row in hffm.iterrows():
        sec = row['Sn_eyeT'][-1].astype(float) - row['Sn_eyeT'][0].astype(float)
        sp = len(row['Sn_spikeT'])
        hffm.at[ind, 'Sn_fr'] = sp/sec
        
        hffm.at[ind, 'raw_mod_for_Sn'] = psth_modind(row['Sn_on_background_psth'], baseval='zero')

        hffm.at[ind, 'norm_mod_for_Sn'] = psth_modind(row['norm_Sn_psth'], baseval='zero')
        
    hffm['Sn_responsive'] = False
    for ind, row in hffm.iterrows():
        if (row['raw_mod_for_Sn']>1) and (row['norm_mod_for_Sn']>0.1):
            hffm.at[ind, 'Sn_responsive'] = True
    # print(hffm['Sn_responsive'].sum())