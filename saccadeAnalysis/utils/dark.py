
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import fmEphys as fme
import saccadeAnalysis as sacc



# dark modulation
for ind, row in data.iterrows():
    norm_psth = row['pref_dark_gazeshift_psth'].copy().astype(float)
    data.at[ind, 'norm_dark_modulation'] = psth_modind(norm_psth)
    
    raw_psth = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])].copy().astype(float)
    data.at[ind, 'dark_modulation'] = psth_modind(raw_psth)
cmaps = [cat_cmap['early'],cat_cmap['late'],cat_cmap['biphasic'],cat_cmap['negative']]

plotvals = data[data['gazeshift_responsive']][data['gazecluster_ind']!=4].copy()




light_pref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
light_nonpref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]

dark_pref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
dark_nonpref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]



for ind in plotvals.index.values:
    
    if (plotvals.loc[ind,'dark_modulation']>1) and (plotvals.loc[ind,'norm_dark_modulation']>0.1) and (plotvals.loc[ind,'FmDk_gazeshift_peakT']<=0.035):
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
    model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
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
    model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
    data.at[ind, 'FmDk_model_active'] = model_active.astype(object)
    data.at[ind, 'FmDk_modelT'] = modelT.astype(object)
data['FmLt_fr'] = ((data['FmLt_rate'].apply(np.sum)*0.025) / data['FmLt_eyeT'].apply(np.nanmax)).to_numpy()
data['FmDk_fr'] = ((data['FmDk_rate'].apply(np.sum)*0.025) / data['FmDk_eyeT'].apply(np.nanmax)).to_numpy()
FmLt_fr = np.zeros([len(data.index.values)])
FmDk_fr = np.zeros([len(data.index.values)])
for ind, row in data.iterrows():
    data.at[ind,'FmLt_active_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']>40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']>40])
    data.at[ind,'FmLt_inactive_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']<40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']<40])
    data.at[ind,'FmDk_active_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']>40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']>40])
    data.at[ind,'FmDk_inactive_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']<40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']<40])
## light/dark correlation
for ind, row in data[['pref_dark_gazeshift_psth','pref_gazeshift_psth']].iterrows():
    r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['pref_dark_gazeshift_psth'].astype(float)[1000:1250])
    data.at[ind, 'gaze_ltdk_maxcc'] = r[0,1]
# Head-fixed vs. gaze correlation
for ind, row in data[['norm_Rc_psth','norm_Sn_psth','pref_gazeshift_psth']].iterrows():
    if (np.sum(~np.isnan(row['norm_Rc_psth'].astype(float)[1000]))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
        r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Rc_psth'].astype(float)[1000:1250])
        data.at[ind, 'gaze_rc_maxcc'] = r[0,1]
    if (np.sum(~np.isnan(row['norm_Sn_psth'].astype(float)))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
        r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Sn_psth'].astype(float)[1000:1250])
        data.at[ind, 'gaze_sn_maxcc'] = r[0,1]



# Save the updated h5 file
_saveas = os.path.join(savepath,
                       'LtDk_   .h5'
fme.write_h5(data, _saveas)