import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import sklearn.cluster
import sklearn.decomposition
import pickle
import matplotlib.gridspec as gridspec

import fmEphys

def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def calc_PSTH_latency(psth):
    """
    use normalized PSTH
    """
    psth_bins = np.arange(-1,1.001,1/1000)

    ind = np.argmax(psth[1025:1250]) + 1025

    peakT = psth_bins[ind]
    
    val = psth[ind]
    
    return peakT, val

def calc_PSTH_modind(psth, baseval='range'):
    # modulation in terms of spike rate
    psth = psth.astype(float)
    if baseval=='range':
        use = psth - np.mean(psth[0:800].copy())
    elif baseval=='zero':
        use = psth - psth[1000]
    elif baseval=='tight':
        use = psth - np.mean(psth[750:975].copy())
    mod = np.max(np.abs(use[1000:1250]))
    return mod

def calc_PSTH_DS(left, right):
    # use raw PSTH
    
    leftmod = calc_PSTH_modind(left)
    rightmod = calc_PSTH_modind(right)

    ind = np.argmax([leftmod, rightmod])
    
    pref = [left, right][ind]
    nonpref = [left, right][1-ind]
    
    prefname = ['left','right'][ind]
    nonprefname = ['left','right'][1-ind]
    
    return pref, nonpref, prefname, nonprefname


def calc_PSTH_DSI(pref, nonpref):
    # use pref
    
    prefmod = calc_PSTH_modind(pref)
    nonprefmod = calc_PSTH_modind(nonpref)
    
    mod = (prefmod - nonprefmod) / (prefmod + nonprefmod)
    
    return mod
    
def norm_PSTH(psth, raw_pref=None, baseline_val='zero'):
    
    if raw_pref is None:
        raw_pref = psth.copy()

    if baseline_val is None:
        baseline_val = np.mean(psth[0:800].astype(float))
        
    if baseline_val == 'zero':
        baseline_val = psth[1000]
        
    if baseline_val == 'tight':
        baseline_val = np.mean(psth[750:975].astype(float))
        
    
    norm_psth = (psth - baseline_val) / np.max(raw_pref[750:1250].astype(float))
    
    return norm_psth

def drop_nan_along(x, axis=1):
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x

def main():
    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    colors = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey',
        'gaze': 'firebrick',
        'comp': 'mediumblue',
        'rc': 'indigo'
    }
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams.update({'font.size':10})

    flhf = fmEphys.read_group_h5('/home/niell_lab/Data/freely_moving_ephys/batch_files/013023/flhf_013123_v4.h5')

    # Normalize
    psth_keys = {
        'Rc_stim_PSTH': 'Rc_norm_psth',
        'Rc_stim_PSTH': 'Rc_norm_psth',
        'SnI_recalc_stim_psth': 'SnI_norm_psth',
        'Sn_recalc_stim_psth': 'Sn_norm_psth',
        'SnR_stim_PSTH_onSub_bckgndRF': 'SnR_norm_psth'
    }
    check_has_keys = {
        'Rc_norm_psth': 'hasRc',
        'SnI_norm_psth': 'hasSnI',
        'Sn_norm_psth': 'hasSn',
        'SnR_norm_psth': 'hasSnR'
    }

    for x in [x for x in flhf.columns.values if '_eyeT' in x]:
        stim = (x.split('_'))[0]
        print(stim)
        flhf['has'+stim] = False
        for ind, row in flhf.iterrows():
            if not np.isnan(row[x]).all():
                flhf.at[ind,'has'+stim] = True

    # normalize PSTHs
    tmp_arr = np.zeros(2001)*np.nan
    tmp_series = pd.Series([])
    for i in range(len(flhf.index.values)):
        tmp_series.at[i] = tmp_arr.astype(object)
    for x in psth_keys.values():
        flhf[x] = tmp_series.copy()

    for ind, row in tqdm(flhf.iterrows()):
        for key, newcol in psth_keys.items():
            if type(row[key]) != float:
                flhf.at[ind, newcol] = norm_PSTH(row[key])
    
    # Sn responsive
    for ind, row in flhf[flhf['hasSn']].iterrows():
        sec = row['Sn_eyeT'][-1].astype(float) - row['Sn_eyeT'][0].astype(float)
        sp = len(row['Sn_spikeT'])
        flhf.at[ind, 'Sn_fr'] = sp/sec

        flhf.at[ind, 'raw_mod_for_Sn'] = calc_PSTH_modind(row['Sn_recalc_stim_psth'], baseval='tight')

        flhf.at[ind, 'norm_mod_for_Sn'] = calc_PSTH_modind(row['Sn_norm_psth'], baseval='tight')
    
    flhf['Sn_responsive'] = False
    for ind, row in flhf[flhf['hasSn']].iterrows():
        if (row['raw_mod_for_Sn']>1) and (row['norm_mod_for_Sn']>0.1):
            flhf.at[ind, 'Sn_responsive'] = True
    print(flhf['Sn_responsive'].sum(), '/', len(flhf.index))

    # SnI responsive
    for ind, row in flhf[flhf['hasSnI']].iterrows():
        sec = row['SnI_eyeT'][-1].astype(float) - row['SnI_eyeT'][0].astype(float)
        sp = len(row['SnI_spikeT'])
        flhf.at[ind, 'Sn_fr'] = sp/sec

        flhf.at[ind, 'raw_mod_for_SnI'] = calc_PSTH_modind(row['SnI_recalc_stim_psth'], baseval='tight')

        flhf.at[ind, 'norm_mod_for_SnI'] = calc_PSTH_modind(row['SnI_norm_psth'], baseval='tight')
        
    flhf['SnI_responsive'] = False
    for ind, row in flhf[flhf['hasSn']].iterrows():
        if (row['raw_mod_for_SnI']>1) and (row['norm_mod_for_SnI']>0.1):
            flhf.at[ind, 'SnI_responsive'] = True
    print(flhf['SnI_responsive'].sum(), '/', len(flhf.index))

    pca_input = np.zeros([len(flhf[flhf['hasSn']].index), 250])*np.nan

    # cluster Sn

    for i, ind in enumerate(flhf[flhf['hasSn']].index.values):
        
        if flhf.loc[ind, 'Sn_responsive'] == True:
            pca_input[i,:] = flhf.loc[ind, 'Sn_norm_psth'][950:1200]
        
        elif flhf.loc[ind, 'Sn_responsive'] == False:
            pca_input[i,:] = np.zeros(250)

    n_pcas = 10
    req_explvar=0.95

    pca = sklearn.decomposition.PCA(n_components=n_pcas)
    pca.fit(pca_input)

    explvar = pca.explained_variance_ratio_

    proj = pca.transform(pca_input)

    keep_pcas = int(np.argwhere(np.cumsum(explvar)>req_explvar)[0])
    print('using best {} PCs'.format(keep_pcas))

    gproj = proj[:,:keep_pcas]

    km = sklearn.cluster.KMeans(n_clusters=5)
    km.fit_predict(gproj)
    Z = km.labels_

    