""" make_marmo_dataset.py

"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

import fmEphys as fme
import saccadeAnalysis as sacc


def make_marm_dataset(base_path=None):

    if base_path is None:
        base_path = '/home/dmartins/FastData/gazeshift_dataset/marmoset/'

    psth_bins = np.arange(-200,401)

    # Load preprocessed data
    data = loadmat(os.path.join(base_path,'Pooled_V1Hart_Preload_Final.mat'))
    totdata = loadmat(os.path.join(base_path,'Pooled_V1Hart_Preload_TotalInfo.mat'))['TotalInfo']
    # Recalculated saccades
    raw_sacc = np.load(os.path.join(base_path,'marmoset_recalc_saccades.npy'))
    
    # Load gaze clusters
    clusters = np.load(os.path.join(base_path,'marmoset_clusters.npy'))
    k_to_name = {
        0:'early',
        2:'late',
        3:'biphasic',
        1:'negative'
    }

    # Open arrays
    sacc_psth = data['ISACMOD2']
    grat_psth = data['GSACMOD']
    sf_tuning = data['SFTUNE']
    tf_tuning = data['TFTUNE']
    ori_tuning = data['ORTUNE']
    bsln_fr = data['BASEMU2']
    peakT = data['PEAKIM2']
    animal = data['ANIMID']

    n_cells = np.size(peakT,0)

    # Normalize the PSTHs
    norm_sacc_psth = np.zeros([n_cells, len(psth_bins)])
    for ind in range(n_cells):
        norm_sacc_psth[ind,:] = sacc.marm_normalize_psth(sacc_psth[ind].copy())


    # Spatial and temporal frequencies used
    sf_vals = np.array([1,2,4,8,16])
    tf_vals = (60/16) * np.arange(9)

    # Empty arrays for tuning values
    tf_pref = np.zeros(n_cells)
    sf_pref = np.zeros(n_cells)
    ori_index = np.zeros(n_cells)
    tf_index = np.zeros(n_cells)
    sf_index = np.zeros(n_cells)

    for ind in range(n_cells):

        sf = sf_tuning[ind,:].copy()
        tf = tf_tuning[ind,:].copy()
        ori = ori_tuning[ind,:].copy()
        
        ofi = np.nanstd(ori) / np.nanmean(ori)
        sfi = np.nanstd(sf) / np.nanmean(sf)
        tfi = np.nanstd(tf) / np.nanmean(tf)
        
        svec = sf.copy()-1
        svec[svec<0] = 0
        svec = svec**2
        spref = np.nansum(svec * sf_vals) / np.nansum(svec)
        
        tvec = tf.copy()-1
        tvec[tvec<0] = 0
        tvec = tvec**2
        tpref = np.nansum(tvec * tf_vals) / np.nansum(tvec)
        
        sf_pref[ind] = spref
        tf_pref[ind] = tpref
        ori_index[ind] = ofi
        sf_index[ind] = sfi
        tf_index[ind] = tfi

    # Saccade responsive
    sacc_resp = np.zeros(n_cells)
    mods = np.zeros(n_cells)
    for ind in range(n_cells):
        mod = sacc.marm_psth_modind(norm_sacc_psth[ind,:])
        mods[ind] = mod
        if mod > 0.1:
            sacc_resp[ind] = True

    # Gratings responsive
    grat_resp = np.zeros(n_cells)
    for ind in range(n_cells):
        ffi = np.sqrt(ori_index[ind]**2 + sf_index[ind]**2)
        if ffi >= 0.2:
            grat_resp[ind] = True
    grat_resp = grat_resp.astype(bool)

    # Assemble dataframe to save out
    df = pd.DataFrame(sf_pref, columns=['sf_pref'])
    df['gazecluster_ind'] = clusters
    df['animal'] = animal.T[0]

    _tmpSer = fme.blank_col(length=n_cells, size=601)
    df['PSTH'] = _tmpSer.copy()
    df['normPSTH'] = _tmpSer.copy()
    df['sf_tuning'] = fme.blank_col(length=n_cells, size=5)
    df['tf_tuning'] = fme.blank_col(length=n_cells, size=9)
    df['ori_tuning'] = fme.blank_col(length=n_cells, size=16)
    df['spikeT'] = fme.blank_col(length=n_cells, size=1)
    df['session'] = None

    for i in df.index.values:
        df.at[i,'gazecluster'] = k_to_name[df.loc[i,'gazecluster_ind']]
        df.at[i,'PSTH'] = sacc_psth[i]
        df.at[i,'normPSTH'] = norm_sacc_psth[i]

        df.at[i,'gratings_responsive'] = grat_resp[i]
        df.at[i,'peakT'] = peakT[i]
        df.at[i,'baseline_fr'] = bsln_fr[i]
        df.at[i,'saccade_responsive'] = sacc_resp[i]
        df.at[i,'tf_pref'] = tf_pref[i]
        df.at[i,'sf_tuning'] = sf_tuning[i]
        df.at[i,'tf_tuning'] = tf_tuning[i]
        df.at[i,'ori_tuning'] = ori_tuning[i]
        
        unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                    'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
        unit_dict = dict(zip(unitlabels, list(totdata[i][0][0][0])))

        sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                    'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                    'StimSU2','BaseMu','BaseMu2']
        sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

        rast = sacim_dict['StimRast2'].copy()

        trials = np.unique(rast[:,1]).astype(int)-1

        all_sps = []
        for tnum in trials:
            sps = rast.copy()
            sps = sps[sps[:,1].astype(int)==int(tnum)]
            sps = sps[:,0]
            all_sps.extend(sps)

        df.at[i, 'spikeT'] = all_sps
        
    savepath = os.path.join(base_path,'marm.h5')
    df.to_hdf(savepath, key='marm')
