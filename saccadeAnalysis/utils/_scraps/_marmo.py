import os
from tqdm import tqdm

from scipy.io import loadmat

import numpy as np
import pandas as pd

import itertools
import scipy.stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt




psth_bins = np.arange(-200,401)

# savepath = '/home/niell_lab/Desktop/'
data = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_Final.mat')
totdata = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_TotalInfo.mat')['TotalInfo']
raw_sacc = np.load('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy')


uNum = 0
unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
            'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
unit_dict = dict(zip(unitlabels, list(totdata[uNum][0][0][0])))

sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
            'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
            'StimSU2','BaseMu','BaseMu2']
sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

raw_sacc = np.load('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy')


sacc_psth = data['ISACMOD2']
grat_psth = data['GSACMOD']
sf_tuning = data['SFTUNE']
tf_tuning = data['TFTUNE']
ori_tuning = data['ORTUNE']
bsln_fr = data['BASEMU2']
peakT = data['PEAKIM2']
animal = data['ANIMID']



n_cells = np.size(peakT,0)

norm_sacc_psth = np.zeros([n_cells, len(psth_bins)])
for ind in range(n_cells):
    norm_sacc_psth[ind,:] = normalize_psth(sacc_psth[ind].copy())


grat_resp = np.zeros(n_cells)
for ind in range(n_cells):
    ffi = np.sqrt(ori_index[ind]**2 + sf_index[ind]**2)
    if ffi >= 0.2:
        grat_resp[ind] = True
grat_resp = grat_resp.astype(bool)