import os, json
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import multiprocessing

def calc_PSTH(sps, n_events):

    bandwidth = 0.010
    end = 0.4
    start = -0.2
    resamp = 0.001
    edge = 0.015
    edge_ind = 15

    bins = np.arange(start-edge, end+edge, resamp)

    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:, np.newaxis])
    density = kernel.score_samples(bins[:, np.newaxis])

    psth = np.exp(density) * (np.size(sps) / n_events)

    psth = psth[edge_ind:-edge_ind]

    return psth

def flatten(l):
    return [item for sublist in l for item in sublist]

def normalize_psth(psth):
    pref = psth.copy()
    bsln = np.mean(psth[0:150]) # was -100 to -50 ms ; now, -200 to -50 ms
    norm_psth = (psth - bsln) / np.max(pref[200:]) # 0 to 200
    return norm_psth

def mpPSTH(sps):

    out = []

    trials = np.unique(sps[:,1]).astype(int)-1

    train_inds = np.array(sorted(np.random.choice(trials, size=int(np.floor(trials.size/2)), replace=False)))
    # test_inds = trials.copy()
    # test_inds = np.delete(test_inds, train_inds)
    # inds = [train_inds, test_inds]
    
    # for tt in range(2): # [train, test]
        
    # use_i = inds[tt]

    use_i = train_inds
    
    use_sp = np.array(flatten([sps[sps[:,1].astype(int)==int(tnum)][:,0] for tnum in use_i]))

    psth = calc_PSTH(use_sp, use_i.size)

    # out.append(psth)
    
    return psth

def main():

    # data = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_Final.mat')
    totdata = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_TotalInfo.mat')['TotalInfo']

    spikeT = {}
    for u in range(len(totdata)):
        spikeT[u] = totdata[u][0][0][0]['SacImage'][0][0]['StimRast2']

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    param_mp = [pool.apply_async(mpPSTH, args=(spikeT[u],)) for u in spikeT.keys()]
    params_output = [result.get() for result in param_mp]

    out = np.zeros([334,601])
    for u in range(334):
        # for x in range(2):
        out[u,:] = params_output[u]

    np.save('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy', out)

if __name__ == '__main__':
    main()