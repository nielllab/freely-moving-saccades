
import os
import pandas as pd
import xarray as xr
import numpy as np

import sklearn.cluster
import sklearn.decomposition
import pickle
import sklearn.neighbors
import sklearn.linear_model


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
    

def norm_PSTH(psth, raw_pref=None, baseline_val=None):
    
    if raw_pref is None:
        raw_pref = psth.copy()

    if baseline_val is None:
        baseline_val = np.mean(psth[0:800].astype(float))
        
    if baseline_val == 'zero':
        baseline_val = psth[1000]
        
    if baseline_val == 'tight':
        baseline_val = np.mean(psth[750:975].astype(float))
        
    if baseline_val == 'tight1':
        baseline_val = np.mean(psth[850:1000].astype(float))
    
    norm_psth = (psth - baseline_val) / np.max(raw_pref[750:1250].astype(float))
    
    return norm_psth


def calc_kde_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    # Unit conversions
    bandwidth = bandwidth / 1000
    resample_size = resample_size / 1000
    win = win / 1000
    edgedrop = edgedrop / 1000
    edgedrop_ind = int(edgedrop / resample_size)

    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # Timestamps of spikes (`sps`) relative to `eventT`
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        # Only keep spikes in this window
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] 
        sps.extend(sp)
    sps = np.array(sps)

    if len(sps) < 10:
        n_bins = int((win * 1000 * 2) + 1)
        return np.zeros(n_bins)*np.nan

    kernel = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:, np.newaxis])
    density = kernel.score_samples(bins[:, np.newaxis])

    # Multiply by the # spikes to get spike count per point. Divide
    # by # events for rate/event.
    psth = np.exp(density) * (np.size(sps ) / np.size(eventT))

    # Drop padding at start & end to eliminate edge effects.
    psth = psth[edgedrop_ind:-edgedrop_ind]

    return psth

