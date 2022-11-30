import numpy as np

import fmEphys

psth_bins = np.arange(-1,1.001,1/1000)

def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def calc_PSTH_modind(psth):
    # modulation in terms of spike rate
    psth = psth.astype(float)
    use = psth - np.mean(psth[0:800].copy())
    mod = np.max(np.abs(use[1000:1250]))
    return mod

def calc_latency(psth):
    # use norm PSTH
    ind = np.argmax(psth[1025:1250])+1025 # was 1000:1170
    peakT = psth_bins[ind]
    val = psth[ind]
    return peakT, val

def calc_direction_pref(left, right):
    # use raw PSTH
    
    leftmod = psth_modind(left)
    rightmod = psth_modind(right)

    ind = np.argmax([leftmod, rightmod])
    
    pref = [left, right][ind]
    nonpref = [left, right][1-ind]
    
    prefname = ['left','right'][ind]
    nonprefname = ['left','right'][1-ind]
    
    return pref, nonpref, prefname, nonprefname

def calc_PSTH_DSI(pref, nonpref):
    # use pref
    
    prefmod = psth_modind(pref)
    nonprefmod = psth_modind(nonpref)
    
    mod = (prefmod - nonprefmod) / (prefmod + nonprefmod)
    
    return mod
    
def normalize_psth(psth, raw_pref=None, baseline_val=None):
    if raw_pref is None:
        raw_pref = psth.copy()
    if baseline_val is None:
        baseline_val = np.mean(psth[0:800].astype(float))
    norm_psth = (psth - baseline_val) / np.max(raw_pref[750:1250].astype(float)) # [1000:1250]
    return norm_psth

def normalize_gt_psth(psth, baseind=4, zeroind = 5):
    baseline_val = np.nanmedian(psth[:5])
    norm_psth = (psth - baseline_val) / np.nanmax(psth[5:14].astype(float))
    return norm_psth

def gt_modind(psth):
    psth = psth.astype(float)
    use = psth - np.mean(psth[1:5].copy())
    mod = np.max(np.abs(use[5:8]))
    return mod