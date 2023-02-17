

import numpy as np



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
    
def norm_PSTH(psth, raw_pref=None, baseline_val=None):
    
    if raw_pref is None:
        raw_pref = psth.copy()

    if baseline_val is None:
        baseline_val = np.mean(psth[0:800].astype(float))
    
    norm_psth = (psth - baseline_val) / np.max(raw_pref[750:1250].astype(float))
    
    return norm_psth

# def normalize_gt_psth(psth, baseind=4, zeroind = 5):
#     baseline_val = np.nanmedian(psth[:5])
#     norm_psth = (psth - baseline_val) / np.nanmax(psth[5:14].astype(float))
#     return norm_psth

# def gt_modind(psth):
#     psth = psth.astype(float)
#     use = psth - np.mean(psth[1:5].copy())
#     mod = np.max(np.abs(use[5:8]))
#     return mod