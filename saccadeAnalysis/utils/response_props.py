"""
fmSaccades/utils/response_props.py

Functions
---------
calc_PSTH_modind
    PSTH modulation index.
norm_PSTH



Written by DMM, 2022
"""


import numpy as np


def calc_PSTH_modind(psth, trange='fm'):
    """ PSTH modulation index.
    
    There are three possible ranges to use for the
    baseline and activity periods:
    For eye/head movements, the baseline value is calculated
    from -1000 ms until -200 ms before the movement onset. The
    active period


    Parameters
    ----------
    psth : array
        PSTH to calculate modulation index for. Should have
        the shape (time,) and be either of the length 2001
        or 3001 (i.e., either -1000 to 1000 (plus the 0
        timepoint) or -1500 to 1500 (plus the 0 timepoint)).
    trange : str
        Time range to use for the baseline and active periods.
        Options are
            'fm'
        For freely moving eye/head movements. Baseline
        is -1000 to -200 ms before movement onset, and active
        is 0 to 1000 ms after movement onset.
            'fl'
        For fast flashed stimuli. Baseline is 0 ms before
        stimulus onset (meaning it is only that single timepoint),
        and active is 0 to 1000 ms after onset.
            'gt'
        For gratings. These PSTHs were calculated from -1500 to
        1500 ms, so the baseline is centered around the stimulus
        onset at the index 1501 instead of at the index 1001. The
        baseline is -400 ms to -100 ms before stimulus onset, and
        the active period is 0 ms until 1500 ms after stimulus.

        'fl' (for fast flashed stimuli which do not have time
        to return to baseline after each stimulus)

    """

    psth = psth.astype(float)

    # FOr freely moving eye/head movements
    if trange=='fm':

        # Use the response window with the baseline subtracted
        # baseline = t[-1000 ms -> -200 ms]
        use = psth - np.mean(psth[0:800].copy())

        # Calculate modulation
        # maximum response of |response window|
        mod = np.max(np.abs(use[1000:1250]))

    # For flashed head-fixed stimuli
    elif trange=='fl':

        # Subtract the reponse at t=0
        bsln = psth[1000]
        use = psth.copy() - bsln
        
        mod = np.max(np.abs(use[1000:1250]))

    # For drifting gratings
    elif trange=='gt':

        bsln = np.mean(psth[1100:1400].copy())
        use = psth.copy() - bsln

        mod = np.max(np.abs(use[1500:2500]))

    return mod


def norm_PSTH(psth, rawpref=None, trange='fm'):

    # When PSTHs are passed in, make sure they are not
    # of dtype==object, which is the case when they are
    # formatted as a column
    psth = psth.astype(float)
    if rawpref is not None:
        rawpref = rawpref.astype(float)

    # For freely moving eye/head movements
    if trange=='fm':

        # Compare to self, if there was not another prefered response
        # to use normalize by.
        if rawpref is None:
            rawpref = psth.copy()
        
        # Baseline value
        bsln = np.mean(psth[0:800])

        # Normalize
        norm_psth = (psth - bsln) / np.nanmax(rawpref[750:1250])

    # For flashed head-fixed stimuli
    elif trange=='gt':

        bsln = np.mean(psth[1100:1400])

        norm_psth = (psth - bsln) / np.nanmax(psth)

    elif trange == 'sn':

        blsn = psth[1000]

        norm_psth = (psth - bsln) / np.nanmax(psth)

    return norm_psth


def calc_PSTH_DSI(pref, nonpref, trange='fm'):

    prefmod = calc_PSTH_modind(pref, trange=trange)
    nonprefmod = calc_PSTH_modind(nonpref, trange=trange)

    dsi = (prefmod - nonprefmod) / (prefmod + nonprefmod)

    return dsi


def calc_PSTH_latency(normpsth):
    """
    Use the normalized PSTH, not the raw spike rate.

    Returns
    -------
    peakT : float
        Peak time of the PSTH (in seconds).
    val : float
        Value of the response peak of the PSTH.
    """

    psth_bins = np.arange(-1,1.001,1/1000)

    # Calculate latency within the window of +25 ms to
    # +250 ms after stimulus onset.
    ind = np.argmax(normpsth[1025:1250])+1025 # was 1000:1170

    peakT = psth_bins[ind]
    val = normpsth[ind]

    return peakT, val


def get_direction_pref(left, right):
    # use raw PSTH
    
    leftmod = psth_modind(left)
    rightmod = psth_modind(right)

    ind = np.argmax([leftmod, rightmod])
    
    pref = [left, right][ind]
    nonpref = [left, right][1-ind]
    
    prefname = ['left','right'][ind]
    nonprefname = ['left','right'][1-ind]
    
    return pref, nonpref, prefname, nonprefname


def norm_grat_histPSTH(psth, baseind=4, zeroind = 5):
    """
    for hist gratings psth
    """
    
    baseline_val = np.nanmedian(psth[:5])
    norm_psth = (psth - baseline_val) / np.nanmax(psth[5:14].astype(float))

    return norm_psth


def calc_grat_histPSTH_modind(psth):
    """
    for hist gratings psth
    """

    psth = psth.astype(float)
    use = psth - np.mean(psth[1:5].copy())
    mod = np.max(np.abs(use[5:8]))

    return mod