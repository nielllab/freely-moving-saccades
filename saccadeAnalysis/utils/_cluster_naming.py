
def label_movcluster(psth, el_bound=0.08):
    """
    PSTH should be the neural response to eye movements
    between -0.0625 and 0.3125 sec, where 0 is the moment
    of the eye movement.
    """

    # find peaks and troughs in PSTH
    p, peak_props = find_peaks(psth, height=.30)
    t, trough_props = find_peaks(-psth, height=.20)

    # get the time index of the highest peaks
    if len(p) > 1:
        p = p[np.argmax(peak_props['peak_heights'])]
    if len(t) > 1:
        t = t[np.argmax(trough_props['peak_heights'])]
    if p.size == 0:
        p = np.nan
    if t.size == 0:
        t = np.nan
    if ~np.isnan(p):
        p = int(p)
    if ~np.isnan(t):
        t = int(t)

    # some filtering to choose the best position for the peak
    if ~np.isnan(p):
        has_peak = True
        peak_cent = p
    else:
        has_peak = False
        peak_cent = None
    if ~np.isnan(t):
        has_trough = True
        trough_cent = t
    else:
        has_trough = False
        trough_cent = None

    # now we decide which cluster each of these should be
    el_bound_ind = np.argmin(np.abs(psth_bins-el_bound))
    if has_peak and has_trough:
        return 'biphasic'
    elif has_trough and ~has_peak:
        return 'negative'
    elif peak_cent is not None and peak_cent <= el_bound_ind:
        return 'early'
    elif peak_cent is not None and peak_cent > el_bound_ind:
        return 'late'
    else:
        return 'unresponsive'
    

