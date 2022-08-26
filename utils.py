def cell_type(celldict):
    """
    excitatory/inhibitory cell types
    """

    unitdata['waveform']

    data['norm_waveform'] = data['waveform']
    for ind, row in data.iterrows():
        if type(row['waveform']) == list:
            starting_val = np.mean(row['waveform'][:6])
            center_waveform = [i-starting_val for i in row['waveform']]
            norm_waveform = center_waveform / -np.min(center_waveform)
            data.at[ind, 'waveform_trough_width'] = len(norm_waveform[norm_waveform < -0.2])
            data.at[ind, 'AHP'] = norm_waveform[27]
            data.at[ind, 'waveform_peak'] = norm_waveform[18]
            data.at[ind, 'norm_waveform'] = norm_waveform

    km_labels = KMeans(n_clusters=2).fit(list(data['norm_waveform'][data['waveform_peak'] < 0].to_numpy())).labels_
    # make inhibitory is always group 0
    # excitatory should always have a smaller mean waveform trough
    # if it's larger, flip the kmeans labels
    if np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==0]) > np.mean(data['waveform_trough_width'][data['waveform_peak']<0][km_labels==1]):
        km_labels = [0 if i==1 else 1 for i in km_labels]

    data['waveform_km_label'] = np.nan
    count = 0
    for ind, row in data.iterrows():
        if row['waveform_peak'] < 0 and row['AHP'] < 0.7:
            data.at[ind, 'waveform_km_label'] = km_labels[count]
            count = count+1

    data['exc_or_inh'] = np.nan
    # make new column of strings for excitatory vs inhibitory clusters
    for ind, row in data.iterrows():
        if row['waveform_km_label'] == 0:
            data.at[ind, 'exc_or_inh'] = 'inh'
        elif row['waveform_km_label'] == 1:
            data.at[ind, 'exc_or_inh'] = 'exc'

def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def psth_modind(psth):
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

def calc_psth_DSI(pref, nonpref):
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