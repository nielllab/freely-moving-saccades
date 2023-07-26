
def gt_modind(psth):
    # modulation in terms of spike rate
    psth = psth.astype(float)
    
    use = psth - np.mean(psth[1100:1400].copy())
    
    mod = np.max(np.abs(use[1500:2500]))
    
    return mod


def normalize_gt_psth(psth):
    
    psth = psth.astype(float)
    
    bsln = np.mean(psth[1100:1400])

    norm_psth = (psth - bsln) / np.nanmax(psth)
    
    return norm_psth

def plot_linregress1(ax, x_in, y_in):
    x = x_in[(~np.isnan(x_in)) * (~np.isnan(y_in))]
    y = y_in[(~np.isnan(x_in)) * (~np.isnan(y_in))]
    res = linregress(x, y)
    minval = np.min(x); maxval = np.max(x)
    # border = (maxval - minval) * 0.1
    # plotx = np.linspace(minval+border, maxval-border, 2)
    plotx = np.linspace(0, maxval, 2)
    ax.plot(plotx, (res.slope*plotx) + res.intercept, 'k--', linewidth=1)
    return res

def running_median(panel, x, y, n_bins=7):

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, bin_number = stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    panel.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
               bin_means,
               '-', color='k')
    
    panel.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                       bin_means-tuning_err,
                       bin_means+tuning_err,
                       color='k', alpha=0.2)


def stderr(a, axis=0):
    return np.nanstd(a,axis=axis) / np.sqrt(np.size(a,axis=axis))

def calc_latency(psth):
    # use norm PSTH

    ind = np.argmax(psth[1025:1250])+1025 # was 1000:1170
    peakT = psth_bins[ind]
    val = psth[ind]
    return peakT, val

def psth_modind(psth, baseval='range'):
    # modulation in terms of spike rate
    psth = psth.astype(float)
    if baseval=='range':
        use = psth - np.mean(psth[0:800].copy())
    elif baseval=='zero':
        use = psth - psth[1000]
    mod = np.max(np.abs(use[1000:1250]))
    return mod

def normalize_psth(psth, raw_pref=None, baseline_val=None):
    if raw_pref is None:
        raw_pref = psth.copy()
    if baseline_val is None:
        baseline_val = np.mean(psth[0:800].astype(float))
    norm_psth = (psth - baseline_val) / np.max(raw_pref[750:1250].astype(float)) # [1000:1250]
    return norm_psth


def drop_repeat_events(eventT, do_onset=True, win=0.020):
    duplicates = set([])
    for t in eventT:
        if do_onset==True:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return thinned
