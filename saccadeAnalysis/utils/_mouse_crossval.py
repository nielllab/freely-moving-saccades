
def drop_nan_along(x, axis=1):
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x

def plot_tempseq(panel, tseq, return_img=False, freev=None):
    tseq = drop_nan_along(tseq, 1)
    panel.set_xlabel('msec')
    panel.set_ylim([np.size(tseq,0),0])
    vmin = -0.75; vmax = 0.75
    if freev is not None:
        vmin = -freev
        vmax = freev
    img = panel.imshow(tseq, cmap='coolwarm', vmin=vmin, vmax=vmax)
    panel.set_xlim([800,1400])
    panel.set_xticks(np.linspace(800,1400,4), labels=np.linspace(-200,400,4).astype(int))
    panel.vlines(1000, 0, np.size(tseq,0), color='k', linestyle='dashed', linewidth=1)
    panel.set_aspect(2.8)
    if return_img:
        return img

def calc_kde_sdf(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """
    bandwidth (in msec)
    resample_size (msec)
    edgedrop (msec to drop at the start and end of the window so eliminate artifacts of filtering)
    win = 1000msec before and after
    """
    # some conversions
    bandwidth = bandwidth/1000 # msec to sec
    resample_size = resample_size/1000 # msec to sec
    win = win/1000 # msec to sec
    edgedrop = edgedrop/1000
    edgedrop_ind = int(edgedrop/resample_size)

    # setup time bins
    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # get timestamp of spikes relative to events in eventT
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] # only keep spikes in this window
        sps.extend(sp)
    sps = np.array(sps) # all values in here are between -1 and 1

    # kernel density estimation
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:,np.newaxis])
    density = kernel.score_samples(bins[:,np.newaxis])
    sdf = np.exp(density)*(np.size(sps)/np.size(eventT)) # convert back to spike rate
    sdf = sdf[edgedrop_ind:-edgedrop_ind]

    return sdf

def calc_latency(psth):
    # use norm PSTH
    ind = np.argmax(psth[1025:1250])+1025 # was 1000:1170
    peakT = psth_bins[ind]
    val = psth[ind]
    return peakT, val

def apply_win_to_comp_sacc(comp, gazeshift, win=0.25):
    bad_comp = np.array([c for c in comp for g in gazeshift if ((g>(c-win)) & (g<(c+win)))])
    comp_times = np.delete(comp, np.isin(comp, bad_comp))

    return comp_times

def keep_first_saccade(eventT, win=0.020):
    duplicates = set([])
    for t in eventT:
        new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))

    return out

def make_datasets():

    df = pd.read_pickle('/home/niell_lab/Data/freely_moving_ephys/batch_files/061522/hffm_061522.pickle')

    train_psth = np.zeros([len(df.index.values), 2001])
    test_psth = np.zeros([len(df.index.values), 2001])
    print('num cells = {}'.format(len(df.index.values)))
    for i, ind in tqdm(enumerate(df.index.values)):
        if df.loc[ind, 'pref_gazeshift_direction']=='left':
            fullT = df.loc[ind, 'FmLt_gazeshift_left_saccTimes_dHead1'].copy().astype(float)
        elif df.loc[ind, 'pref_gazeshift_direction']=='right':
            fullT = df.loc[ind, 'FmLt_gazeshift_right_saccTimes_dHead1'].copy().astype(float)
        else:
            print(df.loc[ind, 'pref_gazeshift_direction'])
        
        train_inds = np.random.choice(np.arange(0, fullT.size), size=int(np.floor(fullT.size/2)), replace=False)
        test_inds = np.arange(0, fullT.size)
        test_inds = np.delete(test_inds, train_inds)

        train = fullT[train_inds].copy()
        test = fullT[test_inds].copy()
        
        spikeT = df.loc[ind,'FmLt_spikeT']
        
        train_psth[i,:] = calc_kde_sdf(spikeT, train)
        test_psth[i,:] = calc_kde_sdf(spikeT, test)

    np.save('/home/niell_lab/Desktop/train_psth1.npy', train_psth)
    np.save('/home/niell_lab/Desktop/test_psth1.npy', test_psth)

def get_direction_pref(left, right):
    evok_left = left[1000:1250]
    evok_right = right[1000:1250]
    
    ind = np.argmax([np.max(np.abs(evok_left)), np.max(np.abs(evok_right))])
    pref = ['left','right'][ind]
    nonpref = ('left' if pref=='right' else 'right')
    
    return pref, nonpref
    
def normalize_psth(psth, raw_pref=None, baseline_val=None):
    if raw_pref is None:
        raw_pref = psth.copy()
    if baseline_val is None:
        baseline_val = np.nanmean(psth[0:800].astype(float))
    norm_psth = (psth - baseline_val) / np.nanmax(raw_pref[750:1250].astype(float))
    return norm_psth

def validate():

    train_psth = np.load('/home/niell_lab/Data/freely_moving_ephys/eye_movements/cross validation/June22/train_psth1.npy')
    test_psth = np.load('/home/niell_lab/Data/freely_moving_ephys/eye_movements/cross validation/June22/test_psth1.npy')
    good_inds = np.load('/home/niell_lab/Data/freely_moving_ephys/eye_movements/cross validation/June22/good_inds.npy')

    norm_train = np.zeros([len(good_inds),2001])
    norm_test = np.zeros([len(good_inds),2001])
    for i, ind in enumerate(good_inds):
        norm_train[i,:] = normalize_psth(train_psth[ind,:])
        norm_test[i,:] = normalize_psth(test_psth[ind,:])

    psth_bins = np.arange(-1,1.001,1/1000)

    train_peakT = np.zeros(np.size(norm_train,0))
    test_peakT = np.zeros(np.size(norm_test,0))
    for i in range(np.size(norm_train,0)):
        train_peakT[i], _ = calc_latency(norm_train[i,:])
        test_peakT[i], _ = calc_latency(norm_test[i,:])

    # sort peak times
    order = np.argsort(train_peakT)

    sort_train_psths = norm_train[order,:].copy()
    sort_test_psths = norm_test[order,:].copy()

def figures():
    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(4,4), dpi=300)

    ax0_img = plot_tempseq(ax0, sort_train_psths)

    ax1_img = plot_tempseq(ax1, sort_test_psths)
    ax1.set_yticklabels([])

    ax0.set_title('train')
    ax1.set_title('test')

    # fig.savefig('/home/niell_lab/Desktop/crossval.pdf')

    plt.figure(figsize=(2,2), dpi=300)
    plt.plot(train_peakT[(train_peakT>.025) * (train_peakT<.250)],
            test_peakT[(test_peakT>.025) * (test_peakT<.250)], 'k.', markersize=3)
    plt.xlabel('train latency (msec)'); plt.ylabel('test latency (msecs)')
    plt.plot([0.02,.250], [0.02,.250], linestyle='dashed', color='tab:red', linewidth=1)
    plt.xlim([.02, .20]); plt.ylim([.02, .250])
    plt.xticks(np.linspace(0.020,.250,4), labels=np.linspace(20,250,4).astype(int))
    plt.yticks(np.linspace(0.020,.250,4), labels=np.linspace(20,250,4).astype(int))

    maxcc = np.zeros([len(sort_train_psths)])*np.nan
    for i in range(len(sort_train_psths)):
        
        train = sort_train_psths[i,:].copy()
        test = sort_test_psths[i,:].copy()
        
        r = np.corrcoef(train[1000:1250], test[1000:1250])
        maxcc[i] = r[0,1]**2

    # cross valudation correlation histogram
    fig, ax0 = plt.subplots(1,1,figsize=(2.5,1.5), dpi=300)

    weights = np.ones_like(maxcc) / float(len(maxcc))
    n,_,_ = ax0.hist(maxcc, color='grey', bins=np.linspace(-1,1,21), weights=weights)
    # ax0.set_xlabel('gaze shift cc');
    ax0.set_ylabel('frac. cells')
    ax0.set_xticks(np.arange(-1,1,3),labels=[])
    ax0.plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    ax0.set_ylim([0,.22])

    fig.tight_layout()
    fig.savefig('/home/niell_lab/Desktop/mouse_crossval_gazeshift_correlation.pdf')