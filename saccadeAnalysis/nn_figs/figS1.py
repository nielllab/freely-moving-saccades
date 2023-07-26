
### movement counts
Lgaze_count = []; Rgaze_count = []
Lcomp_count = []; Rcomp_count = []
mins = []
for session in hffm['session'].unique():
    mins.append(((hffm['FmLt_eyeT'][hffm['session']==session].iloc[0]).size)/60/60)
    Lgaze_count.append(len(hffm['FmLt_gazeshift_left_saccTimes_dHead1'][hffm['session']==session].iloc[0]))
    Rgaze_count.append(len(hffm['FmLt_gazeshift_right_saccTimes_dHead1'][hffm['session']==session].iloc[0]))
    Lcomp_count.append(len(hffm['FmLt_comp_left_saccTimes_dHead1'][hffm['session']==session].iloc[0]))
    Rcomp_count.append(len(hffm['FmLt_comp_right_saccTimes_dHead1'][hffm['session']==session].iloc[0]))

fig, ax = plt.subplots(1,1,figsize=(1.5,1.5),dpi=300)

ax.plot(jitter_ax(0, len(Lgaze_count)), Lgaze_count, '.', color=colors['gaze'])
ax.plot(jitter_ax(1, len(Rgaze_count)), Rgaze_count, '.', color=colors['gaze'])
ax.plot(jitter_ax(2, len(Lcomp_count)), Lcomp_count, '.', color=colors['comp'])
ax.plot(jitter_ax(3, len(Rcomp_count)), Rcomp_count, '.', color=colors['comp'])

for i, vals in enumerate([Lgaze_count, Rgaze_count, Lcomp_count, Rcomp_count]):
    center = i+1
    med = np.median(vals)
    err = np.std(vals) / np.sqrt(len(vals))
    ax.hlines(med, center-.2, center+.2, color='k', linewidth=2)
    ax.vlines(center, med-err, med+err, color='k', linewidth=2)

ax.set_xticks(np.arange(1,5,1), labels=['L','R','L','R'])
ax.set_xlim([0.5,4.5])
ax.set_ylim([0, 450])
ax.set_ylabel('saccades/min')

plt.tight_layout()
fig.savefig(os.path.join(figpath, 'hffm_saccade_rate-285.pdf'))

# movemement counts for freely moving light/dark
Lgaze_countL = []; Rgaze_countL = []
Lcomp_countL = []; Rcomp_countL = []
for session in ltdk['session'].unique():
    minutes = ((ltdk['FmLt_eyeT'][ltdk['session']==session].iloc[0]).size)/60/60
    Lgaze_countL.append(len(ltdk['FmLt_gazeshift_left_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Rgaze_countL.append(len(ltdk['FmLt_gazeshift_right_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Lcomp_countL.append(len(ltdk['FmLt_comp_left_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Rcomp_countL.append(len(ltdk['FmLt_comp_right_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
Lgaze_countD = []; Rgaze_countD = []
Lcomp_countD = []; Rcomp_countD = []
for session in ltdk['session'].unique():
    minutes = ((ltdk['FmDk_eyeT'][ltdk['session']==session].iloc[0]).size)/60/60
    Lgaze_countD.append(len(ltdk['FmDk_gazeshift_left_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Rgaze_countD.append(len(ltdk['FmDk_gazeshift_right_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Lcomp_countD.append(len(ltdk['FmDk_comp_left_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)
    Rcomp_countD.append(len(ltdk['FmDk_comp_right_saccTimes_dHead'][ltdk['session']==session].iloc[0])/minutes)

    fig, ax = plt.subplots(1,1,figsize=(1.5,1.5),dpi=300)

    ax.plot(jitter_ax(0, len(Lgaze_countL)), Lgaze_countL, '.', color=colors['gaze'])
    ax.plot(jitter_ax(1, len(Rgaze_countL)), Rgaze_countL, '.', color=colors['gaze'])
    ax.plot(jitter_ax(2, len(Lcomp_countL)), Lcomp_countL, '.', color=colors['comp'])
    ax.plot(jitter_ax(3, len(Rcomp_countL)), Rcomp_countL, '.', color=colors['comp'])


for i, vals in enumerate([Lgaze_countL, Rgaze_countL, Lcomp_countL, Rcomp_countL]):
    center = i+1
    med = np.median(vals)
    err = np.std(vals) / np.sqrt(len(vals))
    ax.hlines(med, center-.2, center+.2, color='k', linewidth=2)
    ax.vlines(center, med-err, med+err, color='k', linewidth=2)

ax.set_xticks(np.arange(1,5,1), labels=['L','R','L','R'])
ax.set_xlim([0.5,4.5])
ax.set_ylim([0, 450])
ax.set_ylabel('saccades/min')

plt.tight_layout()
fig.savefig(os.path.join(figpath, 'ltdk_Lt_saccade_rate.pdf'))

fig, ax = plt.subplots(1,1,figsize=(1.5,1.5),dpi=300)

ax.plot(jitter_ax(0, len(Lgaze_countD)), Lgaze_countD, '.', color=colors['gaze'])
ax.plot(jitter_ax(1, len(Rgaze_countD)), Rgaze_countD, '.', color=colors['gaze'])
ax.plot(jitter_ax(2, len(Lcomp_countD)), Lcomp_countD, '.', color=colors['comp'])
ax.plot(jitter_ax(3, len(Rcomp_countD)), Rcomp_countD, '.', color=colors['comp'])


for i, vals in enumerate([Lgaze_countD, Rgaze_countD, Lcomp_countD, Rcomp_countD]):
    center = i+1
    med = np.median(vals)
    err = np.std(vals) / np.sqrt(len(vals))
    ax.hlines(med, center-.2, center+.2, color='k', linewidth=2)
    ax.vlines(center, med-err, med+err, color='k', linewidth=2)

ax.set_xticks(np.arange(1,5,1), labels=['L','R','L','R'])
ax.set_xlim([0.5,4.5])
# ax.set_yticks(np.linspacLe(0,24000,6))
ax.set_ylim([0, 450])
ax.set_ylabel('saccades/min')

plt.tight_layout()
fig.savefig(os.path.join(figpath, 'ltdk_Dk_saccade_rate.pdf'))





    # Eye movement counts
    sessions = hffm['session'].unique()
    saccRates = np.zeros([len(sessions), 4]) # session, [gazeRun, gazeStill, compRun, compStill]

    for i, s in enumerate(sessions):

        row = hffm[hffm['session']==s].iloc[0].copy()
        eyeT = row['FmLt_eyeT']
        
        gazeTimes = []
        gazeTimes.extend(row['FmLt_gazeshift_left_saccTimes_dHead1'])
        gazeTimes.extend(row['FmLt_gazeshift_right_saccTimes_dHead1'])
        
        compTimes = []
        compTimes.extend(row['FmLt_comp_left_saccTimes_dHead1'])
        compTimes.extend(row['FmLt_comp_right_saccTimes_dHead1'])
        
        runTimes = eyeT[row['FmLt_top_speed']>2]
        stillTimes = eyeT[row['FmLt_top_speed']<2]
        
        dT = 1/60
        saccRates[i,0] = len([t for t in gazeTimes if t in runTimes]) / (len(runTimes)*dT*(1/60))
        saccRates[i,1] = len([t for t in gazeTimes if t in stillTimes]) / (len(stillTimes)*dT*(1/60))
        
        saccRates[i,2] = len([t for t in compTimes if t in runTimes]) / (len(runTimes)*dT*(1/60))
        saccRates[i,3] = len([t for t in compTimes if t in stillTimes]) / (len(stillTimes)*dT*(1/60))

    fig, [ax0, ax1] = plt.subplots(1,2,figsize=(2.75,1.75), dpi=300)

    ax0.plot(jitter_ax(0, len(saccRates[:,0])), saccRates[:,0], '.', color=colors['gaze'], markersize=3)
    ax0.plot(jitter_ax(1, len(saccRates[:,1])), saccRates[:,1], '.', color=colors['gaze'], markersize=3)

    ax1.plot(jitter_ax(0, len(saccRates[:,2])), saccRates[:,2], '.', color=colors['comp'], markersize=3)
    ax1.plot(jitter_ax(1, len(saccRates[:,3])), saccRates[:,3], '.', color=colors['comp'], markersize=3)

    for i, vals in enumerate([saccRates[:,0], saccRates[:,1]]):
        center = i+1
        med = np.median(vals)
        err = np.std(vals) / np.sqrt(len(vals))
        ax0.hlines(med, center-.2, center+.2, color='k', linewidth=2)
        ax0.vlines(center, med-err, med+err, color='k', linewidth=2)
        print(i, med, err)
        
    for i, vals in enumerate([saccRates[:,2], saccRates[:,3]]):
        center = i+1
        med = np.median(vals)
        err = np.std(vals) / np.sqrt(len(vals))
        ax1.hlines(med, center-.2, center+.2, color='k', linewidth=2)
        ax1.vlines(center, med-err, med+err, color='k', linewidth=2)
        print(i+2, med, err)

    ax0.set_xticks(np.arange(1,3,1), labels=['run','still'], rotation=90)
    ax0.set_xlim([0.5,2.5])
    ax0.set_ylim([0, np.max(saccRates[:,:2])+3])
    ax0.set_ylabel('saccade/min')

    ax1.set_xticks(np.arange(1,3,1), labels=['run','still'], rotation=90)
    ax1.set_xlim([0.5,2.5])
    ax1.set_ylim([0, np.max(saccRates[:,2:])+8])
    ax1.set_ylabel('saccade/min')

    plt.tight_layout()
    fig.savefig(os.path.join(figpath, 'hffm_saccade_rate-237.pdf'))


def position_during_saccades():
    
    ### position during saccades
    session_names = sorted(hffm['session'].unique())
    row = hffm[hffm['session']==session_names[1]].iloc[0].copy()

    gazeL = row['FmLt_gazeshift_left_saccTimes_dHead1'].copy()
    gazeR = row['FmLt_gazeshift_right_saccTimes_dHead1'].copy()
    compL = row['FmLt_comp_left_saccTimes_dHead1'].copy()
    compR = row['FmLt_comp_right_saccTimes_dHead1'].copy()

    session_names = sorted(hffm['session'].unique())

    theta_mean = np.zeros([len(session_names), 4, 50])
    head_mean = np.zeros([len(session_names), 4, 50])
    gaze_mean = np.zeros([len(session_names), 4, 50])

    for si, sess in tqdm(enumerate(session_names)):
        
        row = hffm[hffm['session']==sess].iloc[0].copy()
        
        print(sess, row.name)
        
        eyeT = row['FmLt_eyeT'].copy()
        
        head = interp1d(row['FmLt_imuT'], row['FmLt_gyro_z'], bounds_error=False)(eyeT)

        theta = row['FmLt_theta'].copy()
        
        # gaze = head.copy() + theta.copy()
        
        gazeL = row['FmLt_gazeshift_left_saccTimes_dHead1'].copy()
        gazeR = row['FmLt_gazeshift_right_saccTimes_dHead1'].copy()
        compL = row['FmLt_comp_left_saccTimes_dHead1'].copy()
        compR = row['FmLt_comp_right_saccTimes_dHead1'].copy()
        
        movs = [gazeL, gazeR, compL, compR]
        
        setmin = np.nanmin(eyeT)+2
        setmax = np.nanmax(eyeT)-2
        
        for x in range(4):

            eventT = movs[x].copy()
            eventT = eventT[eventT>setmin]
            eventT = eventT[eventT<setmax]

            theta_arr = np.zeros([len(eventT), 50])
            head_arr = np.zeros([len(eventT), 50])
            # gaze_arr = np.zeros([len(eventT), 100])

            for i, t in enumerate(eventT):
                tind = np.nanargmin(np.abs(eyeT-t))

                tind_use = tind + np.arange(-25,25)

                theta_arr[i,:] = theta[tind_use].copy()
                head_arr[i,:] = np.cumsum(head[tind_use].copy())*.016
                # gaze_arr[i,:] = gaze[tind_use].copy()

            theta_mean[si,x,:] = np.nanmean(theta_arr, axis=0)
            head_mean[si,x,:] = np.nanmean(head_arr, axis=0)
            # gaze_mean[si,x,:] = np.nanmean(gaze_arr, axis=0)

    session_names = sorted(hffm['session'].unique())
    row = hffm[hffm['session']==session_names[1]].iloc[0].copy()

    gazeL = row['FmLt_gazeshift_left_saccTimes_dHead1'].copy()
    gazeR = row['FmLt_gazeshift_right_saccTimes_dHead1'].copy()
    compL = row['FmLt_comp_left_saccTimes_dHead1'].copy()
    compR = row['FmLt_comp_right_saccTimes_dHead1'].copy()

    pos_bins = np.linspace(-250,250,50)

    fig, ax0 = plt.subplots(1,1, dpi=300, figsize=(2,2))

    l_gaze_pos = np.sum([theta_mean[:,0,:], head_mean[:,0,:]], axis=0)
    l_gaze = np.nanmean(l_gaze_pos, axis=0)
    l_gaze_err = stderr(l_gaze_pos, axis=0)

    l_comp_pos = np.sum([theta_mean[:,2,:], head_mean[:,2,:]], axis=0)
    l_comp = np.nanmean(l_comp_pos, axis=0)
    l_comp_err = stderr(l_comp_pos, axis=0)

    l_gaze = l_gaze-l_gaze[15]
    l_comp = l_comp-l_comp[15]

    ax0.plot(pos_bins, l_gaze, color=colors['gaze'])
    ax0.fill_between(pos_bins, l_gaze-l_gaze_err, l_gaze+l_gaze_err, color=colors['gaze'], alpha=0.2)

    ax0.plot(pos_bins, l_comp, color=colors['comp'])
    ax0.fill_between(pos_bins, l_comp-l_comp_err, l_comp+l_comp_err, color=colors['comp'], alpha=0.2)

    ax0.set_xlim([-100,250])

    ax0.set_xticks([-100, 0, 100, 200])

    # ax0.set_xticks(np.linspace(-250,250,3), labels=np.linspace(-250,250,3).astype(int))

    ax0.set_ylim([-9, 34])
    # ax0.set_yticks(np.arange(-10,50,10))

    ax0.set_ylabel('gaze (deg)')
    ax0.set_xlabel('time (ms)')

    ax0.vlines(0, -30, 50, color='k', linestyle='dashed', linewidth=1)

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'gaze_position_around_saccades_m100cent.pdf'))