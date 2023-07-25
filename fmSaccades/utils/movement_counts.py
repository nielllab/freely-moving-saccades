    







def movement_counts():

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