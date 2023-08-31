

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

import fmEphys as fme
import saccadeAnalysis as sacc


def _plotCol(ax, c, data, demoPt=None, color='k'):

    x_jitter = np.random.uniform(c-0.07, c+0.07, np.size(data,0))
    ax.plot(x_jitter, data, '.', color=color, markersize=3)
    if demoPt is not None:
        ax.plot(x_jitter[demoPt], data[demoPt], '.', color='tab:orange', markersize=4)


def figS1(hffm, ltdk, savepath):

    sacc.set_plt_params()
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']


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

    ax.plot(sacc.jitter_ax(0, len(Lgaze_count)), Lgaze_count, '.', color=colors['gaze'])
    ax.plot(sacc.jitter_ax(1, len(Rgaze_count)), Rgaze_count, '.', color=colors['gaze'])
    ax.plot(sacc.jitter_ax(2, len(Lcomp_count)), Lcomp_count, '.', color=colors['comp'])
    ax.plot(sacc.jitter_ax(3, len(Rcomp_count)), Rcomp_count, '.', color=colors['comp'])

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
    fig.savefig(os.path.join(savepath, 'hffm_saccade_rate-285.pdf'))

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

        fig, ax = plt.subplots(1,1, figsize=(1.5,1.5), dpi=300)

        ax.plot(sacc.jitter(0, len(Lgaze_countL)), Lgaze_countL, '.', color=colors['gaze'])
        ax.plot(sacc.jitter(1, len(Rgaze_countL)), Rgaze_countL, '.', color=colors['gaze'])
        ax.plot(sacc.jitter(2, len(Lcomp_countL)), Lcomp_countL, '.', color=colors['comp'])
        ax.plot(sacc.jitter(3, len(Rcomp_countL)), Rcomp_countL, '.', color=colors['comp'])


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
    fig.savefig(os.path.join(savepath, 'ltdk_Lt_saccade_rate.pdf'))

    fig, ax = plt.subplots(1,1,figsize=(1.5,1.5),dpi=300)

    ax.plot(sacc.jitter(0, len(Lgaze_countD)), Lgaze_countD, '.', color=colors['gaze'])
    ax.plot(sacc.jitter(1, len(Rgaze_countD)), Rgaze_countD, '.', color=colors['gaze'])
    ax.plot(sacc.jitter(2, len(Lcomp_countD)), Lcomp_countD, '.', color=colors['comp'])
    ax.plot(sacc.jitter(3, len(Rcomp_countD)), Rcomp_countD, '.', color=colors['comp'])


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
    fig.savefig(os.path.join(savepath, 'ltdk_Dk_saccade_rate.pdf'))



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

    ax0.plot(sacc.jitter(0, len(saccRates[:,0])), saccRates[:,0], '.', color=colors['gaze'], markersize=3)
    ax0.plot(sacc.jitter(1, len(saccRates[:,1])), saccRates[:,1], '.', color=colors['gaze'], markersize=3)

    ax1.plot(sacc.jitter(0, len(saccRates[:,2])), saccRates[:,2], '.', color=colors['comp'], markersize=3)
    ax1.plot(sacc.jitter(1, len(saccRates[:,3])), saccRates[:,3], '.', color=colors['comp'], markersize=3)

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
    fig.savefig(os.path.join(savepath, 'hffm_saccade_rate-237.pdf'))



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
    l_gaze_err = fme.stderr(l_gaze_pos, axis=0)

    l_comp_pos = np.sum([theta_mean[:,2,:], head_mean[:,2,:]], axis=0)
    l_comp = np.nanmean(l_comp_pos, axis=0)
    l_comp_err = fme.stderr(l_comp_pos, axis=0)

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
    fig.savefig(os.path.join(savepath, 'gaze_position_around_saccades_m100cent.pdf'))



    sessions = hffm['session'].unique()

    mean_pitch = np.zeros(len(sessions))*np.nan
    mean_roll = np.zeros(len(sessions))*np.nan

    for i, s in enumerate(sessions):
        row = hffm[hffm['session']==s].iloc[0].copy()
        
        mean_pitch[i] = np.nanmean(row['FmLt_pitch'])
        mean_roll[i] = np.nanmean(row['FmLt_roll'])

    fig, ax0 = plt.subplots(1,1,figsize=(1.8,2.1), dpi=300)

    demopt = 1

    cluster_data = mean_pitch
    c=0
    x_jitter = np.random.uniform(c-0.07, c+0.07, np.size(cluster_data,0))

    hline = np.nanmean(cluster_data)
    ax0.hlines(hline, c-0.1, c+0.1, color='k', linewidth=2)
    err = np.std(cluster_data) / np.sqrt(np.size(cluster_data))
    ax0.vlines(c, hline-err, hline+err, color='k', linewidth=2)
    ax0.set_xticks([0,0.5], labels=['pitch','roll'])
    ax0.set_ylim([-35,35])
    ax0.set_xlim([-0.3,0.8])
    # ax0.set_ylabel('head pitch (deg)')
    ax0.plot(x_jitter, cluster_data, '.', color='grey', markersize=4)
    ax0.plot(x_jitter[demopt], cluster_data[demopt], '.', color='tab:orange', markersize=4)
    print(hline, err)

    cluster_data = mean_roll
    c=0.5
    x_jitter = np.random.uniform(c-0.07, c+0.07, np.size(cluster_data,0))
    hline = np.nanmean(cluster_data)
    ax0.hlines(hline, c-0.1, c+0.1, color='k', linewidth=2)
    err = np.std(cluster_data) / np.sqrt(np.size(cluster_data))
    ax0.vlines(c, hline-err, hline+err, color='k', linewidth=2)
    # ax0.set_xticks([])
    # ax.set_ylim([0,-45])
    # ax0.set_xlim([-0.3,0.3])
    ax0.set_ylabel('mean orientation (deg)')
    ax0.plot(x_jitter, cluster_data, '.', color='grey', markersize=4)
    ax0.plot(x_jitter[demopt], cluster_data[demopt], '.', color='tab:orange', markersize=4)
    # ax0.set_ylim([-35,35])
    print(hline, err)

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'mean_head_pitch_roll.pdf'))


    
    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(5,2), dpi=300)

    _usebins = np.linspace(-50,50,20)

    useRow = hffm[hffm['session']=='070921_J553RT_control_Rig2'].copy()
    usePitch = useRow['FmLt_pitch'].iloc[0]
    useRoll = useRow['FmLt_roll'].iloc[0]

    ax0.hist(usePitch, alpha=0.5, color='k', density=True, bins=_usebins)

    ax1.hist(useRoll, alpha=0.5, color='k', density=True, bins=_usebins)

    ax0.set_xlim([-50,50])
    ax1.set_xlim([-50,50])

    ax0.set_xlabel('head pitch (deg)')
    ax1.set_xlabel('head roll (deg)')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, '0709_hist_pitch_roll.pdf'))

    _mean = np.nanmean(useRoll)
    _stderr = np.std(useRoll) / np.sqrt(np.size(useRoll))
    print(_mean, _stderr)



    
    ### firing rates hf vs fm
    fig, [ax0,ax1] = plt.subplots(1,2, figsize=(4.3,2.2), dpi=300)

    clusters = ['early','late','biphasic','negative']

    use_xlabels = []

    for c, name in enumerate(clusters):
        
        gazebase = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        gtISI = hffm['Gt_drift_spont'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        
        ax0.plot(gtISI, gazebase, '.', color=colors[name], markersize=2)
        
        sacc.make_category_col(ax1, c, gtISI, colors[name])
        sacc.make_category_col(ax1, c+0.4, gazebase, colors[name])
        
        use_xlabels.extend(['hf','fm'])

    ax1.set_xticks([0,0.4,1,1.4,2,2.4,3,3.4], labels=use_xlabels, rotation=90)
        
    ax1.set_ylim([0,20])
    ax1.set_ylabel('baseline firing rate (sp/s)')

    ax0.plot([0,60],[0,60],color='k')
    ax0.set_ylim([0,30])
    ax0.set_xlim([0,30])
    ax0.set_xlabel('head-fixed (sp/s)')
    ax0.set_ylabel('freely moving (sp/s)')
    # ax0.axis('equal')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'baseline_hfm_fm_scatters.pdf'))



    fig, axs = plt.subplots(2,3, figsize=(5, 3.1), dpi=300)

    clusters = ['early','late','biphasic','negative','unresponsive']

    axs = axs.flatten()

    for c, name in enumerate(clusters):
        
        ax = axs[c]
        
        gazebase = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        gtISI = hffm['Gt_drift_spont'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        
        ax.plot(gtISI, gazebase, '.', color=colors[name], markersize=2)
        
        res = sacc.plot_regression(ax, gtISI, gazebase)
        
        print(res.rvalue, res.pvalue, res.slope)
        
        # ax.axis('equal')
        ax.set_xlim([0,30])
        ax.set_ylim([0,30])
        ax.set_xticks(np.linspace(0,30,4).astype(int))
        ax.set_yticks(np.linspace(0,30,4).astype(int))
        
    axs[0].set_ylabel('freely moving (sp/s)')
    axs[3].set_xlabel('head-fixed (sp/s)')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'baseline_hfm_fm_scatters.pdf'))



    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(3.5,2), dpi=300)

    names = ['early','late','biphasic','negative','unresponsive']
    print_names = ['early','late','biph','neg','unresp']

    for i, name in enumerate(names):
        baselines = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].to_numpy()
        err = np.std(baselines) / np.sqrt(np.size(baselines))
        med = np.median(baselines)
        ax0.bar(i, med, color=colors[name])
        ax0.plot([i, i], [med-err, med+err], 'k-')
    ax0.set_xticks(range(5), print_names, rotation=90)
    ax0.set_ylabel('baseline (sp/s)')
    ax0.set_title('freely moving')

    for i, name in enumerate(names):
        baselines = hffm['Gt_drift_spont'][hffm['gazecluster']==name].to_numpy()
        err = np.std(baselines) / np.sqrt(np.size(baselines))
        med = np.median(baselines)
        ax1.bar(i, med, color=colors[name])
        ax1.plot([i, i], [med-err, med+err], 'k-')
    ax1.set_xticks(range(5), print_names, rotation=90)
    ax1.set_ylabel('baseline (sp/s)')
    ax1.set_title('head-fixed')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'baseline_rate_bar_plot.pdf'))




    ### intermediate eye/head movements
    shifted_head = 60 # deg/sec
    still_gaze = 120 # deg/sec
    shifted_gaze = 240 # deg/sec

    sessions = hffm['session'].unique()

    # for s in sessions:
    row = hffm[hffm['session']=='102621_J558NC_control_Rig2'].iloc[0].copy()

    dGaze = row['FmLt_dGaze']
    eyeT = row['FmLt_eyeT'][:-1]
    dHead = row['FmLt_dHead']
    dEye = row['FmLt_dEye_dps']

    gazeL = (dHead > shifted_head) & (dGaze > shifted_gaze)
    gazeR = (dHead < -shifted_head) & (dGaze < -shifted_gaze)

    compL = (dHead > shifted_head) & (dGaze < still_gaze) & (dGaze > -still_gaze)
    compR = (dHead < -shifted_head) & (dGaze > -still_gaze) & (dGaze < still_gaze)

    interL = (dHead > shifted_head) & (dGaze > still_gaze) & (dGaze > -still_gaze) & (dGaze < shifted_gaze)
    interR = (dHead < -shifted_head) & (dGaze < -still_gaze) & (dGaze < still_gaze) & (dGaze > -shifted_gaze)

    plt.figure(figsize=(3,2.5),dpi=300)
    plt.xlim([-600,600])
    plt.ylim([-600,600])
    plt.yticks(np.linspace(-600,600,4).astype(int))
    plt.xticks(np.linspace(-600,600,4).astype(int))
    useColors=[colors['gaze'],colors['gaze'],colors['comp'],colors['comp'],'k','k']
    for i, inds in enumerate([gazeL, gazeR, compL, compR, interL, interR]):
        plt.plot(dHead[inds][::25], dEye[inds][::25], '.', markersize=1, color=useColors[i])
        
    plt.ylabel('eye velocity (deg/s)')
    plt.xlabel('head velocity (deg/s)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'dEye_dHead_with_inter.pdf'))


    shifted_head = 60 # deg/sec
    still_gaze = 120 # deg/sec
    shifted_gaze = 240 # deg/sec

    sessions = hffm['session'].unique()

    interTimes = {}

    blank_arr = np.zeros(2001)*np.nan
    blank_series = pd.Series()
    for ind in hffm.index.values:
        blank_series.at[ind] = blank_arr.astype(object)
    hffm['FmLt_intersacc_left_saccPSTH_dHead1'] = blank_series
    hffm['FmLt_intersacc_right_saccPSTH_dHead1'] = blank_series

    for s in sessions:
        row = hffm[hffm['session']==s].iloc[0].copy()

        dGaze = row['FmLt_dGaze']
        eyeT = row['FmLt_eyeT'][:-1]
        dHead = row['FmLt_dHead']
        dEye = row['FmLt_dEye_dps']
        
        LgazeTimes = row['FmLt_gazeshift_left_saccTimes_dHead1']
        RgazeTimes = row['FmLt_gazeshift_right_saccTimes_dHead1']

        LcompTimes = row['FmLt_comp_left_saccTimes_dHead1']
        RcompTimes = row['FmLt_comp_right_saccTimes_dHead1']

        # gazeL = [(dHead > shifted_head) & (dGaze > shifted_gaze)]
        # gazeR = [(dHead < -shifted_head) & (dGaze < -shifted_gaze)]

        # compL = [(dHead > shifted_head) & (dGaze < still_gaze) & (dGaze > -still_gaze)]
        # compR = [(dHead < -shifted_head) & (dGaze > -still_gaze) & (dGaze < still_gaze)]

        interL = (dHead > shifted_head) & (dGaze > still_gaze) & (dGaze > -still_gaze) & (dGaze < shifted_gaze)
        interR = (dHead < -shifted_head) & (dGaze < -still_gaze) & (dGaze < still_gaze) & (dGaze > -shifted_gaze)
        
        # interL1 = drop_repeat_events(drop_nearby_events(eyeT[interL], LgazeTimes))
        # interR1 = drop_repeat_events(drop_nearby_events(eyeT[interR], RgazeTimes))
        
        interL1 = fme.drop_repeat_events(eyeT[interL])
        interR1 = fme.drop_repeat_events(eyeT[interR])
        
        tmpDict = {}
        tmpDict['leftTimes'] = interL
        tmpDict['rightTimes'] = interR
        tmpDict['leftTimes1'] = interL1
        tmpDict['rightTimes1'] = interR1
        
        interTimes[s] = tmpDict
        
        # print(s)
        # for ind, row in tqdm(hffm[hffm['session']==s].iterrows()):
        #     sps = row['FmLt_spikeT'].copy()
        #     hffm.at[ind, 'FmLt_intersacc_left_saccPSTH_dHead1'] = calc_kde_PSTH(sps, interL1)
        #     hffm.at[ind, 'FmLt_intersacc_right_saccPSTH_dHead1'] = calc_kde_PSTH(sps, interR1)


    ex_units = [215, 579, 277]

    fig, axs = plt.subplots(1,3, dpi=300, figsize=(6.5,2))

    for i, ind in enumerate(ex_units):
        
        LR = hffm.loc[ind, 'pref_gazeshift_direction']
        axs[i].plot(psth_bins, hffm.loc[ind, 'FmLt_comp_{}_saccPSTH_dHead1'.format(LR)], color=colors['comp'])
        axs[i].plot(psth_bins, hffm.loc[ind, 'FmLt_gazeshift_{}_saccPSTH_dHead1'.format(LR)], color=colors['gaze'])
        axs[i].plot(psth_bins, hffm.loc[ind, 'FmLt_intersacc_{}_saccPSTH_dHead1'.format(LR)], color='k')
        
        max_fr = np.nanmax(hffm.loc[ind, 'FmLt_gazeshift_{}_saccPSTH_dHead1'.format(LR)])*1.1
        axs[i].set_ylim([0, max_fr])
        axs[i].set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
        axs[i].set_xlim([-.5,.5])
        if i == 0:
            axs[i].set_ylabel('spike rate (sp/s)')
        axs[i].set_xlabel('time (ms)')
        axs[i].vlines(0, 0, max_fr, color='k', linewidth=1, linestyle='dashed')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'example_inter_PSTH_units.pdf'))

    hffm['pref_intersacc_psth'] = hffm['FmLt_intersacc_left_saccPSTH_dHead1'].copy().astype(object)
    for ind, row in tqdm(hffm.iterrows()):
        LR = row['pref_gazeshift_direction']
        if type(LR)==str:
            hffm.at[ind,'pref_intersacc_psth'] = sacc.norm_PSTH(row['FmLt_intersacc_{}_saccPSTH_dHead1'.format(LR)])
        else:
            hffm.at[ind,'pref_intersacc_psth'] = np.nan

    fig, [ax_ex_gaze, ax_ex_inter, ax_ex_comp] = plt.subplots(3,1,dpi=300,figsize=(2.5,4))

    possible_inds = hffm['pref_comp_psth'][hffm['Fm_fr']>2].index.values
    np.random.seed(2)
    example_inds = np.sort(np.random.choice(possible_inds, size=100, replace=False))

    for ind in example_inds:
        ax_ex_gaze.plot(psth_bins, hffm.loc[ind,'pref_gazeshift_psth'].astype(float), linewidth=1, alpha=0.3)
        ax_ex_comp.plot(psth_bins, hffm.loc[ind,'pref_comp_psth'].astype(float), linewidth=1, alpha=0.3)
        
        # LR = hffm.loc[ind, 'pref_gazeshift_direction']
        # pref_inter = normalize_psth(hffm.loc[ind, 'FmLt_intersacc_{}_saccPSTH_dHead1'.format(LR)])
        
        ax_ex_inter.plot(psth_bins, hffm.loc[ind,'pref_intersacc_psth'].astype(float), linewidth=1, alpha=0.3)
    ax_ex_gaze.set_xlim([-.5,.5])
    ax_ex_inter.set_xlim([-.5,.5])
    # ax_ex_inter.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
    ax_ex_inter.set_ylim([-0.75,1])
    ax_ex_gaze.set_yticks([-0.5,0,0.5,1])
    ax_ex_inter.set_yticks([-0.5,0,0.5,1])
    ax_ex_comp.set_yticks([-0.5,0,0.5,1])
    # ax_ex_gaze.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
    ax_ex_comp.set_xlim([-.5,.5])
    ax_ex_comp.set_xticks(np.linspace(-.5,.5,5), labels=np.linspace(-500,500,5).astype(int))
    ax_ex_gaze.set_xticks(np.linspace(-.5,.5,5), labels=[])
    ax_ex_inter.set_xticks(np.linspace(-.5,.5,5), labels=[])
    ax_ex_gaze.set_ylim([-0.75,1])
    ax_ex_comp.set_ylim([-0.75,1])
    ax_ex_gaze.set_ylabel('norm. sp/s')
    ax_ex_comp.set_ylabel('norm. sp/s')
    ax_ex_inter.set_ylabel('norm. sp/s')
    ax_ex_comp.set_xlabel('time (ms)')
    # ax_ex_gaze.set_xlabel('time (ms)')

    all_comp = fme.flatten_series(hffm['pref_comp_psth'][hffm['gazecluster']!='unresponsive'])
    all_gaze = fme.flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']!='unresponsive'])
    all_inter = fme.flatten_series(hffm['pref_intersacc_psth'][hffm['gazecluster']!='unresponsive'])

    comp_mean = np.nanmean(all_comp,0)
    comp_std = np.std(all_comp,0) / np.sqrt(np.size(all_comp))

    gaze_mean = np.nanmean(all_gaze,0)
    gaze_std = np.std(all_gaze,0) / np.sqrt(np.size(all_gaze))

    inter_mean = np.nanmean(all_inter,0)
    inter_std = np.std(all_inter,0) / np.sqrt(np.size(all_inter))
    # ax_ex_comp.set_title('compensatory')
    # ax_ex_gaze.set_title('gaze-shifting')
    ax_ex_gaze.plot(psth_bins, gaze_mean, color=colors['gaze'], linewidth=2)

    ax_ex_comp.plot(psth_bins, comp_mean, color=colors['comp'], linewidth=2)

    ax_ex_inter.plot(psth_bins, inter_mean, color='k', linewidth=2)

    ax_ex_gaze.vlines(0, -0.75, 1, 'k', linewidth=1, linestyle='dashed')
    ax_ex_comp.vlines(0, -0.75, 1, 'k', linewidth=1, linestyle='dashed')
    ax_ex_inter.vlines(0, -0.75, 1, 'k', linewidth=1, linestyle='dashed')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath,'gaze_inter_comp_means.pdf'))


    
    ### mean amplitude of eye movements

    sessions = hffm['session'].unique()

    dGaze_sizes = {}

    for s in sessions:
        row = hffm[hffm['session']==s].iloc[0].copy()
        
        dGaze = row['FmLt_dGaze']
        eyeT = row['FmLt_eyeT']

        gazeTimes = []
        gazeTimes.extend(row['FmLt_gazeshift_left_saccTimes_dHead1'])
        gazeTimes.extend(row['FmLt_gazeshift_right_saccTimes_dHead1'])

        compTimes = []
        compTimes.extend(row['FmLt_comp_left_saccTimes_dHead1'])
        compTimes.extend(row['FmLt_comp_right_saccTimes_dHead1'])
        
        interTimesList = []
        interTimesList.extend(list(interTimes[s]['leftTimes1']))
        interTimesList.extend(list(interTimes[s]['rightTimes1']))
        # interTimesList = drop_repeat_events(np.array(sorted(interTimesList)))
        
        
        dGaze_gazeshift = np.zeros(np.size(gazeTimes))*np.nan
        dGaze_compensatory = np.zeros(np.size(compTimes))*np.nan
        dGaze_inter = np.zeros(np.size(interTimesList))*np.nan
        
        for i, t in enumerate(gazeTimes):
            saccInd = np.argwhere(eyeT==t)[0][0]
            dGaze_gazeshift[i] = np.abs(dGaze[saccInd])
            
        for i, t in enumerate(compTimes):
            saccInd = np.argwhere(eyeT==t)[0][0]
            dGaze_compensatory[i] = np.abs(dGaze[saccInd])
            
        for i, t in enumerate(interTimesList):
            saccInd = np.argwhere(eyeT==t)[0][0]
            dGaze_inter[i] = np.abs(dGaze[saccInd])
            
        tmpDict = {}
        tmpDict['gaze'] = dGaze_gazeshift
        tmpDict['comp'] = dGaze_compensatory
        tmpDict['inter'] = dGaze_inter
            
        dGaze_sizes[s] = tmpDict


    _useGazeVals = []
    _useInterVals = []
    _useCompVals = []

    for i, k in enumerate(dGaze_sizes.keys()):
        
        _vals = np.nanmean(dGaze_sizes[k]['gaze'])
        _useGazeVals.append(_vals)
        
        _vals = np.nanmean(dGaze_sizes[k]['inter'])
        _useInterVals.append(_vals)
        
        _vals = np.nanmean(dGaze_sizes[k]['comp'])
        _useCompVals.append(_vals)

        hline = np.nanmedian(data)
        ax.hlines(hline, c-0.1, c+0.1, color='k', linewidth=2)
        err = np.std(data) / np.sqrt(np.size(data))
        ax.vlines(c, hline-err, hline+err, color='k', linewidth=2)

        print(hline, err)
        
    fig, ax = plt.subplots(1,1,figsize=(1.75,2.5), dpi=300)

    _plotCol(ax, 0, np.array(_useCompVals) * (1/60), demoPt=1, color=colors['comp'])
    _plotCol(ax, 0.5, np.array(_useInterVals) * (1/60), demoPt=1, color='grey')
    _plotCol(ax, 1, np.array(_useGazeVals) * (1/60), demoPt=1, color=colors['gaze'])

    ax.set_ylim([0, 11])
    ax.set_ylabel('gaze amplitude (deg)')
    ax.set_xticks([0,0.5,1], labels=['comp','inter','gaze'], rotation=90)
    
    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'mean_gaze_amplitude_columns.pdf'))


    plt.figure(dpi=300, figsize=(3,2))

    plt.hist(dGaze_sizes[sessions[1]]['gaze'] * (1/60), bins=np.arange(4,60,.2), color=colors['gaze'])

    plt.hist(dGaze_sizes[sessions[1]]['inter'] * (1/60), bins=np.arange(2,4.2,.2), color='k')

    plt.hist(dGaze_sizes[sessions[1]]['comp'] * (1/60), bins=np.arange(0,2.2,.2), color=colors['comp'])

    plt.xlim([0,25])

    plt.tight_layout()

    plt.xlabel('gaze shift amplitude (deg)')
    plt.ylabel('movement count')

    plt.savefig(os.path.join(savepath, 'gaze_shift_amp_w_inter.pdf'))

