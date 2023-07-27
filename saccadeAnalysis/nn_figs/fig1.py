
"""
Demo data
for figure 1


"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import fmEphys as fme
import saccadeAnalysis as sacc


def fig1_demo_data(hffm):

    # Plotting parameters.
    sacc.set_plt_params()
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']

    # Choose session to use for behavior demo data.
    demo = hffm[hffm['session']=='102621_J558NC_control_Rig2'].iloc[0]

    dHead_data = demo['FmLt_dHead']
    dEye_data = demo['FmLt_dEye_dps']
    dGaze_data = demo['FmLt_dGaze']
    eyeT = demo['FmLt_eyeT']

    left = demo['FmLt_gazeshift_left_saccTimes_dHead1']
    right = demo['FmLt_gazeshift_right_saccTimes_dHead1']

    comp = np.hstack([demo['FmLt_comp_left_saccTimes_dHead1'],
                      demo['FmLt_comp_right_saccTimes_dHead1']])

    plotinds = np.sort(np.random.choice(np.arange(eyeT.size),
                                        size=int(np.ceil(eyeT.size/25)),
                                        replace=False))
    
    gazemovs = np.hstack([left, right])

    # Fig1 panels D,E,F,G
    fig = plt.figure(constrained_layout=True, figsize=(9,8), dpi=300)
    figspec = gridspec.GridSpec(nrows=5, ncols=3,
                                figure=fig,
                                wspace=1.5, hspace=1.5)

    specA = gridspec.GridSpecFromSubplotSpec(3, 1,
                                             subplot_spec=figspec[0:2,1],
                                             wspace=0, hspace=0.01)
    ax_theta = fig.add_subplot(specA[0,0])
    ax_yaw = fig.add_subplot(specA[1,0])
    ax_gaze = fig.add_subplot(specA[2,0])

    specB = gridspec.GridSpecFromSubplotSpec(1, 1,
                                             subplot_spec=figspec[0:2,2],
                                             wspace=0, hspace=0)
    ax_dEyeHead = fig.add_subplot(specB[0,0])

    specC = gridspec.GridSpecFromSubplotSpec(3, 3,
                                             subplot_spec=figspec[2:,0:2],
                                             wspace=0.15, hspace=-.05)
    ax_pos_rasterG = fig.add_subplot(specC[0,0])
    ax_biph_rasterG = fig.add_subplot(specC[0,1])
    ax_neg_rasterG = fig.add_subplot(specC[0,2])
    ax_pos_rasterC = fig.add_subplot(specC[1,0])
    ax_biph_rasterC = fig.add_subplot(specC[1,1])
    ax_neg_rasterC = fig.add_subplot(specC[1,2])
    ax_pos_psth = fig.add_subplot(specC[2,0])
    ax_biph_psth = fig.add_subplot(specC[2,1])
    ax_neg_psth = fig.add_subplot(specC[2,2])

    specD = gridspec.GridSpecFromSubplotSpec(2, 1,
                                             subplot_spec=figspec[2:,2:],
                                             wspace=0.2, hspace=0.1)
    ax_ex_gaze = fig.add_subplot(specD[0,0])
    ax_ex_comp = fig.add_subplot(specD[1,0])

    start = 2090
    win = 60 # frames, not sec
    ex_units = [215, 579, 277]

    ylim_val = 20

    # Theta
    theta_data = demo['FmLt_theta'][start:start+win]
    theta_data = theta_data - np.nanmean(theta_data)
    ax_theta.plot(theta_data, 'k-', linewidth=2, scaley=10)
    ax_theta.set_xlim([0,60])
    ax_theta.set_xticks(ticks=np.linspace(0,60,5),
                        labels=np.linspace(0,1,5))
    ax_theta.set_ylabel('theta (deg)')
    ax_theta.set_ylim([-ylim_val, ylim_val])
    ax_theta.axes.get_xaxis().set_visible(False)
    ax_theta.axes.spines.bottom.set_visible(False)

    # Head yaw
    pYaw = np.nancumsum(demo['FmLt_dHead'][start:start+win]*0.016)
    pYaw = pYaw - np.nanmean(pYaw)
    ax_yaw.plot(pYaw, 'k-', linewidth=2)
    ax_yaw.set_xlim([0,60])
    ax_yaw.set_xticks(ticks=np.linspace(0,60,5),
                      labels=np.linspace(0,1,5))
    ax_yaw.set_ylabel('yaw (deg)')
    ax_yaw.axes.get_xaxis().set_visible(False)
    ax_yaw.axes.spines.bottom.set_visible(False)
    ax_yaw.set_ylim([-ylim_val, ylim_val])

    # Gaze position
    ax_gaze.plot(pYaw + theta_data, 'k-', linewidth=2)
    ax_gaze.set_xlim([0,60])
    ax_gaze.set_xticks(ticks=np.linspace(0,60,5), labels=np.linspace(0,1000,5).astype(int))
    ax_gaze.set_ylabel('gaze (deg)')
    ax_gaze.set_ylim([-ylim_val,ylim_val])
    ax_gaze.set_xlabel('time (msec)')

    # 
    for i in plotinds:
        dGaze_i = np.abs(dHead_data[i]+dEye_data[i])
        if (eyeT[i] in gazemovs) or (dGaze_i>240):
            c = colors['gaze']
        elif (eyeT[i] in comp) or (dGaze_i<120):
            c = colors['comp']
        elif (dGaze_i<240) and (dGaze_i>120): 
            c = 'dimgray'
        else:
            continue
        ax_dEyeHead.plot(dHead_data[i], dEye_data[i], '.', color=c, markersize=2)

    ax_dEyeHead.set_aspect('equal','box')
    ax_dEyeHead.set_xlim([-600,600])
    ax_dEyeHead.set_ylim([-600,600])
    ax_dEyeHead.set_xlabel('head velocity (deg/sec)')
    ax_dEyeHead.set_ylabel('eye velocity (deg/sec)')
    ax_dEyeHead.plot([-500,500],[500,-500], linestyle='dashed', color='k', linewidth=1)

    ax_dEyeHead.set_xticks(np.linspace(-600,600,5))
    ax_dEyeHead.set_yticks(np.linspace(-600,600,5))

    num_movements = 500
    raster_panelsG = [ax_pos_rasterG, ax_biph_rasterG, ax_neg_rasterG]
    raster_panelsC = [ax_pos_rasterC, ax_biph_rasterC, ax_neg_rasterC]
    sdf_panels = [ax_pos_psth, ax_biph_psth, ax_neg_psth]

    for i, u in enumerate(ex_units):
        row = hffm.iloc[u]
        rasterG = raster_panelsG[i]
        rasterC = raster_panelsC[i]
        sdf_panel = sdf_panels[i]
        LR = hffm.loc[u, 'pref_gazeshift_direction']
        
        rasterG.set_title(['positive','biphasic','negative'][i])

        gazeshifts = row['FmLt_gazeshift_{}_saccTimes_dHead1'.format(LR)].copy()
        compmovs = np.hstack([row['FmLt_comp_left_saccTimes_dHead1'], row['FmLt_comp_right_saccTimes_dHead1']])
        
        plot_gs = np.random.choice(gazeshifts, size=num_movements, replace=False)
        plot_cp = np.random.choice(compmovs, size=num_movements, replace=False)

        for n, s in enumerate(plot_gs):
            sp = row['FmLt_spikeT']-s
            sp = sp[np.abs(sp)<=0.5]
            rasterG.plot(sp, np.ones(sp.size)*n, '|', color=colors['gaze'], markersize=0.3)

        for n, s in enumerate(plot_cp):
            sp = row['FmLt_spikeT']-s
            sp = sp[np.abs(sp)<=0.5]
            rasterC.plot(sp, np.ones(sp.size)*n, '|', color=colors['comp'], markersize=0.3)
        
        rasterG.set_ylim([num_movements, 0]); rasterC.set_ylim([num_movements,0])
        rasterG.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
        rasterC.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
        
        if i == 0:
            rasterG.set_ylabel('gaze shifts'); rasterC.set_ylabel('compensatory')
            rasterG.set_yticks(np.linspace(0, num_movements, 3))
            rasterC.set_yticks(np.linspace(0, num_movements, 3))
        
        else:
            rasterG.set_yticks(np.linspace(0, num_movements, 3),labels=[])
            rasterC.set_yticks(np.linspace(0, num_movements, 3),labels=[])
        
        rasterG.set_xticks([]); rasterC.set_xticks([])
        rasterG.set_xlim([-.5,.5]); rasterC.set_xlim([-.5,.5])
        rasterG.axes.spines.bottom.set_visible(False); rasterC.axes.spines.bottom.set_visible(False)
        
        sdf_panel.plot(psth_bins, row['FmLt_comp_{}_saccPSTH_dHead1'.format(LR)], color=colors['comp'])
        sdf_panel.plot(psth_bins, row['FmLt_gazeshift_{}_saccPSTH_dHead1'.format(LR)], color=colors['gaze'])
        max_fr = np.nanmax(row['FmLt_gazeshift_{}_saccPSTH_dHead1'.format(LR)])*1.1
        sdf_panel.set_ylim([0,max_fr])
        sdf_panel.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
        sdf_panel.set_xlim([-.5,.5])
        if i == 0:
            sdf_panel.set_ylabel('spike rate (sp/sec)')
        sdf_panel.set_xlabel('time (msec)')
        sdf_panel.vlines(0, 0, max_fr, color='k', linewidth=1, linestyle='dashed')

    possible_inds = hffm['pref_comp_psth'][hffm['Fm_fr']>2].index.values
    np.random.seed(2)
    example_inds = np.sort(np.random.choice(possible_inds, size=100, replace=False))

    for ind in example_inds:
        ax_ex_gaze.plot(psth_bins, hffm.loc[ind,'pref_gazeshift_psth'].astype(float), linewidth=1, alpha=0.3)
        ax_ex_comp.plot(psth_bins, hffm.loc[ind,'pref_comp_psth'].astype(float), linewidth=1, alpha=0.3)
    ax_ex_gaze.set_xlim([-.5,.5])
    ax_ex_gaze.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
    ax_ex_comp.set_xlim([-.5,.5])
    ax_ex_comp.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
    ax_ex_gaze.set_ylim([-0.75,1])
    ax_ex_comp.set_ylim([-0.75,1])
    ax_ex_gaze.set_ylabel('norm. spike rate')
    ax_ex_comp.set_ylabel('norm. spike rate')
    ax_ex_comp.set_xlabel('time (msec)')
    ax_ex_gaze.set_xlabel('time (msec)')

    all_comp = fme.flatten_series(hffm['pref_comp_psth'][hffm['gazecluster']!='unresponsive'][hffm['gazeshift_responsive']])
    all_gaze = fme.flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']!='unresponsive'][hffm['gazeshift_responsive']])

    comp_mean = np.nanmean(all_comp,0)
    comp_std = np.std(all_comp,0) / np.sqrt(np.size(all_comp))

    gaze_mean = np.nanmean(all_gaze,0)
    gaze_std = np.std(all_gaze,0) / np.sqrt(np.size(all_gaze))
    ax_ex_comp.set_title('compensatory')
    ax_ex_gaze.set_title('gaze-shifting')
    ax_ex_comp.plot(psth_bins, comp_mean, color=colors['comp'], linewidth=3)
    ax_ex_gaze.plot(psth_bins, gaze_mean, color=colors['gaze'], linewidth=3)

    ax_ex_gaze.vlines(0, -0.75, 1, 'k', linewidth=1, linestyle='dashed')
    ax_ex_comp.vlines(0, -0.75, 1, 'k', linewidth=1, linestyle='dashed')

    fig1.savefig(os.path.join(figpath, '1_gazeshift_v_comp.pdf'))




def figure1B():
    """
    example rasters
    """
    
    gaze = row['FmLt_gazeshift_{}_saccTimes_dHead1'.format(row['pref_gazeshift_direction'])].copy()
    comp = row['FmLt_comp_{}_saccTimes_dHead1'.format(row['pref_gazeshift_direction'])].copy()

    gaze_rand = np.random.choice(gaze, size=500, replace=False)
    comp_rand = np.random.choice(comp, size=500, replace=False)

    n_inds = np.arange(1000)
    gaze_inds = np.array(sorted(np.random.choice(n_inds, size=500, replace=False)))
    comp_inds = np.array(sorted(np.delete(n_inds.copy(), gaze_inds.copy())))

    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(6,3.5), dpi=300)
    row = hffm.loc[215].copy()

    for i in range(gaze_inds.size):
        
        sacc_time = gaze_rand[i]
        raster_row = gaze_inds[i]
        
        sp = row['FmLt_spikeT'] - sacc_time
        sp = sp[np.abs(sp) <= 0.5]
        ax0.plot(sp, np.ones(sp.size)*raster_row, '|', color='k', markersize=0.5)
        ax1.plot(sp, np.ones(sp.size)*raster_row, '|', color=colors['gaze'], markersize=0.5)
        
    for i in range(comp_inds.size):
        
        sacc_time = comp_rand[i]
        raster_row = comp_inds[i]
        
        sp = row['FmLt_spikeT'] - sacc_time
        sp = sp[np.abs(sp) <= 0.5]
        ax0.plot(sp, np.ones(sp.size)*raster_row, '|', color='k', markersize=0.5)
        ax1.plot(sp, np.ones(sp.size)*raster_row, '|', color=colors['comp'], markersize=0.5)

    ax0.set_ylim([1000, 0]); ax0.set_xlim([-0.5, 0.5])
    ax1.set_ylim([1000, 0]); ax1.set_xlim([-0.5, 0.5])
    ax0.vlines(0, 0, 1000, 'k', linewidth=0.75)
    ax1.vlines(0, 0, 1000, 'k', linewidth=0.75)
    ax0.set_ylabel('eye movement'); ax1.set_ylabel('eye movement')
    ax0.set_xlabel('time (ms)'); ax1.set_xlabel('time (ms)')
    ax0.set_xticks(np.linspace(-.5, .5, 5), labels=np.linspace(-500, 500, 5).astype(int))
    ax1.set_xticks(np.linspace(-.5, .5, 5), labels=np.linspace(-500, 500, 5).astype(int))

    fig.tight_layout()

    fig.savefig(os.path.join(figpath, '0_gaze_comp_rasters_1.pdf'))


    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(6,2), dpi=300)
    row = hffm.loc[215].copy()

    for n, i in enumerate(range(gaze_inds.size)):
        
        sacc_time = gaze_rand[i]
        # raster_row = gaze_inds[i]
        
        sp = row['FmLt_spikeT'] - sacc_time
        sp = sp[np.abs(sp) <= 0.5]
        # ax0.plot(sp, np.ones(sp.size)*raster_row, '|', color='k', markersize=0.3)
        ax0.plot(sp, np.ones(sp.size)*n, '|', color=colors['gaze'], markersize=0.3)
        
    for n, i in enumerate(range(comp_inds.size)):
        
        sacc_time = comp_rand[i]
        # raster_row = comp_inds[i]
        
        sp = row['FmLt_spikeT'] - sacc_time
        sp = sp[np.abs(sp) <= 0.5]
        # ax0.plot(sp, np.ones(sp.size)*raster_row, '|', color='k', markersize=0.3)
        ax1.plot(sp, np.ones(sp.size)*n, '|', color=colors['comp'], markersize=0.3)

    ax0.set_ylim([500, 0]); ax0.set_xlim([-0.5, 0.5])
    ax1.set_ylim([500, 0]); ax1.set_xlim([-0.5, 0.5])
    ax0.vlines(0, 0, 500, 'k', linewidth=0.75)
    ax1.vlines(0, 0, 500, 'k', linewidth=0.75)
    ax0.set_title('gaze-shifting'); ax1.set_title('compensatory')
    ax0.set_ylabel('eye movement'); ax1.set_ylabel('eye movement')
    ax0.set_xlabel('time (ms)'); ax1.set_xlabel('time (ms)')
    ax0.set_yticks(np.linspace(0, 500, 3)); ax1.set_yticks(np.linspace(0, 500, 3))
    ax0.set_xticks(np.linspace(-.5, .5, 5), labels=np.linspace(-500, 500, 5).astype(int))
    ax1.set_xticks(np.linspace(-.5, .5, 5), labels=np.linspace(-500, 500, 5).astype(int))

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, '0_gaze_comp_rasters_2.pdf'))

    
    fig, [ax0, ax1] = plt.subplots(1,2, figsize=(6,2), dpi=300)
    row = hffm.loc[215].copy()

    ax0.plot(psth_bins, row['FmLt_gazeshift_right_saccPSTH_dHead1'], color=colors['gaze'])
    ax0.plot(psth_bins, row['FmLt_comp_right_saccPSTH_dHead1'], color=colors['comp'])

    ax0.set_xlim([-0.5, 0.5])
    ax0.vlines(0, 0, 21, 'k', linewidth=0.75)
    ax0.set_ylabel('sp/sec')
    ax0.set_xlabel('time (ms)')
    ax0.set_xticks(np.linspace(-.5, .5, 5), labels=np.linspace(-500, 500, 5).astype(int))

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, '0_gaze_comp_rasters_3.pdf'))