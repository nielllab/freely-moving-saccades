

import os
import cv2
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

import fmEphys as fme
import saccadeAnalysis as sacc


def fig4(hffm, savepath):

    sacc.set_plt_params()
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']


    # head-fixed tepmoral sequences
    sorted_df = hffm[['FmLt_gazeshift_peakT','gazecluster',
                      'pref_gazeshift_psth','nonpref_gazeshift_psth','pref_comp_psth',
                        'nonpref_comp_psth','norm_Rc_psth','norm_Sn_psth',
                        'tf_pref_cps','sf_pref_cpd',
                        'Sn_responsive','Rc_responsive']].copy()

    sorted_df = sorted_df[sorted_df['gazecluster']!='unresponsive']
    sorted_df.sort_values(by='FmLt_gazeshift_peakT', inplace=True)
    sorted_df = sorted_df.reset_index()
    tseq_rc_gaze = fme.flatten_series(sorted_df['pref_gazeshift_psth'][sorted_df['Rc_responsive']].copy())
    tseq_sn_gaze = fme.flatten_series(sorted_df['pref_gazeshift_psth'][sorted_df['Sn_responsive']].copy())

    tseq_either_hf = fme.flatten_series(
        sorted_df['pref_gazeshift_psth'][sorted_df['Sn_responsive'] | sorted_df['Rc_responsive']].copy()
        )
    tseq_rc = fme.flatten_series(sorted_df['norm_Rc_psth'][sorted_df['Rc_responsive']].copy())
    tseq_sn = fme.flatten_series(sorted_df['norm_Sn_psth'][sorted_df['Sn_responsive']].copy())

    ex_units = [413, 415, 456]
    ex_units_direcprefs = ['left','left','left']
    for ind in ex_units:
        print(hffm['session'].iloc[ind])

    ind = 413
    path = hffm['original_session_path'].iloc[ind]
    worldpath = fme.find('*revchecker*world.nc', path)[0]
    ephyspath = fme.find('*revchecker*ephys_props.h5', path)[0]
    origephys = pd.read_hdf(ephyspath)
    ephysT0 = origephys['t0'].iloc[0]
    worldxr = xr.open_dataset(worldpath)
    vid = worldxr.WORLD_video.values.astype(np.uint8)
    worldT = worldxr.timestamps.values
    eyeT = hffm['Rc_eyeT'].iloc[ind].copy()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    num_frames = np.size(vid, 0); vid_width = np.size(vid, 1); vid_height = np.size(vid, 2)
    kmeans_input = vid.reshape(num_frames, vid_width*vid_height)
    _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_diff = np.diff(np.ndarray.flatten(labels))
    stim_state = interp1d(worldT[:-1]-ephysT0, label_diff, bounds_error=False)(eyeT)
    eventT = eyeT[np.where((stim_state<-0.1)+(stim_state>0.1))]

    fig4A = plt.figure(constrained_layout=True, figsize=(5,4), dpi=300)
    fig4Aspec = gridspec.GridSpec(nrows=3, ncols=3, figure=fig4A, wspace=0.07, hspace=0.15)

    ax_pos_rasterG = fig4A.add_subplot(fig4Aspec[0,0])
    ax_biph_rasterG = fig4A.add_subplot(fig4Aspec[0,1])
    ax_neg_rasterG = fig4A.add_subplot(fig4Aspec[0,2])

    ax_pos_rasterR = fig4A.add_subplot(fig4Aspec[1,0])
    ax_biph_rasterR = fig4A.add_subplot(fig4Aspec[1,1])
    ax_neg_rasterR = fig4A.add_subplot(fig4Aspec[1,2])

    ax_pos_psth = fig4A.add_subplot(fig4Aspec[2,0])
    ax_biph_psth = fig4A.add_subplot(fig4Aspec[2,1])
    ax_neg_psth = fig4A.add_subplot(fig4Aspec[2,2])

    num_movements = 247
    raster_panelsG = [ax_pos_rasterG, ax_biph_rasterG, ax_neg_rasterG]
    raster_panelsR = [ax_pos_rasterR, ax_biph_rasterR, ax_neg_rasterR]
    sdf_panels = [ax_pos_psth, ax_biph_psth, ax_neg_psth]
    for i, u in enumerate(ex_units):
        row = hffm.iloc[u]
        rasterG = raster_panelsG[i]
        rasterR = raster_panelsR[i]
        sdf_panel = sdf_panels[i]
        LR = ex_units_direcprefs[i]
        
        rasterG.set_title(['positive','biphasic','negative'][i])

        gazeshifts = row['FmLt_gazeshift_{}_saccTimes_dHead'.format(LR)].copy()
        
        plot_gs = np.random.choice(gazeshifts, size=num_movements, replace=False)
        plot_rc = np.random.choice(eventT.copy(), size=num_movements, replace=False)

        for n, s in enumerate(plot_gs):
            sp = row['FmLt_spikeT']-s
            sp = sp[np.abs(sp)<=0.5]
            rasterG.plot(sp, np.ones(sp.size)*n, '|', color=colors['gaze'], markersize=.25)

        for n, s in enumerate(plot_rc):
            sp = row['Rc_spikeT']-s
            sp = sp[np.abs(sp)<=0.5]
            rasterR.plot(sp, np.ones(sp.size)*(n), '|', color='k', markersize=.25)
        
        rasterG.set_ylim([num_movements, 0])
        rasterR.set_ylim([num_movements, 0])
        rasterG.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
        rasterR.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
        if i == 0:
            rasterG.set_ylabel('gaze shifts'); rasterR.set_ylabel('checkerboard')
            rasterG.set_yticks(np.linspace(0, 250, 3))
            rasterR.set_yticks(np.linspace(0, 250, 3))
        else:
            rasterG.set_yticks(np.linspace(0, 250, 3),labels=[])
            rasterR.set_yticks(np.linspace(0, 250, 3),labels=[])
        rasterG.set_xticks([]); rasterR.set_xticks([])
        rasterG.set_xlim([-.5,.5]); rasterR.set_xlim([-.5,.5])
        rasterG.axes.spines.bottom.set_visible(False); rasterR.axes.spines.bottom.set_visible(False)
        
        sdf_panel.plot(psth_bins, row['Rc_psth'], color='k')
        sdf_panel.plot(psth_bins, row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(LR)], color=colors['gaze'])
        max_fr = np.nanmax(np.hstack([row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(LR)], row['Rc_psth']]))*1.1
        sdf_panel.set_ylim([0,max_fr])
        sdf_panel.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
        sdf_panel.set_xlim([-.5,.5])
        if i == 0:
            sdf_panel.set_ylabel('sp/sec')
        sdf_panel.set_xlabel('msec')
        sdf_panel.vlines(0, 0, max_fr, color='k', linewidth=1, linestyle='dashed')
        
    fig4A.savefig(os.path.join(savepath, '4_demo_units.pdf'))


    
    fig4B = plt.figure(constrained_layout=True, figsize=(9,2.5), dpi=300)
    fig4Bspec = gridspec.GridSpec(nrows=1, ncols=3, figure=fig4B, wspace=0.01, hspace=0)

    ax_light_clusters_gaze = fig4B.add_subplot(fig4Bspec[:,0])
    ax_light_clusters = fig4B.add_subplot(fig4Bspec[:,1])
    ax_dark_clusters = fig4B.add_subplot(fig4Bspec[:,2])

    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        data = hffm[hffm['gazecluster']==name]
        cluster_psths = fme.flatten_series(data['pref_gazeshift_psth'])
        cluster_psths = cluster_psths[~np.isnan(cluster_psths).any(axis=1)]
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_light_clusters_gaze.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_light_clusters_gaze.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2) 
    ax_light_clusters_gaze.set_xlim([-0.2,0.4]); ax_light_clusters_gaze.set_ylim([-.6,.6])
    ax_light_clusters_gaze.set_ylabel('norm. spike rate'); ax_light_clusters.set_xlabel('msec')
    ax_light_clusters_gaze.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_light_clusters_gaze.set_yticks(np.linspace(-0.5,0.5,3))
    ax_light_clusters_gaze.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    step = 0.14
    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        data = hffm[hffm['gazecluster']==name][hffm['Rc_responsive']]
        cluster_psths = fme.flatten_series(data['norm_Rc_psth'])
        cluster_psths = cluster_psths[~np.isnan(cluster_psths).any(axis=1)]
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_light_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_light_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2) 
    ax_light_clusters.set_xlim([-0.2,0.4]); ax_light_clusters.set_ylim([-.6,.6])
    # ax_dark_clusters.annotate('early', xy=[0.3,-0.22], color=colors['early'], fontsize=11)
    # ax_dark_clusters.annotate('late', xy=[0.3,-0.22-(step*1)], color=colors['late'], fontsize=11)
    # ax_dark_clusters.annotate('biphasic', xy=[0.3,-0.22-(step*2)], color=colors['biphasic'], fontsize=11)
    # ax_dark_clusters.annotate('negative', xy=[0.3,-0.22-(step*3)], color=colors['negative'], fontsize=11)
    ax_light_clusters.set_ylabel('norm. spike rate'); ax_light_clusters.set_xlabel('msec')
    # ax_light_clusters.set_title('Light')
    ax_light_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_light_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_light_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    for count, name in enumerate(names):
        data = hffm[hffm['gazecluster']==name][hffm['Sn_responsive']]
        cluster_psths = fme.flatten_series(data['norm_Sn_psth'])
        cluster_psths = cluster_psths[~np.isnan(cluster_psths).any(axis=1)]
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_dark_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_dark_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3) 
    ax_dark_clusters.set_xlim([-0.2,0.4]); ax_dark_clusters.set_ylim([-.6,.6]); ax_dark_clusters.set_xlabel('msec')
    # ax_dark_clusters.set_title('Dark')
    ax_dark_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_dark_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_dark_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax_dark_clusters.vlines(.25,-1,1,color='k',linestyle='dashed',linewidth=1)

    fig4B.savefig(os.path.join(savepath, '4_clustering_withUnresp.pdf'))


    # temporal sequences
    fig3C = plt.figure(constrained_layout=True, figsize=(9,4), dpi=300)
    fig3Cspec = gridspec.GridSpec(nrows=1, ncols=4, figure=fig3C, wspace=0, hspace=0.5)

    ax_tseq_gaze1 = fig3C.add_subplot(fig3Cspec[:,0])
    ax_tseq_rc = fig3C.add_subplot(fig3Cspec[:,1])
    ax_tseq_gaze2 = fig3C.add_subplot(fig3Cspec[:,2])
    ax_tseq_sn = fig3C.add_subplot(fig3Cspec[:,3])

    tseq_aspect = 2.8

    sz = np.size(fme.drop_nan_along(tseq_either_hf),0)
    sacc.plot_PSTH_heatmap(ax_tseq_gaze1, fme.drop_nan_along(tseq_either_hf))
    ax_tseq_gaze1.set_aspect(tseq_aspect)
    ax_tseq_gaze1.set_ylabel('cell')
    ax_tseq_gaze1.set_title('gaze-shifting')
    ax_tseq_gaze1.set_yticks(np.arange(0,sz,100))

    sz = np.size(fme.drop_nan_along(tseq_rc),0)
    sacc.plot_tempseq(ax_tseq_rc, fme.drop_nan_along(tseq_rc))
    ax_tseq_rc.set_aspect(tseq_aspect)
    ax_tseq_rc.set_title('checkerboard')
    ax_tseq_rc.set_yticks(np.arange(0,sz,100), labels=[])

    sz = np.size(fme.drop_nan_along(tseq_either_hf),0)
    sacc.plot_tempseq(ax_tseq_gaze2, fme.drop_nan_along(tseq_either_hf))
    ax_tseq_gaze2.set_aspect(tseq_aspect)
    ax_tseq_gaze2.set_title('gaze-shifting')
    ax_tseq_gaze2.set_ylabel('cell')
    ax_tseq_gaze2.set_yticks(np.arange(0,sz,100))

    sz = np.size(fme.drop_nan_along(tseq_sn),0)
    sacc.plot_tempseq(ax_tseq_sn, fme.drop_nan_along(tseq_sn))
    ax_tseq_sn.set_aspect(tseq_aspect)
    ax_tseq_sn.set_title('sparse noise')
    ax_tseq_sn.set_yticks(np.arange(0,sz,100), labels=[])
    ax_tseq_sn.vlines(1250, 0, sz, linestyle='dashed', linewidth=1, color='k')

    fig3C.savefig(os.path.join(savepath, '4_temp_seq-either_for_gaze.pdf'))

    # temporal sequence cluseter legend
    tseq_legend_col = sorted_df['gazecluster'][sorted_df['Sn_responsive'] | sorted_df['Rc_responsive']].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = mpl.colors.to_rgba(colors[n])

    fig, ax = plt.subplots(1,1,figsize=(0.5,2), dpi=300)
    ax.imshow(tseq_legend, aspect=.05)
    ax.set_yticks([]); ax.set_xticks([])
    ax.axes.spines.bottom.set_visible(False)
    ax.axes.spines.right.set_visible(False)
    ax.axes.spines.left.set_visible(False)
    ax.axes.spines.top.set_visible(False)
    fig.savefig(os.path.join(savepath, '4_temp_seq_legend_either_hf.pdf'))


    tseq_legend_col = sorted_df['gazecluster'][sorted_df['Rc_responsive']].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = mpl.colors.to_rgba(colors[n])

    fig, ax = plt.subplots(1,1,figsize=(0.5,2), dpi=300)
    ax.imshow(tseq_legend, aspect=.05)
    ax.set_yticks([]); ax.set_xticks([])
    ax.axes.spines.bottom.set_visible(False)
    ax.axes.spines.right.set_visible(False)
    ax.axes.spines.left.set_visible(False)
    ax.axes.spines.top.set_visible(False)
    fig.savefig(os.path.join(savepath, '4_temp_seq_legend_Rc.pdf'))

    hffm['gaze_rc_maxcc'] = np.nan
    hffm['gaze_sn_maxcc'] = np.nan
    for ind, row in hffm[['norm_Rc_psth','norm_Sn_psth',
                          'pref_gazeshift_psth','Rc_responsive',
                          'Sn_responsive']][hffm['gazecluster']!='unresponsive'].iterrows():
        if row['Rc_responsive']:
            r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Rc_psth'].astype(float)[1000:1250])
            hffm.at[ind, 'gaze_rc_maxcc'] = r[0,1]
        if row['Sn_responsive']:
            r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Sn_psth'].astype(float)[1000:1250])
            hffm.at[ind, 'gaze_sn_maxcc'] = r[0,1]

    rc_vals = hffm['gaze_rc_maxcc'][hffm['Rc_responsive']][hffm['gazecluster']!='unresponsive'].copy().dropna().reset_index(drop=True).to_numpy().astype(float)
    sn_vals = hffm['gaze_sn_maxcc'][hffm['Sn_responsive']][hffm['gazecluster']!='unresponsive'].copy().dropna().reset_index(drop=True).to_numpy().astype(float)


    fig, [ax0,ax1] = plt.subplots(2,1,figsize=(2.5,3.2), dpi=300)

    Rc_weights = np.ones_like(rc_vals) / float(len(rc_vals))
    n,_,_ = ax0.hist(rc_vals, color='grey', bins=np.linspace(-1,1,21), alpha=0.8, weights=Rc_weights)
    # ax0.set_xlabel('gaze shift cc');
    ax0.set_ylabel('frac. cells')
    # ax0.set_xticks(np.arange(-1,1,3))#,labels=[])
    ax0.plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    ax0.set_ylim([0,.22])
    ax0.set_xlim([-1,1])

    Sn_weights = np.ones_like(sn_vals) / float(len(sn_vals))
    n,_,_ = ax1.hist(sn_vals, color='grey', bins=np.linspace(-1,1,21), alpha=0.8, weights=Sn_weights)
    ax1.set_xlabel('gaze shift cc')
    ax1.set_ylabel('frac. cells')
    ax1.plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    ax1.set_ylim([0,.22])
    ax1.set_xlim([-1,1])

    ax0.set_title('checkerboard')
    ax1.set_title('sparse noise')

    ax0.plot(np.ones(2)*hffm.loc[413,'gaze_rc_maxcc'].copy(),
             np.ones(2)*0.2, marker='v', markersize=2, color='tab:blue')
    ax0.plot(np.ones(2)*hffm.loc[415,'gaze_rc_maxcc'].copy(),
             np.ones(2)*0.2, marker='v', markersize=2, color='tab:orange')
    ax0.plot(np.ones(2)*hffm.loc[456,'gaze_rc_maxcc'].copy(),
             np.ones(2)*0.2, marker='v', markersize=2, color='tab:green')

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, 'rc_sn_gaze_corr_marked.pdf'))

    
    # 
    _cols2iter = hffm[['gazecluster','FmLt_gazeshift_peakT','rc_peakT','sn_peakT',
            'Rc_responsive','Sn_responsive']][hffm['gazecluster']!='unresponsive']

    fig, axs = plt.subplots(2,2,figsize=(5,4.5), dpi=300)
    for ind, row in _cols2iter.iterrows():
        if row['Rc_responsive']==True:
            axs[0,0].plot(row['FmLt_gazeshift_peakT'], row['rc_peakT'],
                          '.', color=colors[row['gazecluster']], markersize=2)
        if row['Sn_responsive']==True:
            axs[0,1].plot(row['FmLt_gazeshift_peakT'], row['sn_peakT'],
                          '.', color=colors[row['gazecluster']], markersize=2)

    axs[0,0].plot([0,.25], [0,.25], color='k', linewidth=1, linestyle='dashed')
    axs[0,1].plot([0,.25], [0,.25], color='k', linewidth=1, linestyle='dashed')
            
    axs[0,0].set_xlim([.02,.200]); axs[0,0].set_ylim([.02,.200])
    axs[0,1].set_xlim([.02,.200]); axs[0,1].set_ylim([.02,.200])
    axs[0,0].set_xticks(np.linspace(.02, 0.200, 4), labels=np.linspace(20, 200, 4).astype(int))
    axs[0,0].set_yticks(np.linspace(.02, 0.200, 4), labels=np.linspace(20, 200, 4).astype(int))
    axs[0,1].set_xticks(np.linspace(.02, 0.200, 4), labels=np.linspace(20, 200, 4).astype(int))
    axs[0,1].set_yticks(np.linspace(.02, 0.200, 4), labels=np.linspace(20, 200, 4).astype(int))
    axs[0,0].set_ylabel('latency (msec)'); axs[0,0].set_xlabel('gaze shift latency (msec)')
    axs[0,1].set_ylabel('latency (msec)'); axs[0,1].set_xlabel('gaze shift latency (msec)')
    axs[0,0].set_title('checkerboard')
    axs[0,1].set_title('sparse noise')

    # use_Rc = hffm[hffm['FmLt_gazeshift_peakT']>.035][hffm['FmLt_gazeshift_peakT']<.2][hffm['rc_peakT']>.035][hffm['rc_peakT']<.2][hffm['gazecluster']!='unresponsive'][hffm['Rc_responsive']==True].copy()
    # use_Sn = hffm[hffm['FmLt_gazeshift_peakT']>.035][hffm['FmLt_gazeshift_peakT']<.2][hffm['rc_peakT']>.035][hffm['rc_peakT']<.2][hffm['gazecluster']!='unresponsive'][hffm['Sn_responsive']==True].copy()

    # rc_R = plot_linregress(axs[0,0], use_Rc['FmLt_gazeshift_peakT'], use_Rc['rc_peakT'])
    # sn_R = plot_linregress(axs[0,1], use_Sn['FmLt_gazeshift_peakT'], use_Sn['sn_peakT'])

    # print('Rc:{}; Sn:{}'.format(rc_R.rvalue, sn_R.rvalue))

    #, 'k--', linewidth=1)#, linestyle='dashed')
    # axs[0,1].plot([.03,.03], [.25,.25], 'k--', linewidth=1)#, linestyle='dashed')

    Rc_weights = np.ones_like(rc_vals) / float(len(rc_vals))
    n,_,_ = axs[1,0].hist(rc_vals, color='grey', bins=np.linspace(-1,1,21), alpha=0.8, weights=Rc_weights)
    axs[1,0].set_xlabel('gaze shift cc'); axs[1,0].set_ylabel('frac. cells')
    axs[1,0].plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    axs[1,0].set_ylim([0,.22])

    Sn_weights = np.ones_like(sn_vals) / float(len(sn_vals))
    n,_,_ = axs[1,1].hist(sn_vals, color='grey', bins=np.linspace(-1,1,21), alpha=0.8, weights=Sn_weights)
    axs[1,1].set_xlabel('gaze shift cc'); axs[1,1].set_ylabel('frac. cells')
    axs[1,1].plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    axs[1,1].set_ylim([0,.22])

    axs[1,0].set_title('checkerboard')
    axs[1,1].set_title('sparse noise')

    plt.tight_layout()
    fig.savefig(os.path.join(savepath, '4_props_cc.pdf'))




