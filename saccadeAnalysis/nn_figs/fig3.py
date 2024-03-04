"""
saccadeAnalysis/nn_figs/fig3.py

Written by DMM, 2022
Last modified March 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import saccadeAnalysis as sacc
import fmEphys as fme



def fig3(ltdk, ltdk_dict, savepath):
    """ Make subpanels on light/dark data.

    Parameters
    ----------
    ltdk : pd.DataFrame
        Light/dark dataset, where columns are properties and
        each index is a cell.
    ltdk_dict : dict
        Dictionary of light/dark data (returned from `sacc.make_ltdk_dataset()` as
        the dictionary `out`) containing key:val pairs for data summarized across
        cells.
    savepath : str
        Path to save the figures.
    """

    sacc.set_plt_params()

    props = sacc.propsdict()

    colors = props['colors']
    psth_bins = props['psth_bins']

    tseq_light1 = ltdk_dict['tseq_pref_light_by_light_w_unresp']
    tseq_dark_pref1 = ltdk_dict['tseq_pref_dark_by_light_w_unresp']
    tseq_dark_nonpref1 = ltdk_dict['tseq_nonpref_dark_by_light_w_unresp']
    tseq_dark_comp1 = ltdk_dict['tseq_comp_dark_by_light_w_unresp']
    tseq_legend1 = ltdk_dict['tseq_legend_w_unresp']

    tseq_aspect = 2.8

    # plot temporal sequences
    fig, [ax0, ax1, ax2, ax3] = plt.subplots(1,4,figsize=(10,5), dpi=300)

    sacc.plot_PSTH_heatmap(ax0, tseq_light1)
    ax0.set_aspect(tseq_aspect)
    ax0.set_title('Light gaze-shift')
    ax0.set_ylabel('cells')
    # ax0.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    sacc.plot_PSTH_heatmap(ax1, tseq_dark_pref1)
    ax1.set_aspect(tseq_aspect)
    ax1.set_title('Dark pref')
    ax1.set_yticklabels([])
    # ax1.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    sacc.plot_PSTH_heatmap(ax2, tseq_dark_nonpref1)
    ax2.set_aspect(tseq_aspect)
    ax2.set_title('Dark nonpref')
    ax2.set_yticklabels([])
    # ax2.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    sacc.plot_PSTH_heatmap(ax3, tseq_dark_comp1)
    ax3.set_aspect(tseq_aspect)
    ax3.set_title('Dark comp')
    ax3.set_yticklabels([])
    # ax3.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    fig.savefig(os.path.join(savepath, '3_temporal_seq.pdf'))

    # light dark clustering fig
    fig3A = plt.figure(constrained_layout=True, figsize=(7,3.5), dpi=300)
    fig3Aspec = gridspec.GridSpec(nrows=2, ncols=4, figure=fig3A, wspace=0.07, hspace=0.12)

    ax_early_l = fig3A.add_subplot(fig3Aspec[0,0])
    ax_late_l = fig3A.add_subplot(fig3Aspec[0,1])
    ax_biphasic_l = fig3A.add_subplot(fig3Aspec[0,2])
    ax_negative_l = fig3A.add_subplot(fig3Aspec[0,3])

    ax_early_d = fig3A.add_subplot(fig3Aspec[1,0])
    ax_late_d = fig3A.add_subplot(fig3Aspec[1,1])
    ax_biphasic_d = fig3A.add_subplot(fig3Aspec[1,2])
    ax_negative_d = fig3A.add_subplot(fig3Aspec[1,3])

    names = ['early','late','biphasic','negative']
    light_panels = [ax_early_l, ax_late_l, ax_biphasic_l, ax_negative_l]
    dark_panels = [ax_early_d, ax_late_d, ax_biphasic_d, ax_negative_d]

    for count, name in enumerate(names):
        lpanel = light_panels[count]; dpanel = dark_panels[count]
        
        for x in ltdk['pref_gazeshift_psth'][ltdk['gazecluster']==name]:
            lpanel.plot(psth_bins, x, '-', linewidth=1, alpha=.3)
        lpanel.plot(psth_bins, np.nanmean(
            fme.flatten_series(ltdk['pref_gazeshift_psth'][ltdk['gazecluster']==name][ltdk['gazeshift_responsive']==True]),0),
            '-', linewidth=3, color=colors[name])
        lpanel.set_xlim([-0.2,0.4]); lpanel.set_ylim([-1,1])
    #     lpanel.set_title(name.capitalize())
        lpanel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
        lpanel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        
        for x in ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name]:
            dpanel.plot(psth_bins, x, '-', linewidth=1, alpha=.3)
        dpanel.plot(
            psth_bins, np.nanmean(fme.flatten_series(ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name][ltdk['gazeshift_responsive']==True]),0),
            '-', linewidth=3, color=colors[name])
        dpanel.set_xlim([-0.2,0.4]); dpanel.set_ylim([-1,1])
        dpanel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
        dpanel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        
        if name=='early':
            lpanel.set_ylabel('norm. spike rate')
            # dpanel.set_ylabel('norm. spike rate')
        if name!='early':
            lpanel.set_yticklabels([])
            dpanel.set_yticklabels([])
        lpanel.set_xticklabels([])
        dpanel.set_xlabel('time (msec)')
        lpanel.set_title(name)
            
    fig3A.savefig(os.path.join(savepath, '3_clustering.pdf'))



    # light/dark summary
    fig3B = plt.figure(constrained_layout=True, figsize=(5.5,2.5), dpi=300)
    fig3Bspec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig3B, wspace=0.01, hspace=0)

    ax_light_clusters = fig3B.add_subplot(fig3Bspec[:,0])
    ax_dark_clusters = fig3B.add_subplot(fig3Bspec[:,1])

    step = 0.14
    names = ['early','late','biphasic','negative']

    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = fme.flatten_series(data['pref_gazeshift_psth'])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_light_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_light_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr,
                                       color=colors[name], alpha=0.2) 
    ax_light_clusters.set_xlim([-0.2,0.4]); ax_light_clusters.set_ylim([-.6,.6])
    # ax_dark_clusters.annotate('early', xy=[0.3,-0.22], color=colors['early'], fontsize=11)
    # ax_dark_clusters.annotate('late', xy=[0.3,-0.22-(step*1)], color=colors['late'], fontsize=11)
    # ax_dark_clusters.annotate('biphasic', xy=[0.3,-0.22-(step*2)], color=colors['biphasic'], fontsize=11)
    # ax_dark_clusters.annotate('negative', xy=[0.3,-0.22-(step*3)], color=colors['negative'], fontsize=11)
    ax_light_clusters.set_ylabel('norm. spike rate'); ax_light_clusters.set_xlabel('time (msec)')
    ax_light_clusters.set_title('Light gaze shift')
    ax_light_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_light_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_light_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = fme.flatten_series(data['pref_dark_gazeshift_psth'])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_dark_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_dark_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3) 
    ax_dark_clusters.set_xlim([-0.2,0.4]); ax_dark_clusters.set_ylim([-.6,.6]); ax_dark_clusters.set_xlabel('time (msec)')
    ax_dark_clusters.set_title('Dark gaze shift')
    ax_dark_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_dark_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_dark_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    fig3B.savefig(os.path.join(savepath, '3_cluster_summary_noUnresp.pdf'))


    fig3B = plt.figure(constrained_layout=True, figsize=(5.5,2.5), dpi=300)
    fig3Bspec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig3B, wspace=0.01, hspace=0)

    ax_light_clusters = fig3B.add_subplot(fig3Bspec[:,0])
    ax_dark_clusters = fig3B.add_subplot(fig3Bspec[:,1])

    step = 0.14
    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = fme.flatten_series(data['pref_comp_psth'][ltdk['movement']==False])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_light_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_light_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2) 
    ax_light_clusters.set_xlim([-0.2,0.4]); ax_light_clusters.set_ylim([-.6,.6])
    # ax_dark_clusters.annotate('early', xy=[0.3,-0.22], color=colors['early'], fontsize=11)
    # ax_dark_clusters.annotate('late', xy=[0.3,-0.22-(step*1)], color=colors['late'], fontsize=11)
    # ax_dark_clusters.annotate('biphasic', xy=[0.3,-0.22-(step*2)], color=colors['biphasic'], fontsize=11)
    # ax_dark_clusters.annotate('negative', xy=[0.3,-0.22-(step*3)], color=colors['negative'], fontsize=11)
    ax_light_clusters.set_ylabel('norm. spike rate');# ax_light_clusters.set_xlabel('time (msec)')
    ax_light_clusters.set_title('light compensatory')
    # ax_light_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_light_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=[])
    ax_light_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_light_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = fme.flatten_series(data['pref_dark_comp_psth'][ltdk['movement']==False])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_dark_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_dark_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3) 
    ax_dark_clusters.set_xlim([-0.2,0.4]); ax_dark_clusters.set_ylim([-.6,.6]); ax_dark_clusters.set_xlabel('time (msec)')
    ax_dark_clusters.set_title('dark compensatory')
    ax_dark_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_dark_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_dark_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    fig3B.savefig(os.path.join(savepath, '3_comp_summary.pdf'))


    fig, ax = plt.subplots(1,1,figsize=(0.5,2), dpi=300)
    ax.imshow(tseq_legend1, aspect=.05)
    ax.set_yticks([]); ax.set_xticks([])
    ax.axes.spines.bottom.set_visible(False)
    ax.axes.spines.right.set_visible(False)
    ax.axes.spines.left.set_visible(False)
    ax.axes.spines.top.set_visible(False)
    fig.savefig(os.path.join(savepath, '3_temp_seq_legend.pdf'))



    names = ['early','late','biphasic','negative']

    fig3E = plt.figure(constrained_layout=True, figsize=(5,6), dpi=300)
    fig3Espec = gridspec.GridSpec(nrows=4, ncols=6, figure=fig3E, wspace=.3, hspace=0.5)

    ax_stationary_fr = fig3E.add_subplot(fig3Espec[1:3,0:3])
    ax_active_fr = fig3E.add_subplot(fig3Espec[1:3,3:6])

    for name in names:
        light_iafr = ltdk['FmLt_inactive_fr'][ltdk['gazecluster']==name].to_numpy()
        dark_iafr = ltdk['FmDk_inactive_fr'][ltdk['gazecluster']==name].to_numpy()
        
        light_afr = ltdk['FmLt_active_fr'][ltdk['gazecluster']==name].to_numpy()
        dark_afr = ltdk['FmDk_active_fr'][ltdk['gazecluster']==name].to_numpy()
        
        for i in range(np.size(light_iafr,0)):
            ax_active_fr.plot(dark_afr[i], light_afr[i], '.', color=colors[name], markersize=3)
            ax_stationary_fr.plot(dark_iafr[i], light_iafr[i], '.', color=colors[name], markersize=3)
        
    fr_axis_max = 21
    ax_active_fr.axis('square')
    ax_stationary_fr.axis('square')
    ax_stationary_fr.set_ylim([0,fr_axis_max]); ax_active_fr.set_ylim([0,fr_axis_max])
    ax_stationary_fr.set_xlim([0,fr_axis_max]); ax_active_fr.set_xlim([0,fr_axis_max])
    ax_stationary_fr.plot([0,fr_axis_max], [0,fr_axis_max], color='k',linestyle='dashed')
    ax_active_fr.plot([0,fr_axis_max], [0,fr_axis_max], color='k',linestyle='dashed')
    ax_stationary_fr.set_xlabel('dark (sp/sec)'); ax_stationary_fr.set_ylabel('light (sp/sec)')
    ax_active_fr.set_title('Active')
    ax_active_fr.set_xlabel('dark (sp/sec)'); ax_active_fr.set_ylabel('light (sp/sec)')
    ax_stationary_fr.set_title('Stationary')
    ax_active_fr.set_xticks(np.arange(0,fr_axis_max,5))
    ax_stationary_fr.set_yticks(np.arange(0,fr_axis_max,5))
    ax_active_fr.set_xticks(np.arange(0,fr_axis_max,5))
    ax_stationary_fr.set_yticks(np.arange(0,fr_axis_max,5))

    fig3E.savefig(os.path.join(savepath, '3_ltdk_firing_rates.pdf'))

