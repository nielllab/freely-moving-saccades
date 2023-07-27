"""

Written by DMM, 2022
"""


import os
from tqdm import tqdm
import cv2
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

import fmEphys as fme
import saccadeAnalysis as sacc


def main(dataset_dict):
    """
    
    Parameters
    ----------
    dataset_dict : dict
        A dictionary of paths to datasets saved as h5 files.
        
    """

    # Set plt parameters
    sacc.set_plt_params()

    # Get plotting properties
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']
    psth_bins_long = props['psth_bins_long']

    hffm_path = dataset_dict['hffm']
    ltdk_path = dataset_dict['ltdk']
    savepath = dataset_dict['savepath']

    # load data
    print('Reading HfFm data...')
    hffm = fme.read_group_h5(hffm_path)

    print('Reading LtDk data...')
    ltdk = fme.read_group_h5(ltdk_path)

    # Set up a subdirectory for the figures to be saved to.
    figpath = os.path.join(savepath, 'figures')
    if not os.path.exists(figpath):
        os.mkdir(figpath)





















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

    ax0.plot(np.ones(2)*hffm.loc[413,'gaze_rc_maxcc'].copy(), np.ones(2)*0.2, marker='v', markersize=2, color='tab:blue')
    ax0.plot(np.ones(2)*hffm.loc[415,'gaze_rc_maxcc'].copy(), np.ones(2)*0.2, marker='v', markersize=2, color='tab:orange')
    ax0.plot(np.ones(2)*hffm.loc[456,'gaze_rc_maxcc'].copy(), np.ones(2)*0.2, marker='v', markersize=2, color='tab:green')

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'rc_sn_gaze_corr_marked.pdf'))


    fig, axs = plt.subplots(2,2,figsize=(5,4.5), dpi=300)
    for ind, row in hffm[['gazecluster','FmLt_gazeshift_peakT','rc_peakT','sn_peakT','Rc_responsive','Sn_responsive']][hffm['gazecluster']!='unresponsive'].iterrows():
        if row['Rc_responsive']==True:
            axs[0,0].plot(row['FmLt_gazeshift_peakT'], row['rc_peakT'], '.', color=colors[row['gazecluster']], markersize=2)
        if row['Sn_responsive']==True:
            axs[0,1].plot(row['FmLt_gazeshift_peakT'], row['sn_peakT'], '.', color=colors[row['gazecluster']], markersize=2)

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
    fig.savefig(os.path.join(figpath, '4_props_cc.pdf'))


    grat_bins = np.arange(-1500, 1501)


    hffm['raw_mod_for_Gt'] = np.nan
    hffm['norm_mod_for_Gt'] = np.nan
    hffm['Gt_kde_psth_norm'] = hffm['Gt_kde_psth'].copy()

    for ind, row in hffm.iterrows():
        sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        sp = len(row['Gt_spikeT'])
        hffm.at[ind, 'Gt_fr'] = sp/sec
        
        norm_psth = normalize_gt_psth(row['Gt_kde_psth'])
        hffm.at[ind, 'Gt_kde_psth_norm'] = norm_psth.copy().astype(object)
        
        hffm.at[ind, 'raw_mod_for_Gt'] = gt_modind(row['Gt_kde_psth'])
        
        hffm.at[ind, 'norm_mod_for_Gt'] = gt_modind(norm_psth)
        
    hffm['Gt_responsive'] = False
    for ind, row in hffm.iterrows():
        if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):
            hffm.at[ind, 'Gt_responsive'] = True

    print(hffm['Gt_responsive'].sum())
    print(hffm['Gt_responsive'].sum()/len(hffm.index.values))


    for sf in ['low','mid','high']:
        hffm['norm_ori_tuning_'+sf] = hffm['Gt_ori_tuning_tf'].copy().astype(object)
    for ind, row in hffm.iterrows():
        orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
        for sfnum in range(3):
            sf = ['low','mid','high'][sfnum]
            hffm.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
        mean_for_sf = np.array([np.mean(hffm.at[ind,'norm_ori_tuning_low']), np.mean(hffm.at[ind,'norm_ori_tuning_mid']), np.mean(hffm.at[ind,'norm_ori_tuning_high'])])
        mean_for_sf[mean_for_sf<0] = 0
        hffm.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)

    hffm['osi_for_sf_pref'] = np.nan
    hffm['dsi_for_sf_pref'] = np.nan
    for ind, row in hffm.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            hffm.at[ind, 'osi_for_sf_pref'] = row[(['Gt_osi_low','Gt_osi_mid','Gt_osi_high'][best_sf_pref-1])]
            hffm.at[ind, 'dsi_for_sf_pref'] = row[(['Gt_dsi_low','Gt_dsi_mid','Gt_dsi_high'][best_sf_pref-1])]

    hffm['osi_for_sf_pref'][hffm['osi_for_sf_pref']<0] = 0
    hffm['dsi_for_sf_pref'][hffm['dsi_for_sf_pref']<0] = 0
    for ind, row in hffm.iterrows():
        try:
            mean_for_sf = np.array([np.mean(hffm.at[ind,'norm_ori_tuning_low']), np.mean(hffm.at[ind,'norm_ori_tuning_mid']), np.mean(hffm.at[ind,'norm_ori_tuning_high'])])
            mean_for_sf[mean_for_sf<0] = 0
            hffm.at[ind, 'Gt_evoked_rate'] = np.max(mean_for_sf) - row['Gt_drift_spont']
        except:
            pass

    for ind, row in hffm.iterrows():
        if type(row['Gt_ori_tuning_tf']) != float:
            tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
            tuning = tuning - row['Gt_drift_spont']
            tuning[tuning < 0] = 0
            mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
            hffm.at[ind, 'tf_pref'] = tf_pref

    for ind, row in hffm.iterrows():
        tf = 2 + (6 * (row['tf_pref']-1))
        sf = 0.02 * 4 ** (row['sf_pref']-1)
        hffm.at[ind,'tf_pref_cps'] = tf
        hffm.at[ind,'sf_pref_cpd'] = sf
        hffm.at[ind,'grat_speed_dps'] = tf / sf



    hffm['raw_mod_for_Gt'] = np.nan
    hffm['norm_mod_for_Gt'] = np.nan
    hffm['Gt_kde_psth_norm'] = hffm['Gt_kde_psth'].copy()

    for ind, row in hffm.iterrows():
        sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        sp = len(row['Gt_spikeT'])
        hffm.at[ind, 'Gt_fr'] = sp/sec
        
        norm_psth = normalize_gt_psth(row['Gt_kde_psth'])
        hffm.at[ind, 'Gt_kde_psth_norm'] = norm_psth.copy().astype(object)
        
        hffm.at[ind, 'raw_mod_for_Gt'] = gt_modind(row['Gt_kde_psth'])
        
        hffm.at[ind, 'norm_mod_for_Gt'] = gt_modind(norm_psth)
        
    hffm['Gt_responsive'] = False
    for ind, row in hffm.iterrows():
        if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1) and (np.max(row['Gt_kde_psth_norm'][1500:2500]) > 0.5):
            hffm.at[ind, 'Gt_responsive'] = True

    print(hffm['Gt_responsive'].sum())
    print(hffm['Gt_responsive'].sum()/len(hffm.index.values))


    fig, [[ax0, ax1],[ax2, ax3]] = plt.subplots(2,2, dpi=300, figsize=(7.5/2,3.5))

    panels = [ax0, ax1, ax2, ax3]
    movtypes = ['early','late','biphasic','negative']
    for count, panel in enumerate(panels):
        movtype = movtypes[count]
        thisclust = hffm['Gt_kde_psth_norm'][hffm['gazecluster']==movtype][hffm['Gt_responsive']==True]
        for i, psth in enumerate(thisclust):
            panel.plot(grat_bins, psth, '-', linewidth=1, alpha=0.2)
        clustmean = np.nanmean(flatten_series(thisclust),0)
        panel.plot(grat_bins, clustmean, '-', linewidth=2, color=colors[movtype])
        
        panel.vlines(0,-1.5,1.5, color='k',linestyle='dashed',linewidth=1)
        panel.set_ylim([-0.5,1.2])
        panel.set_xlim([-400,1600])
        # panel.set_xticks(np.arange(-400, 1600, 400))
        panel.set_title(movtype)
        if movtype=='late' or movtype=='negative':
            panel.set_yticklabels([])
        else:
            panel.set_ylabel('norm sp/s')
        if movtype=='biphasic' or movtype=='negative':
            panel.set_xlabel('time (ms)')
        else:
            panel.set_xticklabels([])
    plt.tight_layout()

    fig.savefig(os.path.join(figpath, '5_clusters.pdf'))

    fig, ax = plt.subplots(1,1,figsize=(2.5,2.2), dpi=300)

    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        cluster_psths = flatten_series(hffm['Gt_kde_psth_norm'][hffm['gazecluster']==name][hffm['Gt_responsive']==True])
        cluster_psths = cluster_psths[~np.isnan(cluster_psths).any(axis=1)]
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax.plot(grat_bins, clustmean, '-', linewidth=1.5, color=colors[name])
        ax.fill_between(grat_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2)
    ax.set_ylabel('norm sp/s'); ax.set_xlabel('time (ms)')
    ax.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax.set_ylim([-.1,.75])
    ax.set_xlim([-500,1500])
    ax.set_xlim([-100,200])

    fig.savefig(os.path.join(figpath, '5_grat_onsets.pdf'))

    fig5 = plt.figure(constrained_layout=False, figsize=(7,8), dpi=300)
    fig5spec = gridspec.GridSpec(nrows=3, ncols=2, figure=fig5, wspace=.5, hspace=.5)

    fig5Cspec = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=fig5spec[1:3,0:2], wspace=.5, hspace=.5)
    ax_tf_cat = fig5.add_subplot(fig5Cspec[0,0])
    ax_tf_scatter = fig5.add_subplot(fig5Cspec[1,0])
    ax_sf_cat = fig5.add_subplot(fig5Cspec[0,1])
    ax_sf_scatter = fig5.add_subplot(fig5Cspec[1,1])

    use_data = hffm[hffm['gazecluster']!='unresponsive'][hffm['Gt_responsive']==True].copy()
    # [hffm['FmLt_gazeshift_peakT']>0.035][hffm['FmLt_gazeshift_peakT']<.2]

    plot_cprop_scatter(ax_tf_cat, use_data, 'tf_pref_cps', use_median=True)
    ax_tf_cat.set_ylabel('temporal freq (cps)')
    ax_tf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    # ax_tf_cat.set_ylim([0, 8.05])

    plot_cprop_scatter(ax_sf_cat, use_data, 'sf_pref_cpd', use_median=True)
    ax_sf_cat.set_ylabel('spatial freq (cpd)')
    ax_sf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    ax_sf_cat.set_ylim([0, 0.25])

    ###
    names = ['early','late','biphasic','negative']
    for i, name in enumerate(names):
        # cluster = hffm[hffm['movcluster']==name][hffm['responsive_to_gratings']==True][hffm['Fm_fr']>2][hffm['Gt_fr']>2]
        cluster = use_data[use_data['gazecluster']==name][hffm['FmLt_gazeshift_peakT']<.2].copy()
        ax_tf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['tf_pref_cps'], '.', color=colors[name], markersize=3)
        ax_sf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['sf_pref_cpd'], '.', color=colors[name], markersize=3)
        #[cluster['fr']>2]
    # running_median(ax_tf_scatter, hffm['FmLt_gazeshift_peakT'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True], hffm['tf_pref_cps'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True])
    # running_median(ax_sf_scatter, hffm['FmLt_gazeshift_peakT'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True], hffm['sf_pref_cpd'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True])

    running_median(ax_tf_scatter, use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2], use_data['tf_pref_cps'][hffm['FmLt_gazeshift_peakT']<.2], n_bins=7)
    running_median(ax_sf_scatter, use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2], use_data['sf_pref_cpd'][hffm['FmLt_gazeshift_peakT']<.2], n_bins=7)

    # ax_tf_scatter.set_yscale('log')
    # ax_sf_scatter.set_yscale('log')

    # tf_R = plot_linregress(ax_tf_scatter, use_data['FmLt_gazeshift_peakT'], use_data['tf_pref_cps'])
    # sf_R = plot_linregress(ax_sf_scatter, use_data['FmLt_gazeshift_peakT'], use_data['sf_pref_cpd'])

    # ax_tf_scatter.set_ylim([0, 8.05])
    ax_tf_scatter.set_xlim([.020,.205])
    ax_sf_scatter.set_xlim([.020,.205])
    ax_tf_scatter.set_xticks(np.linspace(.025, 0.200, 6), labels=np.linspace(25, 200, 6).astype(int))
    ax_sf_scatter.set_xticks(np.linspace(.025, 0.200, 6), labels=np.linspace(25, 200, 6).astype(int))

    # ax_tf_scatter.set_xticks(np.linspace(.035, 0.2, 4), labels=np.linspace(35, 200, 4).astype(int))
    ax_tf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_tf_scatter.set_ylabel('temporal freq (cps)')

    # ax_sf_scatter.set_xticks(np.linspace(.035, 0.2, 4), labels=np.linspace(35, 200, 4).astype(int))
    ax_sf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_sf_scatter.set_ylabel('spatial freq (cpd)')
    ax_sf_scatter.set_ylim([0,0.25])

    fig5.savefig(os.path.join(figpath, '5_props.pdf'))




    use_hf = hffm[hffm['gazecluster']!='unresponsive'][hffm['Gt_responsive']==True].copy()
    tf_responses = np.zeros([len(use_hf.index.values),4])*np.nan
    for i, ind in enumerate(use_hf.index.values):
        row = use_hf.loc[ind]
        if type(row['Gt_ori_tuning_tf']) != float:
            tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
            tuning = tuning - row['Gt_drift_spont']
            tuning[tuning < 0] = 0
            
            tf_responses[i,:2] = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            # tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
            # hffm.at[ind, 'tf_pref'] = tf_pref

    for i in range(np.size(tf_responses,0)):
        _l = tf_responses[i,0]
        _h = tf_responses[i,1]
        tf_responses[i,2] = _l/_h
        tf_responses[i,3] = _h/_l

    use_counts = np.zeros([4,6])*np.nan

    k_list = ['early','late','biphasic','negative']

    fig, [ax,ax1] = plt.subplots(2,1,figsize=(3,3.5), dpi=300)

    for k, kname in enumerate(k_list):
        
        use_hf = hffm[hffm['gazecluster']==kname][hffm['Gt_responsive']==True].copy()
        
        ### TEMPORAL FREQUENCY
        
        tf_responses = np.zeros([len(use_hf.index.values),4])*np.nan
        
        for i, ind in enumerate(use_hf.index.values):
            row = use_hf.loc[ind]
            if type(row['Gt_ori_tuning_tf']) != float:
                tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
                tuning = tuning - row['Gt_drift_spont']
                tuning[tuning < 0] = 0

                tf_responses[i,:2] = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            
        for i in range(np.size(tf_responses,0)):
            _l = tf_responses[i,0]
            _h = tf_responses[i,1]
            tf_responses[i,2] = _l/_h
            tf_responses[i,3] = _h/_l

        num_low_pref = len(use_hf.index.values[tf_responses[:,2]>2.0]) / len(use_hf.index)
        num_high_pref = len(use_hf.index.values[tf_responses[:,3]>2.0]) / len(use_hf.index)

        ax.bar(k, num_low_pref, width=0.35, color=colors[kname])
        ax.bar(k+0.38, num_high_pref, width=0.35, color=colors[kname])
        
        use_counts[k,0] = len(use_hf.index)
        use_counts[k,1] = len(use_hf.index.values[tf_responses[:,2]>2.0])
        use_counts[k,2] = len(use_hf.index.values[tf_responses[:,3]>2.0])
        
        
        ### SPATIAL FREQUENCY
        
        sf_responses = np.zeros([len(use_hf.index.values),6])*np.nan
        
        for i, ind in enumerate(use_hf.index.values):
            row = use_hf.loc[ind]
            orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
            for sfnum in range(3):
                sf = ['low','mid','high'][sfnum]
                hffm.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
            sf_responses[i,:3] = np.array([np.mean(hffm.at[ind,'norm_ori_tuning_low']), np.mean(hffm.at[ind,'norm_ori_tuning_mid']), np.mean(hffm.at[ind,'norm_ori_tuning_high'])])

        for i in range(np.size(sf_responses,0)):
            _l = sf_responses[i,0]
            _m = sf_responses[i,1]
            _h = sf_responses[i,2]
            sf_responses[i,3] = _l/(_m+_h)
            sf_responses[i,4] = _m/(_l+_h)
            sf_responses[i,5] = _h/(_l+_m)

        num_low_pref = len(use_hf.index.values[sf_responses[:,3]>2.0]) / len(use_hf.index)
        num_mid_pref = len(use_hf.index.values[sf_responses[:,4]>2.0]) / len(use_hf.index)
        num_high_pref = len(use_hf.index.values[sf_responses[:,5]>2.0]) / len(use_hf.index)
        
        use_counts[k,3] = len(use_hf.index.values[sf_responses[:,3]>2.0])
        use_counts[k,4] = len(use_hf.index.values[sf_responses[:,4]>2.0])
        use_counts[k,5] = len(use_hf.index.values[sf_responses[:,5]>2.0])

        ax1.bar(k, num_low_pref, width=0.3, color=colors[kname])
        ax1.bar(k+0.32, num_mid_pref, width=0.3, color=colors[kname])
        ax1.bar(k+0.64, num_high_pref, width=0.3, color=colors[kname])
        
    ax.set_xticks([0, 0.38, 1, 1.38, 2, 2.38, 3, 3.38], labels=['l','h','l','h','l','h','l','h'])
    ax.set_ylabel('TF pref. fraction')

    ax1.set_xticks([0,.3,.6,1,1.3,1.6,2,2.3,2.6,3,3.3,3.6], labels=['l','m','h','l','m','h','l','m','h','l','m','h'])
    ax1.set_ylabel('SF pref. fraction')

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'frac_pref_2to1_TF_SF.pdf'))

    fig, ax = plt.subplots(1,1,figsize=(2.5,2),dpi=300)

    use_data = hffm[hffm['gazecluster']!='unresponsive'][hffm['FmLt_gazeshift_peakT']<.2].copy()
    #[hffm['FmLt_gazeshift_peakT']<.2]

    plot_cprop_scatter(ax, use_data, 'FmLt_gazeshift_peakT', use_median=True)
    ax.set_ylabel('gaze shift latency (ms)')
    ax.set_xticks(range(4), labels=['early','late','biph','neg'])
    ax.set_ylim([0, 0.205])

    ax.set_yticks(np.linspace(0,.20,5), labels=np.linspace(0,200,5).astype(int))

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'gaze_latency_scatter.pdf'))

    fig5 = plt.figure(constrained_layout=False, figsize=(7,8), dpi=300)
    fig5spec = gridspec.GridSpec(nrows=3, ncols=2, figure=fig5, wspace=.5, hspace=.5)

    fig5Cspec = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=fig5spec[1:3,0:2], wspace=.5, hspace=.5)
    ax_tf_cat = fig5.add_subplot(fig5Cspec[0,0])
    ax_tf_scatter = fig5.add_subplot(fig5Cspec[1,0])
    ax_sf_cat = fig5.add_subplot(fig5Cspec[0,1])
    ax_sf_scatter = fig5.add_subplot(fig5Cspec[1,1])

    use_data = hffm[hffm['gazecluster']!='unresponsive'][hffm['Gt_responsive']==True].copy()
    # [hffm['FmLt_gazeshift_peakT']>0.035][hffm['FmLt_gazeshift_peakT']<.2]

    plot_cprop_scatter(ax_tf_cat, use_data, 'tf_pref_cps', use_median=True)
    ax_tf_cat.set_ylabel('temporal freq (cps)')
    ax_tf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    # ax_tf_cat.set_ylim([0, 8.05])

    plot_cprop_scatter(ax_sf_cat, use_data, 'sf_pref_cpd', use_median=True)
    ax_sf_cat.set_ylabel('spatial freq (cpd)')
    ax_sf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    ax_sf_cat.set_ylim([0, 0.25])

    ###
    names = ['early','late','biphasic','negative']
    for i, name in enumerate(names):
        # cluster = hffm[hffm['movcluster']==name][hffm['responsive_to_gratings']==True][hffm['Fm_fr']>2][hffm['Gt_fr']>2]
        cluster = use_data[use_data['gazecluster']==name][hffm['FmLt_gazeshift_peakT']<.2].copy()
        ax_tf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['tf_pref_cps'], '.', color=colors[name], markersize=3)
        ax_sf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['sf_pref_cpd'], '.', color=colors[name], markersize=3)
        #[cluster['fr']>2]
    # running_median(ax_tf_scatter, hffm['FmLt_gazeshift_peakT'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True], hffm['tf_pref_cps'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True])
    # running_median(ax_sf_scatter, hffm['FmLt_gazeshift_peakT'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True], hffm['sf_pref_cpd'][hffm['movcluster']!='unresponsive'][hffm['responsive_to_gratings']==True])

    running_median(ax_tf_scatter, use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2], use_data['tf_pref_cps'][hffm['FmLt_gazeshift_peakT']<.2], n_bins=7)
    running_median(ax_sf_scatter, use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2], use_data['sf_pref_cpd'][hffm['FmLt_gazeshift_peakT']<.2], n_bins=7)

    # ax_tf_scatter.set_yscale('log')
    # ax_sf_scatter.set_yscale('log')

    # tf_R = plot_linregress(ax_tf_scatter, use_data['FmLt_gazeshift_peakT'], use_data['tf_pref_cps'])
    # sf_R = plot_linregress(ax_sf_scatter, use_data['FmLt_gazeshift_peakT'], use_data['sf_pref_cpd'])

    # ax_tf_scatter.set_ylim([0, 8.05])
    ax_tf_scatter.set_xlim([.020,.205])
    ax_sf_scatter.set_xlim([.020,.205])
    ax_tf_scatter.set_xticks(np.linspace(.025, 0.200, 6), labels=np.linspace(25, 200, 6).astype(int))
    ax_sf_scatter.set_xticks(np.linspace(.025, 0.200, 6), labels=np.linspace(25, 200, 6).astype(int))

    # ax_tf_scatter.set_xticks(np.linspace(.035, 0.2, 4), labels=np.linspace(35, 200, 4).astype(int))
    ax_tf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_tf_scatter.set_ylabel('temporal freq (cps)')

    # ax_sf_scatter.set_xticks(np.linspace(.035, 0.2, 4), labels=np.linspace(35, 200, 4).astype(int))
    ax_sf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_sf_scatter.set_ylabel('spatial freq (cpd)')
    ax_sf_scatter.set_ylim([0,0.25])

    fig5.savefig(os.path.join(figpath, '5_props.pdf'))



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
    fig.savefig(os.path.join(figpath, 'mean_head_pitch_roll.pdf'))


    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(5,2), dpi=300)

    _usebins = np.linspace(-50,50,20)

    useRow = hffm[hffm['session']=='070921_J553RT_control_Rig2'].copy()
    usePitch = useRow['FmLt_pitch'].iloc[0]
    useRoll = useRow['FmLt_roll'].iloc[0]

    ax0.hist(usePitch, alpha=0.5, color='k', density=True, bins=_usebins)#, bins=_usebins, linewidth=2)

    ax1.hist(useRoll, alpha=0.5, color='k', density=True, bins=_usebins)#, bins=_usebins, linewidth=2)

    # ax1.hist(speeds_arr[:,2,10], alpha=0.5, color=colors['gaze'], density=True, bins=_usebins, linewidth=2)
    # ax1.hist(compspeeds_arr[:,2,10], alpha=0.5, color=colors['comp'], density=True, bins=_usebins, linewidth=2)

    # ax2.hist(speeds_arr[:,3,10], alpha=0.5, color=colors['gaze'], density=True, bins=_usebins, linewidth=2)
    # ax2.hist(compspeeds_arr[:,3,10], alpha=0.5, color=colors['comp'], density=True, bins=_usebins, linewidth=2)

    ax0.set_xlim([-50,50])
    ax1.set_xlim([-50,50])
    # ax1.set_xlim([-1500,1500])

    ax0.set_xlabel('head pitch (deg)')
    ax1.set_xlabel('head roll (deg)')
    # ax2.set_xlabel('dGaze (deg/sec)')

    # ax0.set_yticklabels(['{:,.2%}'.format(x) for x in ax0.get_yticks()])
    # ax1.set_yticklabels(['{:,.2%}'.format(x) for x in ax1.get_yticks()])
    # ax2.set_yticklabels(['{:,.2%}'.format(x) for x in ax2.get_yticks()])

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, '0709_hist_pitch_roll.pdf'))

    _mean = np.nanmean(useRoll)
    _stderr = np.std(useRoll) / np.sqrt(np.size(useRoll))
    print(_mean, _stderr)



    def make_col(ax, c, data, usecolor):
        
        x_jitter = np.random.uniform(c-0.1, c+0.1, np.size(data,0))
        ax.plot(x_jitter, data, '.', color=usecolor, markersize=1)
        hline = np.nanmedian(data)
        ax.hlines(hline, c-0.1, c+0.1, color='k', linewidth=2)
        err = np.std(data) / np.sqrt(np.size(data))
        ax.vlines(c, hline-err, hline+err, color='k', linewidth=2)
        
        
        return hline, err


    ### firing rates hf vs fm
    fig, [ax0,ax1] = plt.subplots(1,2, figsize=(4.3,2.2), dpi=300)

    clusters = ['early','late','biphasic','negative']

    use_xlabels = []

    for c, name in enumerate(clusters):
        
        gazebase = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        gtISI = hffm['Gt_drift_spont'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        
        ax0.plot(gtISI, gazebase, '.', color=colors[name], markersize=2)
        
        make_col(ax1, c, gtISI, colors[name])
        make_col(ax1, c+0.4, gazebase, colors[name])
        
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

    fig, axs = plt.subplots(2,3, figsize=(5, 3.1), dpi=300)

    clusters = ['early','late','biphasic','negative','unresponsive']

    axs = axs.flatten()

    for c, name in enumerate(clusters):
        
        ax = axs[c]
        
        gazebase = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        gtISI = hffm['Gt_drift_spont'][hffm['gazecluster']==name].copy().to_numpy().astype(float)
        
        ax.plot(gtISI, gazebase, '.', color=colors[name], markersize=2)
        
        res = plot_linregress1(ax, gtISI, gazebase)
        
        print(res.rvalue, res.pvalue, res.slope)
        
        # ax.axis('equal')
        ax.set_xlim([0,30])
        ax.set_ylim([0,30])
        ax.set_xticks(np.linspace(0,30,4).astype(int))
        ax.set_yticks(np.linspace(0,30,4).astype(int))
        
    axs[0].set_ylabel('freely moving (sp/s)')
    axs[3].set_xlabel('head-fixed (sp/s)')

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'baseline_hfm_fm_scatters.pdf'))

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
    fig.savefig(os.path.join(figpath, 'baseline_rate_bar_plot.pdf'))



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
    plt.savefig(os.path.join(figpath, 'dEye_dHead_with_inter.pdf'))

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
        
        interL1 = drop_repeat_events(eyeT[interL])
        interR1 = drop_repeat_events(eyeT[interR])
        
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
    fig.savefig(os.path.join(figpath, 'example_inter_PSTH_units.pdf'))

    hffm['pref_intersacc_psth'] = hffm['FmLt_intersacc_left_saccPSTH_dHead1'].copy().astype(object)
    for ind, row in tqdm(hffm.iterrows()):
        LR = row['pref_gazeshift_direction']
        if type(LR)==str:
            hffm.at[ind,'pref_intersacc_psth'] = normalize_psth(row['FmLt_intersacc_{}_saccPSTH_dHead1'.format(LR)])
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

    all_comp = flatten_series(hffm['pref_comp_psth'][hffm['gazecluster']!='unresponsive'])
    all_gaze = flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']!='unresponsive'])
    all_inter = flatten_series(hffm['pref_intersacc_psth'][hffm['gazecluster']!='unresponsive'])

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
    fig.savefig(os.path.join(figpath,'gaze_inter_comp_means.pdf'))

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

    def _plotCol(ax, c, data, demoPt=None, color='k'):

        x_jitter = np.random.uniform(c-0.07, c+0.07, np.size(data,0))
        ax.plot(x_jitter, data, '.', color=color, markersize=3)
        if demoPt is not None:
            ax.plot(x_jitter[demopt], data[demopt], '.', color='tab:orange', markersize=4)
        
        
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
    # ax2.set_ylim([450, 650])


    # ax0.set_xticks([0,0.5], labels=['pitch','roll'])
    # ax0.set_ylim([-35,35])
    # ax0.set_xlim([-0.3,0.8])


    # ax0.set_ylabel('mean orientation (deg)')
    # ax0.plot(x_jitter, cluster_data, '.', color='grey', markersize=4)
    # ax0.plot(x_jitter[demopt], cluster_data[demopt], '.', color='tab:orange', markersize=4)
    # # ax0.set_ylim([-35,35])
    # print(hline, err)

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'mean_gaze_amplitude_columns.pdf'))

    plt.figure(dpi=300, figsize=(3,2))

    plt.hist(dGaze_sizes[sessions[1]]['gaze'] * (1/60), bins=np.arange(4,60,.2), color=colors['gaze'])

    plt.hist(dGaze_sizes[sessions[1]]['inter'] * (1/60), bins=np.arange(2,4.2,.2), color='k')

    plt.hist(dGaze_sizes[sessions[1]]['comp'] * (1/60), bins=np.arange(0,2.2,.2), color=colors['comp'])

    plt.xlim([0,25])
    # plt.ylim([0,40])

    plt.tight_layout()

    plt.xlabel('gaze shift amplitude (deg)')
    plt.ylabel('movement count')

    plt.savefig(os.path.join(figpath, 'gaze_shift_amp_w_inter.pdf'))

if __name__ == '__main__':
    main()