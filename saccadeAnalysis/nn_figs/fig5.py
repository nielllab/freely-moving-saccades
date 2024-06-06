
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


def fig5(hffm, savepath):
    
    sacc.set_plt_params()
    props = sacc.prop_dict()
    grat_bins = props['psth_bins_long']
    colors = props['colors']    

    hffm['raw_mod_for_Gt'] = np.nan
    hffm['norm_mod_for_Gt'] = np.nan
    hffm['Gt_kde_psth_norm'] = hffm['Gt_kde_psth'].copy()

    for ind, row in hffm.iterrows():
        sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
        sp = len(row['Gt_spikeT'])
        hffm.at[ind, 'Gt_fr'] = sp/sec
        
        norm_psth = sacc.norm_PSTH(row['Gt_kde_psth'], trange='gt')
        hffm.at[ind, 'Gt_kde_psth_norm'] = norm_psth.copy().astype(object)
        
        hffm.at[ind, 'raw_mod_for_Gt'] = sacc.calc_PSTH_modind(row['Gt_kde_psth'], trange='gt')
        
        hffm.at[ind, 'norm_mod_for_Gt'] = sacc.calc_PSTH_modind(norm_psth, trange='gt')
        
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
        
        mean_for_sf = np.array([
            np.mean(hffm.at[ind,'norm_ori_tuning_low']),
            np.mean(hffm.at[ind,'norm_ori_tuning_mid']),
            np.mean(hffm.at[ind,'norm_ori_tuning_high'])]
        )
        
        mean_for_sf[mean_for_sf<0] = 0
        hffm.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3)) / np.sum(mean_for_sf)

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
            mean_for_sf = np.array([
                np.mean(hffm.at[ind,'norm_ori_tuning_low']),
                np.mean(hffm.at[ind,'norm_ori_tuning_mid']),
                np.mean(hffm.at[ind,'norm_ori_tuning_high'])]
            )
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
        
        norm_psth = sacc.norm_PSTH(
            row['Gt_kde_psth'],
            trange='gt')
        
        hffm.at[ind, 'Gt_kde_psth_norm'] = norm_psth.copy().astype(object)
        
        hffm.at[ind, 'raw_mod_for_Gt'] = sacc.calc_PSTH_modind(
            row['Gt_kde_psth'],
            trange='gt')
        
        hffm.at[ind, 'norm_mod_for_Gt'] = sacc.calc_PSTH_modind(
            norm_psth,
            trange='gt')
        
    hffm['Gt_responsive'] = False
    for ind, row in hffm.iterrows():
        if ((row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1)   \
                    and (np.max(row['Gt_kde_psth_norm'][1500:2500]) > 0.5)):
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
        clustmean = np.nanmean(fme.flatten_series(thisclust),0)
        panel.plot(grat_bins, clustmean, '-', linewidth=2, color=colors[movtype])
        
        panel.vlines(0,-1.5,1.5, color='k',linestyle='dashed',linewidth=1)
        panel.set_ylim([-0.5,1.2])
        panel.set_xlim([-400,1600])

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

    fig.savefig(os.path.join(savepath, '5_clusters.pdf'))


    fig, ax = plt.subplots(1,1,figsize=(2.5,2.2), dpi=300)

    names = ['early','late','biphasic','negative']
    
    for count, name in enumerate(names):
        
        cluster_psths = fme.flatten_series(
            hffm['Gt_kde_psth_norm'][hffm['gazecluster']==name][hffm['Gt_responsive']==True]
        )
        cluster_psths = cluster_psths[~np.isnan(cluster_psths).any(axis=1)]
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        
        ax.plot(grat_bins, clustmean, '-', linewidth=1.5, color=colors[name])
        ax.fill_between(grat_bins, clustmean-clusterr, clustmean+clusterr,
                        color=colors[name], alpha=0.2)
    
    ax.set_ylabel('norm sp/s'); ax.set_xlabel('time (ms)')
    ax.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax.set_ylim([-.1,.75])
    ax.set_xlim([-500,1500])
    ax.set_xlim([-100,200])


    fig.savefig(os.path.join(savepath, '5_grat_onsets.pdf'))

    fig5 = plt.figure(constrained_layout=False, figsize=(7,8), dpi=300)
    fig5spec = gridspec.GridSpec(nrows=3, ncols=2, figure=fig5, wspace=.5, hspace=.5)

    fig5Cspec = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=fig5spec[1:3,0:2], wspace=.5, hspace=.5)
    ax_tf_cat = fig5.add_subplot(fig5Cspec[0,0])
    ax_tf_scatter = fig5.add_subplot(fig5Cspec[1,0])
    ax_sf_cat = fig5.add_subplot(fig5Cspec[0,1])
    ax_sf_scatter = fig5.add_subplot(fig5Cspec[1,1])

    use_data = hffm[hffm['gazecluster']!='unresponsive'][hffm['Gt_responsive']==True].copy()
    # [hffm['FmLt_gazeshift_peakT']>0.035][hffm['FmLt_gazeshift_peakT']<.2]

    sacc.plot_columns(ax_tf_cat, use_data, 'tf_pref_cps', use_median=True)
    ax_tf_cat.set_ylabel('temporal freq (cps)')
    # ax_tf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    # ax_tf_cat.set_ylim([0, 8.05])

    sacc.plot_columns(ax_sf_cat, use_data, 'sf_pref_cpd', use_median=True)
    ax_sf_cat.set_ylabel('spatial freq (cpd)')
    # ax_sf_cat.set_xticks(range(4), labels=['early','late','biph','neg'])
    ax_sf_cat.set_ylim([0, 0.25])

    ###
    
    names = ['early','late','biphasic','negative']
    
    for i, name in enumerate(names):
        
        # cluster = hffm[hffm['movcluster']==name][hffm['responsive_to_gratings']==True][hffm['Fm_fr']>2][hffm['Gt_fr']>2]
        cluster = use_data[use_data['gazecluster']==name][hffm['FmLt_gazeshift_peakT']<.2].copy()
        
        ax_tf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['tf_pref_cps'],
                           '.', color=colors[name], markersize=3)
        ax_sf_scatter.plot(cluster['FmLt_gazeshift_peakT'], cluster['sf_pref_cpd'],
                           '.', color=colors[name], markersize=3)

    sacc.plot_running_median(ax_tf_scatter,
                             use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2],
                             use_data['tf_pref_cps'][hffm['FmLt_gazeshift_peakT']<.2],
                             n_bins=7)
    
    sacc.plot_running_median(ax_sf_scatter,
                             use_data['FmLt_gazeshift_peakT'][hffm['FmLt_gazeshift_peakT']<.2],
                             use_data['sf_pref_cpd'][hffm['FmLt_gazeshift_peakT']<.2],
                             n_bins=7)

    ax_tf_scatter.set_xlim([.020,.205])
    ax_sf_scatter.set_xlim([.020,.205])
    ax_tf_scatter.set_xticks(np.linspace(.025, 0.200, 6),
                             labels=np.linspace(25, 200, 6).astype(int))
    ax_sf_scatter.set_xticks(np.linspace(.025, 0.200, 6),
                             labels=np.linspace(25, 200, 6).astype(int))

    ax_tf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_tf_scatter.set_ylabel('temporal freq (cps)')

    ax_sf_scatter.set_xlabel('gaze shift latency (msec)')
    ax_sf_scatter.set_ylabel('spatial freq (cpd)')
    ax_sf_scatter.set_ylim([0,0.25])

    fig5.savefig(os.path.join(savepath, '5_props.pdf'))



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
    