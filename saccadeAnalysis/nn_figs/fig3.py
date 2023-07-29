
def fig3():

    ### fig 3 light/dark
    for ind in ltdk.index.values:
        Lt_peakT, Lt_peakVal = calc_latency(ltdk.loc[ind,'pref_gazeshift_psth'])
        
        ltdk.at[ind, 'FmLt_gazeshift_peakT'] = Lt_peakT

    for ind in ltdk.index.values:
        sorted_df = ltdk[['FmLt_gazeshift_peakT','FmDk_gazeshift_peakT','FmLt_gazeshift_troughT','FmDk_gazeshift_troughT','gazecluster',
                                'pref_gazeshift_psth','pref_dark_gazeshift_psth','nonpref_dark_gazeshift_psth','gazeshift_responsive',
                                'pref_dark_comp_psth']].copy()

    # shuffle unresponsive cells
    tseq_unresp = sorted_df.copy()
    tseq_unresp = tseq_unresp[tseq_unresp['gazecluster']=='unresponsive'].sample(frac=1).reset_index(drop=True)
    tseq_l_unresp = flatten_series(tseq_unresp['pref_gazeshift_psth'].copy())
    tseq_d_unresp = flatten_series(tseq_unresp['pref_dark_gazeshift_psth'].copy())
        
    # sort dark by dark times
    tseq_dark_sort = sorted_df.copy()
    tseq_dark_sort = tseq_dark_sort[tseq_dark_sort['gazecluster']!='unresponsive']
    tseq_dark_sort.sort_values(by='FmDk_gazeshift_peakT', inplace=True)

    tseq_dark_by_dark = np.vstack([flatten_series(tseq_dark_sort['pref_dark_gazeshift_psth'].copy()), tseq_d_unresp])

    # sort light/dark by light times
    sort_by_light = sorted_df.copy()
    sort_by_light = sort_by_light[sort_by_light['gazecluster']!='unresponsive']
    sort_by_light.sort_values(by='FmLt_gazeshift_peakT', inplace=True)

    tseq_light = flatten_series(sort_by_light['pref_gazeshift_psth'].copy())
    tseq_dark_pref = flatten_series(sort_by_light['pref_dark_gazeshift_psth'].copy())
    tseq_dark_nonpref = flatten_series(sort_by_light['nonpref_dark_gazeshift_psth'].copy())
    tseq_dark_comp = flatten_series(sort_by_light['pref_dark_comp_psth'].copy())

    tseq_light1 = np.vstack([flatten_series(sort_by_light['pref_gazeshift_psth'].copy()), tseq_l_unresp])
    tseq_dark_pref1 = np.vstack([flatten_series(sort_by_light['pref_dark_gazeshift_psth'].copy()), tseq_d_unresp])
    tseq_dark_nonpref1 = np.vstack([flatten_series(sort_by_light['nonpref_dark_gazeshift_psth'].copy()), tseq_d_unresp])
    tseq_dark_comp1 = np.vstack([flatten_series(sort_by_light['pref_dark_comp_psth'].copy()), tseq_d_unresp])

    tseq_aspect = 2.8

    fig, [ax0, ax1, ax2, ax3] = plt.subplots(1,4,figsize=(10,5), dpi=300)

    plot_tempseq(ax0, tseq_light1)
    ax0.set_aspect(tseq_aspect)
    ax0.set_title('Light gaze-shift')
    ax0.set_ylabel('cells')
    # ax0.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    plot_tempseq(ax1, tseq_dark_pref1)
    ax1.set_aspect(tseq_aspect)
    ax1.set_title('Dark pref')
    ax1.set_yticklabels([])
    # ax1.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    plot_tempseq(ax2, tseq_dark_nonpref1)
    ax2.set_aspect(tseq_aspect)
    ax2.set_title('Dark nonpref')
    ax2.set_yticklabels([])
    # ax2.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    plot_tempseq(ax3, tseq_dark_comp1)
    ax3.set_aspect(tseq_aspect)
    ax3.set_title('Dark comp')
    ax3.set_yticklabels([])
    # ax3.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    fig.savefig(os.path.join(figpath, '3_temporal_seq.pdf'))

    ### light dark clustering
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
        lpanel.plot(psth_bins, np.nanmean(flatten_series(ltdk['pref_gazeshift_psth'][ltdk['gazecluster']==name][ltdk['gazeshift_responsive']==True]),0), '-', linewidth=3, color=colors[name])
        lpanel.set_xlim([-0.2,0.4]); lpanel.set_ylim([-1,1])
    #     lpanel.set_title(name.capitalize())
        lpanel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
        lpanel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        
        for x in ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name]:
            dpanel.plot(psth_bins, x, '-', linewidth=1, alpha=.3)
        dpanel.plot(psth_bins, np.nanmean(flatten_series(ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name][ltdk['gazeshift_responsive']==True]),0), '-', linewidth=3, color=colors[name])
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
            
    fig3A.savefig(os.path.join(figpath, '3_clustering.pdf'))

    # light/dark summary
    fig3B = plt.figure(constrained_layout=True, figsize=(5.5,2.5), dpi=300)
    fig3Bspec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig3B, wspace=0.01, hspace=0)

    ax_light_clusters = fig3B.add_subplot(fig3Bspec[:,0])
    ax_dark_clusters = fig3B.add_subplot(fig3Bspec[:,1])

    step = 0.14
    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = flatten_series(data['pref_gazeshift_psth'])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_light_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_light_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2) 
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
        cluster_psths = flatten_series(data['pref_dark_gazeshift_psth'])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_dark_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_dark_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3) 
    ax_dark_clusters.set_xlim([-0.2,0.4]); ax_dark_clusters.set_ylim([-.6,.6]); ax_dark_clusters.set_xlabel('time (msec)')
    ax_dark_clusters.set_title('Dark gaze shift')
    ax_dark_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_dark_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_dark_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    fig3B.savefig(os.path.join(figpath, '3_cluster_summary_noUnresp.pdf'))

    fig3B = plt.figure(constrained_layout=True, figsize=(5.5,2.5), dpi=300)
    fig3Bspec = gridspec.GridSpec(nrows=1, ncols=2, figure=fig3B, wspace=0.01, hspace=0)

    ax_light_clusters = fig3B.add_subplot(fig3Bspec[:,0])
    ax_dark_clusters = fig3B.add_subplot(fig3Bspec[:,1])

    step = 0.14
    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        data = ltdk[ltdk['gazecluster']==name]
        cluster_psths = flatten_series(data['pref_comp_psth'][ltdk['movement']==False])
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
        cluster_psths = flatten_series(data['pref_dark_comp_psth'][ltdk['movement']==False])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_dark_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_dark_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3) 
    ax_dark_clusters.set_xlim([-0.2,0.4]); ax_dark_clusters.set_ylim([-.6,.6]); ax_dark_clusters.set_xlabel('time (msec)')
    ax_dark_clusters.set_title('dark compensatory')
    ax_dark_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_dark_clusters.set_yticks(np.linspace(-0.5,0.5,3))
    ax_dark_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)

    fig3B.savefig(os.path.join(figpath, '3_comp_summary.pdf'))


    tseq_legend_col = sort_by_light['gazecluster'].copy()
    tseq_legend = np.zeros([len(tseq_legend_col.index.values), 1, 4])
    for i, n in enumerate(tseq_legend_col):
        tseq_legend[i,:,:] = mpl.colors.to_rgba(colors[n])
    ucmap = mpl.colors.to_rgba(colors['unresponsive'])
    u = np.zeros([np.size(tseq_l_unresp,0), 1, 4])
    for x in range(4):
        u[:,:,x] = ucmap[x]
    tseq_legend1 = np.vstack([tseq_legend, u])

    fig, ax = plt.subplots(1,1,figsize=(0.5,2), dpi=300)
    ax.imshow(tseq_legend1, aspect=.05)
    ax.set_yticks([]); ax.set_xticks([])
    ax.axes.spines.bottom.set_visible(False)
    ax.axes.spines.right.set_visible(False)
    ax.axes.spines.left.set_visible(False)
    ax.axes.spines.top.set_visible(False)
    fig.savefig(os.path.join(figpath, '3_temp_seq_legend.pdf'))


    for ind, row in ltdk.iterrows():
        sec = row['FmDk_eyeT'][-1].astype(float) - row['FmDk_eyeT'][0].astype(float)
        sp = len(row['FmDk_spikeT'])
        fm_fr = sp/sec
        ltdk.at[ind, 'FmDk_fr'] = fm_fr
        
        ltdk.at[ind, 'norm_mod_at_pref_peak_dark'] = psth_modind(row['pref_dark_gazeshift_psth'])
        
        ltdk.at[ind, 'raw_mod_at_pref_peak_dark'] = psth_modind(row['pref_dark_gazeshift_psth_raw'])
        
        ltdk.at[ind, 'norm_mod_at_pref_peak'] = psth_modind(row['pref_gazeshift_psth'])
        
        ltdk.at[ind, 'raw_mod_at_pref_peak'] = psth_modind(row['pref_gazeshift_psth_raw'])
        
        
        peakT, peak_val = calc_latency(row['pref_dark_gazeshift_psth'])
        ltdk.at[ind, 'dark_peak_val'] = peak_val
        ltdk.at[ind, 'dark_peakT'] = peakT

    vals = ltdk[ltdk['norm_mod_at_pref_peak_dark']>0.1][ltdk['raw_mod_at_pref_peak_dark']>1][ltdk['dark_peak_val']>0.5][ltdk['dark_peakT']<.1]

    ltdk['dark_responsive'] = False
    for ind in vals.index.values:
        ltdk.at[ind, 'dark_responsive'] = True

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

    fig3E.savefig(os.path.join(figpath, '3_ltdk_firing_rates.pdf'))