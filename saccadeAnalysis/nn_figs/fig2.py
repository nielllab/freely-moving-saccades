    

    fig, ax = plt.subplots(1,1,figsize=(0.5,1.5), dpi=300)
    ax.imshow(tseq_legend1, aspect=.05)
    ax.set_yticks([]); ax.set_xticks([])
    ax.axes.spines.bottom.set_visible(False)
    ax.axes.spines.right.set_visible(False)
    ax.axes.spines.left.set_visible(False)
    ax.axes.spines.top.set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(figpath, '2_hffm_tseq_legend.pdf'))


    ### figure 2: clustering
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
        dpanel.axis('off')
        
        for x in hffm['pref_gazeshift_psth'][hffm['gazecluster']==name]:
            lpanel.plot(psth_bins, x, '-', linewidth=1, alpha=.3)
        lpanel.plot(psth_bins, np.nanmean(flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']==name]),0), '-', linewidth=3, color=colors[name])
        lpanel.set_xlim([-0.2,0.4]); lpanel.set_ylim([-1,1])
    #     lpanel.set_title(name.capitalize())
        lpanel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
        lpanel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        
        # for x in ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name]:
        #     dpanel.plot(psth_bins, x, '-', linewidth=1, alpha=.3)
        # dpanel.plot(psth_bins, np.nanmean(flatten_series(ltdk['pref_dark_gazeshift_psth'][ltdk['gazecluster']==name][ltdk['gazeshift_responsive']==True][ltdk['movement']==False]),0), '-', linewidth=3, color=colors[name])
        # dpanel.set_xlim([-0.2,0.4]); dpanel.set_ylim([-1,1])
        # dpanel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
        # dpanel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        
        if name=='early':
            lpanel.set_ylabel('norm sp/s')
            # dpanel.set_ylabel('norm. spike rate')
        if name!='early':
            lpanel.set_yticklabels([])
            # dpanel.set_yticklabels([])
        # lpanel.set_xticklabels([])
        lpanel.set_xlabel('time (msec)')
        lpanel.set_title(name)
            
    fig3A.savefig(os.path.join(figpath, '2_clustering.pdf'))

    fig2 = plt.figure(constrained_layout=False, figsize=(9,7), dpi=300)
    fig2spec = gridspec.GridSpec(nrows=5, ncols=5, figure=fig2, wspace=1, hspace=1.2)

    fig2Aspec = gridspec.GridSpecFromSubplotSpec(4,2, subplot_spec=fig2spec[:,0:2], wspace=0.8, hspace=1)
    ax_early = fig2.add_subplot(fig2Aspec[0,0])
    ax_late = fig2.add_subplot(fig2Aspec[0,1])
    ax_biphasic = fig2.add_subplot(fig2Aspec[1,0])
    ax_negative = fig2.add_subplot(fig2Aspec[1,1])
    ax_clusters = fig2.add_subplot(fig2Aspec[2:,:])

    panels = [ax_early, ax_late, ax_biphasic, ax_negative]
    movtypes = ['early','late','biphasic','negative']
    for count, panel in enumerate(panels):
        movtype = movtypes[count]
        thisclust = hffm['pref_gazeshift_psth'][hffm['gazecluster']==movtype][hffm['gazeshift_responsive']]
        for i, psth in enumerate(thisclust):
            panel.plot(psth_bins, psth, '-', linewidth=1, alpha=0.25)
        clustmean = np.nanmean(flatten_series(thisclust),0)
        panel.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[movtype])
        panel.set_xlim([-0.2,0.4])
        panel.set_title(movtype.capitalize())
        panel.set_ylim([-1,1])
        if count == 0 or count == 2:
            panel.set_ylabel('norm. spike rate')
    #     else:
    #         panel.set_yticklabels([])
        if count == 3 or count == 2:
            panel.set_xlabel('time (msec)')
        panel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    #     else:
    #         panel.set_xticks([])
        panel.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax_early.set_ylim([-0.3,1])
    ax_late.set_ylim([-0.3,1])
    ax_biphasic.set_ylim([-0.6,0.9])
    ax_negative.set_ylim([-.9,0.7])

    ax_late.set_yticks(np.linspace(0,1,3))
    ax_early.set_yticks(np.linspace(0,1,3))
        

    step = 0.13
    names = ['early','late','biphasic','negative'] # 'unresponsive',
    for count, name in enumerate(names):
        cluster_psths = flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']==name])
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.3)
    ax_clusters.set_xlim([-0.2,0.4]); ax_clusters.set_ylim([-.6,.6])
    # ax_clusters.annotate('early', xy=[0.19,-0.22], color=colors['early'], fontsize=11)
    # ax_clusters.annotate('late', xy=[0.19,-0.22-(step*1)], color=colors['late'], fontsize=11)
    # ax_clusters.annotate('biphasic', xy=[0.19,-0.22-(step*2)], color=colors['biphasic'], fontsize=11)
    # ax_clusters.annotate('negative', xy=[0.19,-0.22-(step*3)], color=colors['negative'], fontsize=11)
    ax_clusters.set_ylabel('norm. spike rate'); ax_clusters.set_xlabel('time (msec)')
    ax_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_clusters.set_yticks(np.linspace(-0.5,0.5,3))

    # plot_cprop_scatter(ax_baseline_fr, 'psth_baseline', use_median=True)
    # ax_baseline_fr.set_ylabel('baseline (sp/sec)')
    # ax_baseline_fr.set_ylim([0,50])

    fig2.savefig(os.path.join(figpath, '2_clustering-1.pdf'))



    # comp clusters
    fig, ax_clusters = plt.subplots(1,1,figsize=(3,3), dpi=300)

    names = ['early','late','biphasic','negative']
    for count, name in enumerate(names):
        cluster_psths = flatten_series(hffm['pref_comp_psth'][hffm['gazecluster']==name][hffm['gazeshift_responsive']])
        cluster_psths = drop_nan_along(cluster_psths)
        clustmean = np.nanmean(cluster_psths, 0)
        clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
        ax_clusters.plot(psth_bins, clustmean, '-', linewidth=2, color=colors[name])
        ax_clusters.fill_between(psth_bins, clustmean-clusterr, clustmean+clusterr, color=colors[name], alpha=0.2)
    ax_clusters.set_xlim([-0.2,0.4]); ax_clusters.set_ylim([-.6,.6])
    ax_clusters.set_ylabel('norm. spike rate'); ax_clusters.set_xlabel('time (msec)')
    ax_clusters.vlines(0,-1,1,color='k',linestyle='dashed',linewidth=1)
    ax_clusters.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
    ax_clusters.set_yticks(np.linspace(-0.5,0.5,3))

    fig.savefig(os.path.join(figpath, '2_comp.pdf'))





    # temporal sequence
    fig2p2 = plt.figure(constrained_layout=False, figsize=(9,5), dpi=300)

    fig2p2A = gridspec.GridSpec(1,3, figure=fig2p2, wspace=.28, hspace=.1)
    ax_tseq_pref = fig2p2.add_subplot(fig2p2A[:,0])
    ax_tseq_nonpref = fig2p2.add_subplot(fig2p2A[:,1])
    ax_tseq_comp = fig2p2.add_subplot(fig2p2A[:,2])

    tseq_aspect = 2.8
    img = plot_tempseq(ax_tseq_pref, tseq_pref1, return_img=True)
    ax_tseq_pref.set_aspect(tseq_aspect)
    ax_tseq_pref.set_title('Pref.')
    ax_tseq_pref.set_ylabel('cells')
    # ax_tseq_pref.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    plot_tempseq(ax_tseq_nonpref, tseq_nonpref1)
    ax_tseq_nonpref.set_aspect(tseq_aspect)
    ax_tseq_nonpref.set_title('Nonpref.')
    ax_tseq_nonpref.set_yticklabels([])
    # ax_tseq_nonpref.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    plot_tempseq(ax_tseq_comp, tseq_comp1)
    ax_tseq_comp.set_aspect(tseq_aspect)
    ax_tseq_comp.set_title('Comp.')
    ax_tseq_comp.set_yticklabels([])
    # ax_tseq_comp.hlines(num_responsive, 800,1400,'k',linestyle='dashed', linewidth=1)

    fig2p2.savefig(os.path.join(figpath, '2_temporal_seq.pdf'))


    fig, ax = plt.subplots(figsize=(2,.25), dpi=300)
    plt.colorbar(img, cax=ax, aspect=10, orientation='horizontal')
    plt.xticks(np.linspace(-0.75,0.75,3))
    # ax.xaxis.set_tick_params(width=2)
    plt.savefig(os.path.join(figpath, '2_seq_legend.pdf'), bbox_inches='tight', pad_inches=.5)
