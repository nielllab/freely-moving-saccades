import numpy as np
import matplotlib.pyplot as plt

def figS2():
    ### depth
    fig2 = plt.figure(constrained_layout=True, figsize=(8,3), dpi=300)
    fig2spec = gridspec.GridSpec(nrows=1, ncols=6, figure=fig2, wspace=.1, hspace=0)

    ax_ex_depth = fig2.add_subplot(fig2spec[0,0])
    ax_early_depth = fig2.add_subplot(fig2spec[0,1])
    ax_late_depth = fig2.add_subplot(fig2spec[0,2])
    ax_biphasic_depth = fig2.add_subplot(fig2spec[0,3])
    ax_negative_depth = fig2.add_subplot(fig2spec[0,4])
    ax_unresp_depth = fig2.add_subplot(fig2spec[0,5])

    mua_power = hffm['Wn_lfp_power'][hffm['session']=='101521_J559NC_control_Rig2'].iloc[0]
    layer5 = hffm['Wn_layer5cent_from_lfp'][hffm['session']=='101521_J559NC_control_Rig2'].iloc[0]
    ch_spacing = 25
    for sh in range(4):
        ax_ex_depth.plot(mua_power[sh], np.arange(0,32)-layer5[sh], 'tab:red')
    # ax_ex_depth.set_title('example recording depth')
    ax_ex_depth.hlines(0,np.min(mua_power),np.max(mua_power), 'k', linestyle='dashed')
    ax_ex_depth.set_ylim([18,-19])
    ax_ex_depth.set_yticks(ticks=np.arange(18,-19,-6), labels=(ch_spacing*np.arange(18,-19,-6)))
    ax_ex_depth.set_ylabel('depth (um)'); ax_ex_depth.set_xlabel('LFP MUA power')
    ax_ex_depth.annotate('layer 5', xy=[0.75, -.5], color='k', fontsize=12)

    panels = [ax_early_depth, ax_late_depth, ax_biphasic_depth, ax_negative_depth, ax_unresp_depth]
    names = ['early','late','biphasic','negative','unresponsive']

    popdata = hffm['Wn_depth_from_layer5'].to_numpy()
    pop_weights = np.ones_like(popdata) / float(len(popdata))

    for i, panel in enumerate(panels):
        name = names[i]
        panel.hist(popdata, color='k', bins=np.arange(-600,800,100),
                orientation='horizontal',
                histtype='step', linewidth=2, weights=pop_weights)
        
        if i != 4:
            paneldata = hffm['Wn_depth_from_layer5'][hffm['gazecluster']==name][hffm['gazeshift_responsive']].to_numpy()
        else:
            paneldata = hffm['Wn_depth_from_layer5'][hffm['gazecluster']==name].to_numpy()
        panel_weights = np.ones_like(paneldata) / float(len(paneldata))
        
        
        panel.hlines(0, 0, .2, 'k', linestyle='dashed')
        # panel.set_xlim([0,0.3])
        panel.hist(paneldata, color=colors[name], bins=np.arange(-600,800,100),
                weights=panel_weights, orientation='horizontal', histtype='stepfilled')
        if i==0:
            panel.set_ylabel('depth (um)')
            panel.set_xlabel('fraction of neurons')
        else:
            panel.set_yticklabels([])
            
        panel.set_title(name.capitalize())
        panel.invert_yaxis()
    #     panel.set_xlim(0,0.4)

    #     panel.set_xticks(np.arange(0.,0.4,.15))
        
    fig2.savefig(os.path.join(figpath, 'S1_depth.pdf'))


    fig, [ax_cellcounts, ax_baseline_fr] = plt.subplots(1,2, figsize=(5.5,2.5), dpi=300)

    vcounts = hffm['gazecluster'].value_counts()
    names = ['early','late','biphasic','negative','unresponsive']
    print_names = ['early','late','biph','neg','unresp']
    for i, name in enumerate(names):
        ax_cellcounts.bar(i, vcounts[name]/np.sum(vcounts), color=colors[name])
    ax_cellcounts.set_xticks(ticks=range(5), labels=print_names, rotation=90)
    ax_cellcounts.set_ylabel('frac. cells')

    for i, name in enumerate(names):
        baselines = hffm['FmLt_gazeshift_med_baseline'][hffm['gazecluster']==name].to_numpy()
        err = np.std(baselines) / np.sqrt(np.size(baselines))
        med = np.median(baselines)
        ax_baseline_fr.bar(i, med, color=colors[name])
        ax_baseline_fr.plot([i, i], [med-err, med+err], 'k-')
    ax_baseline_fr.set_xticks(range(5), print_names, rotation=90)
    ax_baseline_fr.set_ylabel('baseline (sp/s)')

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'S1_frac_and_rate.pdf'))

    ### Direction selectivity index (for dark-responsive cells)
    fig, ax = plt.subplots(1,1, dpi=300, figsize=(2.75,1.5))
    vals = ltdk['gazeshift_DSI'][ltdk['dark_responsive']].copy()
    weights = np.ones_like(vals) / float(len(vals))
    ax.hist(vals, bins=np.linspace(0,1,13),weights=weights,
            color='k', histtype='stepfilled', alpha=0.7)
    ax.set_ylim([0,.5])
    ax.set_xlabel('gaze-shifting DSI')
    ax.set_ylabel('frac. cells')
    fig3A.savefig(os.path.join(figpath, 'S2_dark_responsive_DSI_in_light.pdf'))

    ### Putative cell type
    fig, ax = plt.subplots(1,1, dpi=300, figsize=(3,2))
    for i, name in enumerate(['early','late','biphasic','negative','unresponsive']):
        vcounts = hffm['exc_or_inh'][hffm['gazecluster']==name].value_counts()
        ax.bar(i, vcounts['exc']/np.sum(vcounts), color='grey')
        ax.bar(i, vcounts['inh']/np.sum(vcounts), bottom=vcounts['exc']/np.sum(vcounts), color='k')
    ax.set_xticks(ticks=range(5), labels=['early','late','biph','neg','unresp'])
    ax.set_ylabel('frac. of cluster')
    fig.savefig(os.path.join(figpath, 'S2_celltype.pdf'))

    for ind, row in hffm.iterrows():
        hffm.at[ind, 'gazeshift_DSI'] = calc_psth_DSI(row['pref_gazeshift_psth_raw'].astype(float), row['nonpref_gazeshift_psth_raw'].astype(float))


    ### gaze direction selectivity index (light)
    fig, axs = plt.subplots(4,1,figsize=(2.8,4),dpi=300)

    for p, name in enumerate(['early','late','biphasic','negative']):
        ax = axs[p]
        
        vals = hffm['gazeshift_DSI'][hffm['gazecluster']==name].copy()
        weights = np.ones_like(vals) / float(len(vals))
        ax.hist(vals, bins=np.linspace(0,1,13),weights=weights,
                color=colors[name], histtype='stepfilled', alpha=0.7)
        ax.set_ylim([0,.5])
        if p==3:
            # ax.set_title(name)
            ax.set_xlabel('gaze-shifting DSI')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('frac. cells')
        
    plt.tight_layout()
    fig.savefig(os.path.join(figpath, 'S1_gaze_DSI.pdf'))

    ### plot PCA
    proj = np.load('/home/niell_lab/Data/freely_moving_ephys/batch_files/062022/dEye_PSTH_pca1-new.npy')
    fig, ax = plt.subplots(1,1,figsize=(3,2), dpi=300)
    for name in ['unresponsive','early','late','biphasic','negative']:
        use = np.array([hffm['gazecluster']==name].copy())[0]
        ax.scatter(proj[use,0], proj[use,1], s=2, c=colors[name])
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    fig.savefig(os.path.join(figpath, 'S1_pca.pdf'))

