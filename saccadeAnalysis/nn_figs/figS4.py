    dark_resp_inds = set(ltdk[ltdk['dark_responsive']==True].index.values)
    light_resp_inds = set(ltdk[ltdk['gazecluster']!='unresponsive'].index.values)

    fig, [ax0, ax1] = plt.subplots(1,2,figsize=(4,2.5), dpi=300)
    ax0.set_title('light')
    ax1.set_title('dark')
    light_fracs = [len(light_resp_inds), len(ltdk.index.values)-len(light_resp_inds)]
    dark_fracs = [len(dark_resp_inds), len(ltdk.index.values)-len(dark_resp_inds)]

    ax0.bar(0, light_fracs[0], color=['lightblue'])
    ax0.bar(1, light_fracs[1], color=colors['unresponsive'])

    ax0.set_ylabel('cells')
    ax0.set_xticks(range(2), labels=['resp', 'unresp'], rotation=90)
    ax0.set_ylim([0,265])

    ax1.set_ylim([0,265])
    ax1.set_yticks(np.linspace(0,250,6))
    ax0.set_yticks(np.linspace(0,250,6))
        
    ax1.bar(0, dark_fracs[0], color='lightblue')
    ax1.bar(1, dark_fracs[1], color=colors['unresponsive'])
    ax1.set_ylabel('cells')
    ax1.set_xticks(range(2), labels=['resp', 'unresp'], rotation=90)

    fig.tight_layout()
    fig.savefig(os.path.join(figpath, 'light_dark_resonsive_bar.pdf'))


    fig3A = plt.figure(constrained_layout=True, figsize=(4,3), dpi=300)
    fig3Aspec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig3A, wspace=0.05, hspace=0.1)

    ax_light_pref = fig3A.add_subplot(fig3Aspec[1,0])
    ax_light_nonpref = fig3A.add_subplot(fig3Aspec[1,1])
    ax_dark_pref = fig3A.add_subplot(fig3Aspec[0,0])
    ax_dark_nonpref = fig3A.add_subplot(fig3Aspec[0,1])

    names = ['pref_gazeshift_psth','nonpref_gazeshift_psth','pref_dark_gazeshift_psth','nonpref_dark_gazeshift_psth']
    panels = [ax_light_pref, ax_light_nonpref, ax_dark_pref, ax_dark_nonpref]
    use_cells = ltdk[ltdk['dark_responsive']].copy() # [ltdk['FmLt_gaze_responsive']]

    ax_light_pref.set_title('Light pref.')
    ax_light_nonpref.set_title('Light nonpref.')
    ax_dark_pref.set_title('Dark pref.')
    ax_dark_nonpref.set_title('Dark nonpref.')

    for count, name in enumerate(names):
        panel = panels[count]
        
        for x, p in enumerate(use_cells[name]):
            panel.plot(psth_bins, p, '-', linewidth=1, alpha=.75, color=colors[use_cells['gazecluster'].iloc[x]])
        # panel.plot(psth_bins, np.nanmedian(flatten_series(use_cells['pref_gazeshift_psth']),0), '-', linewidth=2, color='k')
        panel.set_xlim([-0.2,0.4])
        panel.vlines(0,-1,1, color='k',linestyle='dashed',linewidth=1)
        panel.set_xticks(np.linspace(-0.2,0.4,4), labels=np.linspace(-200,400,4).astype(int))
        panel.set_ylim([-.6,1])
        if count==0 or count==1:
            panel.set_xticklabels([])
        else:
            panel.set_xlabel('time (msec)')
            
        if count==1 or count==3:
            panel.set_yticklabels([])
        else:
            panel.set_ylabel('norm. spike rate')
        # if name=='early':
        #     lpanel.set_ylabel('norm. spike rate')
        # if name!='early':
        #     lpanel.set_yticklabels([])
        #     dpanel.set_yticklabels([])
        # lpanel.set_xticklabels([])
        # dpanel.set_xlabel('time (msec)')
            
    fig3A.savefig(os.path.join(figpath, 'S2_dark_responsive.pdf'))