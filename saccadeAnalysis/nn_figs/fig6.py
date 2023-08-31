


def fig6():
    
    example_units = [9, 30, 0]

    fig, axs = plt.subplots(2,3, figsize=(5.5,3), dpi=300)

    for uPos, uNum in enumerate(example_units):

        unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                    'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
        unit_dict = dict(zip(unitlabels, list(totdata[uNum][0][0][0])))

        sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                    'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                    'StimSU2','BaseMu','BaseMu2']
        sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

        mRaster(axs[0,uPos], sacim_dict['StimRast2'], 500)
        if uPos==0:
            axs[0,uPos].set_ylabel('gaze shifts')
        
        axs[0,uPos].set_title('cell {}'.format(uPos+1))

        # psth = data['ISACMOD2'][uNum]
        psth = raw_sacc[uNum,:].copy()
        psth_bins = np.arange(-200,401,1)
        psth[:15] = np.nan
        psth[-15:] = np.nan
        axs[1,uPos].plot(psth_bins, psth, 'k-')
        axs[1,uPos].vlines(0,0,np.max(psth)*1.1, 'k', linestyle='dashed',linewidth=1)
        
        # axs[1,uPos].plot(sacim_dict['StimTT'].flatten(), sacim_dict['StimUU'].flatten(), 'k-')
        # axs[1,uPos].vlines(0,0,np.max(sacim_dict['StimUU'].flatten())*1.1, 'k', linestyle='dashed',linewidth=1)
        if uPos==0:
            axs[1,uPos].set_ylabel('sp/s')
        axs[1,uPos].set_xticks(np.linspace(-200,400,4))
        axs[1,uPos].set_xlim([-200,400])
        axs[1,uPos].set_xticklabels(np.linspace(-200,400,4).astype(int))
        axs[1,uPos].set_xlabel('time (ms)')
        axs[1,uPos].set_ylim([0, np.nanmax(psth)*1.01])

        # axs[2,uPos].plot(hart_dict['SpatOris'].flatten(),
        #               np.array(hart_dict['otune'].flatten())[:int(np.size(hart_dict['SpatOris'].flatten()))],
        #               color='k')
        # axs[2,uPos].set_xticks(hart_dict['SpatOris'].flatten()[::2].astype(int))
        # axs[2,uPos].set_xlabel('orientation (deg)')
        # axs[2,uPos].set_ylabel('sp/s')

    fig.tight_layout()

    fig.savefig(os.path.join(savepath, '6_example_cells_081022.pdf'))