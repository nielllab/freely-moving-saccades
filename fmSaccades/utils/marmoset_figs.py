

import numpy as np
import matplotlib.pyplot as plt

import fmSaccades as fmsacc

fmsacc.set_plt_params()

def spike_raster(ax, rast, n=500):

    usetrials = np.array(sorted(np.random.choice(np.arange(0,
                                        int(np.max(rast[:,1]))),
                                        n,
                                        replace=False)))
    
    for row, tnum in enumerate(usetrials):
        sps = rast.copy()
        sps = sps[sps[:,1].astype(int)==tnum]
        sps = sps[:,0]
        
        ax.plot(sps, np.ones(sps.size)*row, '|', color='k', markersize=0.3)
        
    ax.set_xlim([-0.2,0.4])
    ax.set_ylim([n, 0])
    ax.set_yticks(np.linspace(0,n,3))
    
    ax.set_xticks(np.linspace(-.2,.4,4))
    ax.set_xticklabels([])


def plot_tempseq(panel, tseq, return_img=False, freev=None):
    # tseq = drop_nan_along(tseq, 1)
    panel.set_xlabel('msec')
    panel.set_ylim([np.size(tseq,0),0])
    vmin = -0.75; vmax = 0.75
    if freev is not None:
        vmin = -freev
        vmax = freev
    tseq[:,:5] = np.nan
    tseq[:,-5:] = np.nan
    img = panel.imshow(tseq, cmap='coolwarm', vmin=vmin, vmax=vmax)
    panel.set_xlim([0,601])
    panel.set_yticks(np.arange(0, np.size(train_tseq,0),100))
    panel.set_xticks(np.linspace(0,601,4), labels=np.linspace(-200,400,4).astype(int))
    panel.vlines(200, 0, np.size(tseq,0), color='k', linestyle='dashed', linewidth=1)
    panel.set_aspect(2.8)
    if return_img:
        return img


def fig6B():
    """
    Example marmoset neurons
    """

    
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

        psth = raw_sacc[uNum,:].copy()
        psth_bins = np.arange(-200,401,1)
        psth[:15] = np.nan
        psth[-15:] = np.nan
        axs[1,uPos].plot(psth_bins, psth, 'k-')
        axs[1,uPos].vlines(0,0,np.max(psth)*1.1, 'k', linestyle='dashed',linewidth=1)

        if uPos==0:
            axs[1,uPos].set_ylabel('sp/s')
        axs[1,uPos].set_xticks(np.linspace(-200,400,4))
        axs[1,uPos].set_xlim([-200,400])
        axs[1,uPos].set_xticklabels(np.linspace(-200,400,4).astype(int))
        axs[1,uPos].set_xlabel('time (ms)')
        axs[1,uPos].set_ylim([0, np.nanmax(psth)*1.01])

        fig.tight_layout()

        fig.savefig(os.path.join(savepath, '6_example_cells_081022.pdf'))


def marmo_cross_val_latency_scatter():
    """
    figure S3D
    """

    plt.figure(figsize=(2,2), dpi=300)
    plt.plot(cv_peakT[:,0], cv_peakT[:,1], 'k.', markersize=3)
    plt.xlabel('train latency (ms)'); plt.ylabel('test latency (ms)')
    plt.plot([20,250], [20,250], linestyle='dashed', color='tab:red', linewidth=1)
    # plt.xlim([20,250]); plt.ylim([20,250])
    plt.xticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))
    plt.yticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))
    # plt.savefig('/home/niell_lab/Desktop/marmoset_latency_crossval.pdf', pad_inches=3)


def marmo_cross_val_seq():
    """
    figure S3C
    """
    
    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(4,4), dpi=300)

    ax0_img = plot_tempseq(ax0, train_tseq)

    ax1_img = plot_tempseq(ax1, test_tseq)
    ax1.set_yticklabels([])

    ax0.set_title('train')
    ax1.set_title('test')

    fig.savefig('/home/niell_lab/Desktop/marmoset_crossval.pdf', pad_inches=3)


def plot_marmo_clusters():

    cmap = fmsacc.make_colors()

    
fig, axs = plt.subplots(2,2,figsize=(4,3.25), dpi=300)
for ki, k in enumerate(kord):
    ax = axs.flatten()[ki]
    inds = np.argwhere(clusters==k).flatten()
    for ind in inds:
        psth = norm_sacc_psth[ind,:]
        ax.plot(psth_bins, psth, alpha=0.2)
    ax.plot(psth_bins, np.nanmean(norm_sacc_psth[inds,:], axis=0), color=color_list[ki], linewidth=2)
    ax.set_ylim([-0.8,1])
    ax.vlines(0, -1, 1, color='k', linestyle='dashed')
    ax.set_xticks(np.linspace(-200,400,4))
    ax.set_xlim([-200,400])
    if ki<2:
        ax.set_xticklabels([])
    if ki==0 or ki==2:
        ax.set_ylabel('norm sp/s')
    if ki>1:
        ax.set_xlabel('time (ms)')
fig.tight_layout()
fig.savefig(os.path.join(savepath, '6_clusters_sq.pdf'))