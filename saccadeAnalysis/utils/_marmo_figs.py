import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size':10})

import saccadeAnalysis as fms

# Create colormap
plasma_map = plt.cm.plasma(np.linspace(0,1,15))
colors = {
    'early': plasma_map[10,:],
    'late': plasma_map[8,:],
    'biphasic': plasma_map[5,:],
    'negative': plasma_map[2,:],
    'unresponsive': 'dimgrey',
    'gaze': 'firebrick',
    'comp': 'mediumblue',
    'rc': 'indigo'
}

def fig6B_rasters(totdata, raw_sacc, example_units, ops):

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

        fms.mRaster(axs[0,uPos], sacim_dict['StimRast2'], 500)
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

    fig.tight_layout()

    fig.savefig(os.path.join(ops['savedir'],
                             'fig6B_marmo_rasters.pdf'))
    
def fig6C_


def figS3C_D_crossval(cv_psth, ops):

    cv_norm_psth = np.zeros(cv_psth.shape)
    cv_peakT = np.zeros([334,2])
    for u in range(334):
        for x in range(2):
            n = normalize_psth(cv_psth[u,x,:].copy())
            cv_norm_psth[u,x,:] = n
            cv_peakT[u,x] = np.argmax(n[220:450])+20


    ### Scatter plot of response latency for train and test sets
    fig, ax = plt.subplots(1, 1, figsize=(2,2), dpi=300)

    ax.plot(cv_peakT[:,0], cv_peakT[:,1], 'k.', markersize=3)
    ax.plot([20,250], [20,250], linestyle='dashed', color='tab:red', linewidth=1)

    ax.set_xlabel('train latency (ms)')
    ax.set_ylabel('test latency (ms)')
    ax.set_xticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))
    ax.set_yticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))

    fig.tight_layout()
    fig.savefig(os.path.join(ops['savedir'],
                             'figS3D_marmo_latency_crossval.pdf'))
    
    # Sort PSTHs by latency
    latency_sort = np.argsort(cv_peakT[:,0].flatten().copy())
    train_tseq = cv_norm_psth[:,0,:].copy()[latency_sort]
    test_tseq = cv_norm_psth[:,1,:].copy()[latency_sort]

    ### Plot PSTHs sorted by latency for train and test sets
    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(4,4), dpi=300)

    ax1_img = plot_tempseq(ax0, train_tseq)

    ax2_img = plot_tempseq(ax1, test_tseq)
    ax2.set_yticklabels([])

    ax1.set_title('train')
    ax2.set_title('test')

    fig.tight_layout()
    fig.savefig(os.path.join(ops['savedir'],
                             'figS3C_marmo_crossval.pdf'))
    
