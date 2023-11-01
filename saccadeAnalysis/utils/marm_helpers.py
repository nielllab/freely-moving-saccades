


import numpy as np


def marm_psth_modind(psth):

    psth = psth.astype(float)
    use = psth - np.mean(psth[0:150].copy())
    mod = np.max(np.abs(use[200:]))
    return mod


def mRaster(ax, rast, n=500):
    # rasters for example units

    usetrials = np.array(sorted(np.random.choice(np.arange(0,int(np.max(rast[:,1]))), n, replace=False)))
    
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


def m_plot_tempseq(panel, tseq, return_img=False, freev=None):
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
    

def marm_normalize_psth(psth):
    pref = psth.copy()
    bsln = np.mean(psth[0:150]) # was -100 to -50 ms ; now, -200 to -50 ms
    norm_psth = (psth - bsln) / np.max(pref[200:]) # 0 to 200
    return norm_psth