"""
fmSaccades/utils/marmo_panels.py

Panels for marmoset figures.

Functions
---------
mRaster : plot raster
plot_tempseq : plot temporal sequence



"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size':10})


def mRaster(ax, rast, n=500):
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
    # ax.set_xticklabels(np.linspace(-200,400,4).astype(int))

def plot_tempseq(ax, tseq, return_img=False, freev=None):
    """ Plot temporal sequence
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot to.
    tseq : np.ndarray
        The temporal sequence to plot (n_cells x n_timepoints).
    return_img : bool
        Whether to return the image object. Needs to be True if
        you want to use the colorbar.
    freev : float
        The value to use for the vmin and vmax of the colorbar.
        If it is None, the vmin and vmax will be set to -0.75
        and 0.75.

    Returns
    -------
    img : matplotlib.image.AxesImage
        The image object. Only returned if return_img is True.

    """

    # Set dynamic range
    vmin = -0.75
    vmax = 0.75
    if freev is not None:
        vmin = -freev
        vmax = freev

    # Drop first and last timepoints (weird values)
    tseq[:,:5] = np.nan
    tseq[:,-5:] = np.nan

    # Plot the sequence
    img = ax.imshow(tseq, cmap='coolwarm',
                    vmin=vmin, vmax=vmax)
    
    # Vertical line at t=0
    ax.vlines(200, 0, np.size(tseq,0),
              color='k', linestyle='dashed', linewidth=1)
    
    # Axis labels, ticks, etc.
    ax.set_xlabel('msec')
    ax.set_xlim([0,601])
    ax.set_xticks(np.linspace(0,601,4),
                  labels=np.linspace(-200,400,4).astype(int))
    
    ax.set_yticks(np.arange(0, np.size(tseq,0), 100))
    ax.set_ylim([np.size(tseq,0),0])

    ax.set_aspect(2.8)

    if return_img:
        return img