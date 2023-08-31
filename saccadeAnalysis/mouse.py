"""

Written by DMM, 2022
"""


import os
from tqdm import tqdm
import cv2
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

import fmEphys as fme
import saccadeAnalysis as sacc


def main(dataset_dict):
    """
    
    Parameters
    ----------
    dataset_dict : dict
        A dictionary of paths to datasets saved as h5 files.
        
    """

    # Set plt parameters
    sacc.set_plt_params()

    # Get plotting properties
    props = sacc.propsdict()
    colors = props['colors']
    psth_bins = props['psth_bins']
    psth_bins_long = props['psth_bins_long']

    hffm_path = dataset_dict['hffm']
    ltdk_path = dataset_dict['ltdk']
    savepath = dataset_dict['savepath']

    # load data
    print('Reading HfFm data...')
    hffm = fme.read_group_h5(hffm_path)

    print('Reading LtDk data...')
    ltdk = fme.read_group_h5(ltdk_path)

    # Set up a subdirectory for the figures to be saved to.
    figpath = os.path.join(savepath, 'figures')
    if not os.path.exists(figpath):
        os.mkdir(figpath)


if __name__ == '__main__':
    main()