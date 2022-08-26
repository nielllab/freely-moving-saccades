import os, sys, json, pickle
sys.path.insert(0, '/home/niell_lab/Documents/GitHub/FreelyMovingEphys/')
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from src.utils.path import list_subdirs, find
from src.utils.auxiliary import flatten_series
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

import fmEphys

def main():

    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    cat_cmap = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey'
    }
    colors = {
        'gaze': 'firebrick',
        'comp': 'mediumblue',
        'rc': 'indigo'
    }

    ### Set paths
    probe = 'DB_P128-D'
    new_dir = '/home/niell_lab/Data/freely_moving_ephys/ephys_recordings/032122/J599LT' # new animal directory
    existing_pickle = '/home/niell_lab/Data/freely_moving_ephys/batch_files/051322/ltdk_051322.pickle'

    ###