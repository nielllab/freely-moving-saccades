import os, json, pickle
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

import fmEphys

def make_recording(new_dir):
    """
    new_dir: list of filepaths to new directories
    """

    ### Create session dataframe

    subdirs = fmEphys.utils.path.list_subdirs(new_dir, givepath=True)
    usable_recordings = ['fm1','fm1_dark','fm_dark','hf1_wn',
        'hf2_sparsenoiseflash','hf3_gratings','hf4_revchecker']
    subdirs = [p for p in subdirs if any(s in p for s in usable_recordings)]

    df = pd.DataFrame([])
    for path in subdirs:
        ephys_path = fmEphys.utils.path.find('*_ephys_props.h5',path)[0]
        rec_data = pd.read_hdf(ephys_path)
        rec_type = '_'.join(([col for col in rec_data.columns.values if 'contrast_tuning_bins' in col][0]).split('_')[:-3])
        rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT',
                                            'spikeTraw':rec_type+'_spikeTraw',
                                            'rate':rec_type+'_rate',
                                            'n_spikes':rec_type+'_n_spikes'})
    # Get column names
    column_names = list(df.columns.values) + list(rec_data.columns.values)
    # New columns for same unit within a session
    df = pd.concat([df, rec_data],axis=1,ignore_index=True)
    # Add the list of column names from all sessions plus the current recording
    df.columns = column_names
    # Remove duplicate columns (i.e. shared metadata)
    df = df.loc[:,~df.columns.duplicated()]

    ### Read in the camera calibration values

    ellipse_json_path = find('*fm_eyecameracalc_props.json', new_dir)[0]
    
    with open(ellipse_json_path) as f:
        ellipse_fit_params = json.load(f)

    df['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
    df['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']

    df['original_session_path'] = new_dir
    df['probe_name'] = probe

    df['index'] = df.index.values
    df.reset_index(inplace=True)

    ### Fix gratings spatial frequencies
    if fix_grat:
        for ind, row in df.iterrows():
            tuning = row['Gt_ori_tuning_tf'].copy().astype(float)
            new_tuning = np.roll(tuning, 1, axis=1)
            df.at[ind, 'Gt_ori_tuning_tf'] = new_tuning.astype(object)



def add_recording():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', type=str, default=None)
    parser.add_argument('--new', type=str, default=None)
    args = parser.parse_args()

    new_dir = '/home/niell_lab/Data/freely_moving_ephys/ephys_recordings/032122/J599LT' # new animal directory
    existing_pickle = '/home/niell_lab/Data/freely_moving_ephys/batch_files/051322/ltdk_051322.pickle'



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

if __name__ == '__main__':

    

    main()