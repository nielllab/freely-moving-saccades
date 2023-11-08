"""
saccadeAnalysis/utils/create_dataset.py

Functions
---------
add_stimuli_horizontally
    Add stimuli for a single session horizontally.
add_sessions_vertically
    Add sessions vertically.
stack_dataset
    Stack sessions into a single dataset.


Written by DMM, 2022
"""


import pandas as pd
import numpy as np
import sys

sys.path.insert(0,r'c:\Users\Niell Lab\Documents\GitHub')
import saccadeAnalysis as sacc
import fmEphys as fme

def add_stimuli_horizontally(df_list):
    """ Add stimuli for a single session horizontally.
    
    Parameters
    ----------
    df_list : list
        List of dataframes to add horizontally.
    
    Returns
    -------
    session : pandas.DataFrame
        Horizontally concatenated stimuli for the session.

    """

    session = pd.DataFrame([])
    for r in df_list:
        
        # get column names
        column_names = list(session.columns.values) + list(r.columns.values)
        
        # new columns for same unit within a session
        session = pd.concat([session, r], axis=1, ignore_index=True)
        
        # add the list of column names from all sessions plus the current recording
        session.columns = column_names
        
        # remove duplicate columns (i.e. shared metadata)
        session = session.loc[:, ~session.columns.duplicated()]
        
    return session


def add_sessions_vertically(df_list):
    """ Add sessions vertically.

    Parameters
    ----------
    df_list : list
        List of dataframes to add vertically.
    
    Returns
    -------
    dataset : pandas.DataFrame
        Vertically stacked dataset.
    
    """

    dataset = pd.concat(df_list, axis=0)

    # Reset unit numbers
    dataset.reset_index(inplace=True, drop=True)

    return dataset


def stack_dataset(session_dict, savepath=None):
    """ Stack sessions into a single dataset.

    Parameters
    ----------
    session_dict : dict
        Dictionary of sessions to stack. Keys are session names, values are lists of
        paths to the h5 files for each recording in that session, e.g.,
        {    'ONE SESSION NAME': [fm1_ephys_props.h5', 'wn_ephys_props.h5', ...],
         'ANOTHER SESSION NAME': [fm1_ephys_props.h5', 'wn_ephys_props.h5', ...]}
    savepath : str
        Path to save the dataset to. If None (which is the default), it will not be
        saved.
        
    Returns
    -------
    dataset : pandas.DataFrame
        Stacked dataset.

    """

    print('Stacking dataset.')

    df_list = []

    for skey, sval in session_dict.items():

        print('Adding session {}.'.format(skey))

        _s_dfs = []
        for s in sval:
            _df = pd.read_hdf(s)

            _s_dfs.append(_df)
        
        sdf = add_stimuli_horizontally(_s_dfs)

        df_list.append(sdf)

    dataset = add_sessions_vertically(df_list)

    dataset = fme.replace_xr_obj(dataset)

    if savepath is not None:
        print('Writing file to {}'.format(savepath))
        fme.write_group_h5(dataset, savepath)
    
    return dataset

