

import pandas as pd
import numpy as np

import fmEphys as fme
import fmSaccades as fms


def stderr(A):
    return np.std(A) / np.sqrt(len(A))


def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)


def add_stimuli_horizontally(df_list):

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


def add_sessions_vertically(df_list, savepath=None):
    
    dataset = pd.concat(df_list, axis=0)
    # Reset unit numbers
    dataset.reset_index(inplace=True, drop=True)

    if savepath is not None:
        fme.write_group_h5(dataset, savepath)

    return dataset

def stack_dataset(session_dict):
    """
    format should be {'SESSION NAME': [fm1_ephys_props.h5', 'wn_ephys_props.h5']}
    """

    df_list = []

    for skey, sval in session_dict.items():

        _s_dfs = []
        for s in sval:
            _df = pd.read_hdf(s)

            _s_dfs.append(_df)
        
        sdf = add_stimuli_horizontally(_s_dfs)

    dataset = add_sessions_vertically(df_list, savepath=None)
    
    return dataset
