""" Minimal code required to cluster gaze shift response types.
saccadeAnalysis/utils/minimal_cluster.py


Written by DMM, Nov 2023
"""

import fmEphys as fme
import saccadeAnalysis as sacc

def apply_minimal_clustering(df_in, key, km_model=None, pca_model=None):

    df = df_in.copy

    for ind, row in df.iterrows():

        pref, _, prefname, _ = sacc.get_direction_pref(
            left = row['FmLt_gazeshift_leftPSTH'],
            right = row['FmLt_gazeshift_rightPSTH']
        )
        
        df.at[ind,'pref_gazeshift_direction'] = prefname

        df.at[ind, 'pref_gazeshift_psth'] = sacc.norm_PSTH(
            psth = pref,
            trange = 'fm'
        ).astype(object)

    # Determine gazeshift responsive cells
    for ind, row in df.iterrows():

        # firing rate
        sec = row['FmLt_eyeT'][-1].astype(float) - row['FmLt_eyeT'][0].astype(float)
        sp = len(row['FmLt_spikeT'])
        fm_fr = sp/sec
        df.at[ind, 'Fm_fr'] = fm_fr
        
        raw_psth = row['pref_gazeshift_psth_raw']
        df.at[ind, 'raw_mod_at_pref_peak'] = sacc.calc_PSTH_modind(raw_psth)
        
        norm_psth = row['pref_gazeshift_psth']
        df.at[ind, 'norm_mod_at_pref_peak'] = sacc.calc_PSTH_modind(norm_psth)

    df['gazeshift_responsive'] = False

    for ind, row in df.iterrows():

        if (row['raw_mod_at_pref_peak'] > 1) and (row['norm_mod_at_pref_peak'] > 0.1):
            df.at[ind, 'gazeshift_responsive'] = True

    # Make the inputs for the clustering model.
    pca_input = sacc.make_cluster_model_input(df)

    # Cluster on the data
    if km_model is None or pca_model is None:
        labels, _km_opts = sacc.make_clusters(pca_input)

    elif km_model is not None and pca_model is not None:
        # apply existing clustering
        labels, _km_opts = sacc.apply_saved_cluster_models(
            pca_input,
            km_model,
            pca_model
        )

    k_to_name = {
        3: 'early',
        2: 'late',
        1: 'biphasic',
        4: 'negative',
        0: 'unresponsive'
    }

    df['gazecluster_ind'] = -1

    for i, ind in enumerate(df.index.values):
        df.at[ind, 'gazecluster_ind'] = labels[i]

    for i, ind in enumerate(df.index.values):

        # Index label from clustering
        _l = df.loc[ind,'gazecluster_ind']

        # Name from user-generated dict
        
        _n = fme.invert_dict(k_to_name)[_l]

        df.at[ind, 'gazecluster'] = _n

    ser_out = df['gazecluster']

    return ser_out