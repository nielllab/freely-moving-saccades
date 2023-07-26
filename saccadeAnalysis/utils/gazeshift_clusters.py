"""
saccadeAnalysis/utils/gazeshift_clusters.py

Functions
---------
make_cluster_model_input
    Make the inputs for the clustering model.
make_clusters
    Create the clustering models and fit to data.
add_labels_to_dataset
    Add the cluster labels to the dataset object.
apply_saved_cluster_models
    Apply saved clustering models to novel data.


Written by DMM, 2022
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import sklearn.cluster
import sklearn.decomposition

import fmEphys as fme
import saccadeAnalysis as sacc


def make_cluster_model_input(data):
    """ Make the inputs for the clustering model.

    Parameters
    ----------
    data : DataFrame
        Dataset object.
    
    Returns
    -------
    pca_input : array
        Array of data to be clustered with seeded clusters.
    """

    for i, ind in enumerate(data.index.values):


        if data.loc[ind,'gazeshift_responsive']==True:
            
            # Copy PSTH for responsive cells.
            _psth = data.loc[ind, 'pref_gazeshift_psth'].copy().astype(object)
            data.at[ind, 'pref_gazeshift_psth_for_kmeans'] = _psth


        elif data.loc[ind, 'gazeshift_responsive']==False:

            # Use unresponsive cells to seed an unresponsive cluster by setting the
            # PSTH to all zeros for the PCA input.
            data.at[ind,'pref_gazeshift_psth_for_kmeans'] = np.zeros([2001]).astype(object)
        
    pca_input = fme.flatten_series(data['pref_gazeshift_psth_for_kmeans'])
    pca_input = pca_input[:, 950:1300]

    return pca_input


def make_clusters(pca_input, model_savepath):
    """ Create the clustering models and fit to data.

    Parameters
    ----------
    pca_input : array
        Array of data to be clustered.
    model_savepath : str
        Filepath to save the models.
    
    Returns
    -------
    labels : array
        Array of cluster labels.
    _opt_outputs : dict
        Dictionary of optional outputs that may be useful
        as diagnostics.
    """

    n_pcas = 10
    n_clusters = 5

    # PCA
    pca = sklearn.decomposition.PCA(n_components=n_pcas)
    pca.fit(pca_input)

    # Project the data into PC space
    proj = pca.transform(pca_input)

    # How many PCs does it take to explain 95% variance?
    explvar = pca.explained_variance_ratio_
    keep_pcas = int(np.argwhere(np.cumsum(explvar)>.95)[0])
    print('Keeping {} PCs which explain 95% of variance'.format(keep_pcas))

    # Only keep that number of PCs.
    gproj = proj[:,:keep_pcas]

    # Apply k-means clustering
    km = sklearn.cluster.KMeans(n_clusters=n_clusters)
    km.fit_predict(gproj)

    # Get the k-means labels
    labels = km.labels_

    # Save the models
    km_model_path = os.path.join(model_savepath,
                                 'Gazeshift_KMeans_model_{}.pickle'.format(fme.fmt_now(c=True)))
    with open(km_model_path, 'wb') as f:
        pickle.dump(km, f)

    pca_model_path = os.path.join(model_savepath,
                                 'Gazeshift_PCA_model_{}.pickle'.format(fme.fmt_now(c=True)))
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    
    # Save the projection of cells into the PC space.
    pca_proj_path = os.path.join(model_savepath,
                                 'Gazeshift_PCA_projection_{}.npz'.format(fme.fmt_now(c=True)))
    np.savez(file=pca_proj_path,
             proj=proj, gproj=gproj, labels=labels,
             pca_input=pca_input, explvar=explvar, keep_pcas=keep_pcas)

    _opt_outputs = {
        'proj': proj,
        'gproj': gproj,
        'explvar': explvar,
        'keep_pcas': keep_pcas
    }

    return labels, _opt_outputs


def add_labels_to_dataset(data, labels):
    """ Add the cluster labels to the dataset object.

    Parameters
    ----------
    data : DataFrame
        Dataset object.
    labels : array
        Array of cluster labels.
    
    Returns
    -------
    data : DataFrame
        Updated dataset object with cluster labels added.
    """

    data['gazecluster_ind'] = -1

    for i, ind in enumerate(data.index.values):
        data.at[ind, 'gazecluster_ind'] = labels[i]


    # Label the clusters based on waveform.
    fig, axs = plt.subplots(2,3,figsize=(10,8))
    axs = axs.flatten()

    psth_bins = np.arange(-1,1.001,1/1000)

    for n, name in enumerate(range(-1,5)):
        
        plotvals = data['pref_gazeshift_psth'][data['gazecluster_ind']==name]
        
        if len(plotvals.index.values)==0:
            continue

        cluster = fme.flatten_series(plotvals)

        for i in range(np.size(cluster,0)):
            axs[n].plot(psth_bins, cluster[i,:], alpha=0.5)
        
        axs[n].plot(psth_bins, np.median(cluster,0), 'k-', linewidth=3)
        
        axs[n].set_title('{} (N={})'.format(name, len(plotvals)))
        axs[n].set_xlim([-0.3,0.3])
        axs[n].set_ylim([-1.5,1])

    fig.tight_layout()
    fig.show()

    # Get user-input to assign labels
    name_opts = ['early','late','biphasic','negative','unresponsive']

    k_to_name = dict(zip(name_opts, list(np.zeros(6)*np.nan)))

    for _name in name_opts:
        
        # Get manual label
        manlab = sg.popup_get_text('Which index is k= {} ?'.format(_name),
                                title='Cluster labeling')
        
        # Save label
        k_to_name[_name] = int(manlab)

    # Assign labels with str names
    for i, ind in enumerate(data.index.values):

        # Index label from clustering
        _l = data.loc[ind,'gazecluster_ind']

        # Name from user-generated dict
        
        _n = fme.invert_dict(k_to_name)[_l]

        data.at[ind, 'gazecluster'] = _n

    return data


def apply_saved_cluster_models(pca_input, km_path, pca_path):
    """ Apply saved clustering models to novel data.

    Parameters
    ----------
    pca_input : array
        Array of data to be clustered.
    km_path : str
        Path to the saved k-means model.
    pca_path : str
        Path to the saved PCA model.

    Returns
    -------
    labels : array
        Array of cluster labels.
    _opt_outputs : dict
        Dictionary of optional outputs that may be useful
        as diagnostics.
    """
    
    keep_pcas = 4

    with open(km_path, 'rb') as f:
        km = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    # Transform into PC space.
    proj = pca.transform(pca_input)

    # Only keep required PCs
    gproj = proj[:,:keep_pcas]

    # Map onto k-means clusters.
    labels = km.predict(gproj)

    _opt_outputs = {
        'proj': proj,
        'gproj': gproj,
    }

    return labels, _opt_outputs

