"""

"""

import pickle
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def normalize_PSTH(data):

    # For each cell
    for ind, row in data.iterrows():

        # Calculate firing rate firing rate
        sec = row['FmLt_eyeT'][-1].astype(float) - row['FmLt_eyeT'][0].astype(float)
        sp = len(row['FmLt_spikeT'])
        fm_fr = sp / sec
        data.at[ind, 'Fm_fr'] = fm_fr
        
        raw_psth = row['pref_gazeshift_psth_raw']
        data.at[ind, 'raw_mod_at_pref_peak'] = psth_modind(raw_psth)
        
        norm_psth = row['pref_gazeshift_psth']
        data.at[ind, 'norm_mod_at_pref_peak'] = psth_modind(norm_psth)

    data['gazeshift_responsive'] = False
    for ind, row in data.iterrows():
        if (row['raw_mod_at_pref_peak']>1) and (row['norm_mod_at_pref_peak']>0.1):
            data.at[ind, 'gazeshift_responsive'] = True
            # elif (row['FmLt_gazeshift_peakT']<.035):
            #     data.at[ind, 'movement_responsive'] = True
    print('gaze', data['gazeshift_responsive'].sum())



plt.subplots(5,5,figsize=(15,15))
plot_inds = np.random.randint(0, len(data.index.values), size=25)
for i, ind in enumerate(plot_inds):
    if data.loc[ind,'gazeshift_responsive']==True:
        color = 'b'
    else:
        color = 'r'
    psth = data.loc[ind, 'FmLt_gazeshift_{}_saccPSTH_dHead1'.format(data.loc[ind, 'pref_gazeshift_direction'])]
    plt.subplot(5,5,i+1)
    plt.plot(psth_bins, psth, color=color)
    plt.title('r={:.2f}, n={:.2f}'.format(data.loc[ind,'raw_mod_at_pref_peak'], data.loc[ind,'norm_mod_at_pref_peak']))
    plt.xlim([-.5,.5]); plt.ylim([0, np.max(psth)*1.2])
    
plt.tight_layout()

for i, ind in enumerate(data.index.values):
    if data.loc[ind,'gazeshift_responsive']==True:
        data.at[ind,'pref_gazeshift_psth_for_kmeans'] = data.loc[ind,'pref_gazeshift_psth'].copy().astype(object)
    elif data.loc[ind,'gazeshift_responsive']==False:
        data.at[ind,'pref_gazeshift_psth_for_kmeans'] = np.zeros([2001]).astype(object)
        


plt.scatter(proj[:,0], proj[:,1], c=Z)

data['gazecluster_ind'] = -1
for i, ind in enumerate(data.index.values): # [data['gazeshift_responsive']==True]
        data.at[ind, 'gazecluster_ind'] = Z[i]

plt.subplots(2,3,figsize=(10,8))
for n, name in enumerate(range(-1,5)):
    plt.subplot(2,3,n+1)
    plotvals = data['pref_gazeshift_psth'][data['gazecluster_ind']==name] # pref_gazeshift_psth
    if len(plotvals.index.values)==0:
        continue
    cluster = flatten_series(plotvals)
    for i in range(np.size(cluster,0)):
        plt.plot(psth_bins, cluster[i,:], alpha=0.5)
    plt.plot(psth_bins, np.median(cluster,0), 'k-', linewidth=3)
    plt.title('{} (N={})'.format(name, len(plotvals)))
    plt.xlim([-0.3,0.3])
    plt.ylim([-1.5,1])
plt.tight_layout()


def calc_clusters():

    
    pca_input = flatten_series(data['pref_gazeshift_psth_for_kmeans'])[:,950:1300] # [data['gazeshift_responsive']]
    pca_input.shape

    n_pcas = 10

    pca = PCA(n_components=n_pcas)
    pca.fit(pca_input)

    explvar = pca.explained_variance_ratio_

    proj = pca.transform(pca_input)

    keep_pcas = int(np.argwhere(np.cumsum(explvar)>.95)[0])
    print('using best {} PCs'.format(keep_pcas))

    gproj = proj[:,:keep_pcas]

    km = KMeans(n_clusters=5)
    km.fit_predict(gproj)
    cluster_inds = km.labels_

    fmE

    with open('/home/niell_lab/Desktop/dEye_PSTH_km_model1A-new.pickle', 'wb') as f:
        pickle.dump(km, f)
    with open('/home/niell_lab/Desktop/dEye_PSTH_pca_model1A1-new.pickle', 'wb') as f:
        pickle.dump(pca, f)
    np.save(file='/home/niell_lab/Desktop/dEye_PSTH_pca1-new.npy', arr=proj)

def apply_clusters(input_arr, km_path, pca_path, keep_pcas=4):
    """
    apply existing clustering to novel data
    use for dark experiments, or single recordings w/out enough cells to get good clustering from PCA

    '/home/niell_lab/Desktop/dEye_PSTH_km_model1A-new.pickle'
    '/home/niell_lab/Desktop/dEye_PSTH_pca_model1A1-new.pickle'

    input_arr is a numpy array with shape (n_cells, time) where time is in msec bins
    """

    with open(km_path, 'rb') as f:
        km_model = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca_model = pickle.load(f)

    proj = pca_model.transform(input_arr)

    gproj = proj[:,:keep_pcas]

    cluster_inds = km_model.predict(gproj)
    