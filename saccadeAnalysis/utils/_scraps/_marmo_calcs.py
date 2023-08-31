
def normalize_psth(psth):
    pref = psth.copy()
    bsln = np.mean(psth[0:150]) # was -100 to -50 ms ; now, -200 to -50 ms
    norm_psth = (psth - bsln) / np.max(pref[200:]) # 0 to 200
    return norm_psth

    
def recalc_saccades():
    
    unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']

    sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                'StimSU2','BaseMu','BaseMu2']

    crossval = {}

    for uPos, uNum in tqdm(enumerate(range(601))):
        
        unit_dict = dict(zip(unitlabels, list(totunits[uNum][0][0][0])))

        sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

        rast = sacim_dict['StimRast2'].copy()

        trials = np.unique(rast[:,1]).astype(int)-1

        train_inds = np.array(sorted(np.random.choice(trials, size=int(np.floor(trials.size/2)), replace=False)))
        test_inds = trials.copy()
        test_inds = np.delete(test_inds, train_inds)
        
        prop_dict = {}

        for tt, tname in enumerate(['train', 'test']):
            
            inds = [train_inds, test_inds][tt]
        
            all_sps = []
            for tnum in inds:
                sps = rast.copy()
                sps = sps[sps[:,1].astype(int)==int(tnum)]
                sps = sps[:,0]
                all_sps.extend(sps)

            # PSTH
            psth = calc_PSTH(np.array(all_sps), np.zeros(inds.size), bandwidth=5, win=400)
            psth = psth[200:800]
            norm_psth = normalize_psth(psth)

            # latency
            peakT = np.argmax(norm_psth)-200 # in ms since onset of eye movement\
            
            prop_dict['{}_rawPsth'.format(tname)] = psth
            prop_dict['{}_normPsth'.format(tname)] = norm_psth
            prop_dict['{}_peakT'.format(tname)] = peakT
            
        crossval[uNum] = prop_dict


def calc_SF_TF():
    
# spatial and temporal frequencies used
sf_vals = np.array([1,2,4,8,16])
tf_vals = (60/16) * np.arange(9)

tf_pref = np.zeros(n_cells); sf_pref = np.zeros(n_cells)
ori_index = np.zeros(n_cells); tf_index = np.zeros(n_cells); sf_index = np.zeros(n_cells);

for ind in range(n_cells):
    sf = sf_tuning[ind,:].copy()
    tf = tf_tuning[ind,:].copy()
    ori = ori_tuning[ind,:].copy()
    
    ofi = np.nanstd(ori) / np.nanmean(ori)
    sfi = np.nanstd(sf) / np.nanmean(sf)
    tfi = np.nanstd(tf) / np.nanmean(tf)
    
    svec = sf.copy()-1
    svec[svec<0] = 0
    svec = svec**2
    spref = np.nansum(svec * sf_vals) / np.nansum(svec)
    
    tvec = tf.copy()-1
    tvec[tvec<0] = 0
    tvec = tvec**2
    tpref = np.nansum(tvec * tf_vals) / np.nansum(tvec)
    
    sf_pref[ind] = spref
    tf_pref[ind] = tpref
    ori_index[ind] = ofi
    sf_index[ind] = sfi
    tf_index[ind] = tfi


def cluster_marmo():

    
    

    pca_input = norm_sacc_psth.copy()

    n_pcas = 10
    pca = PCA(n_components=n_pcas)
    pca.fit(pca_input)

    explvar = pca.explained_variance_ratio_

    proj = pca.transform(pca_input)

    # keep_pcas = int(np.argwhere(np.cumsum(explvar)>.95)[0])
    # print('using best {} PCs'.format(keep_pcas))

    gproj = proj[:,:7]

    km = KMeans(n_clusters=4) ### using 4 clusters, not 5...
    km.fit_predict(gproj)
    Z = km.labels_

    clusters = Z.copy()

def load_marmo_clusters():
    Z = np.load('/home/niell_lab/Desktop/marmoset_clusters.npy')

    clusters = Z.copy()