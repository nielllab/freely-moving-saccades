import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
%matplotlib inline
mpl.rcParams.update({'font.size':10})
plasma_map = plt.cm.plasma(np.linspace(0,1,15))
kcolors = {
    'movement': plasma_map[12,:],
    'early': plasma_map[10,:],
    'late': plasma_map[8,:],
    'biphasic': plasma_map[5,:],
    'negative': plasma_map[2,:],
    'unresponsive': 'dimgrey'
}
tcolors = {
    'gaze': 'firebrick',
    'comp': 'mediumblue',
    'rc': 'indigo'
}
from tqdm import tqdm


def marmoset():

    psth_bins = np.arange(-200,401)
    base_path = r'C:\Users\dmartins\Documents\Dropbox\Research\Niell_lab\Data\Gaze_shift_data\Marmoset\062022'
    data = loadmat(os.path.join'Pooled_V1Hart_Preload_Final.mat')
    totdata = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_TotalInfo.mat')['TotalInfo']

    # Raster around single saccade examples
    uNum = 0
    unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
    unit_dict = dict(zip(unitlabels, list(totdata[uNum][0][0][0])))

    sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                'StimSU2','BaseMu','BaseMu2']
    sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))





    raw_sacc = np.load('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy')
    raw_sacc.shape


    # crossval

    cv_psth = np.load('/home/niell_lab/Desktop/crossval.npy')
    cv_psth.shape

    cv_norm_psth = np.zeros(cv_psth.shape)
    cv_peakT = np.zeros([334,2])
    for u in range(334):
        for x in range(2):
            n = normalize_psth(cv_psth[u,x,:].copy())
            cv_norm_psth[u,x,:] = n
            cv_peakT[u,x] = np.argmax(n[220:450])+20


    plt.figure(figsize=(2,2), dpi=300)
    plt.plot(cv_peakT[:,0], cv_peakT[:,1], 'k.', markersize=3)
    plt.xlabel('train latency (ms)'); plt.ylabel('test latency (ms)')
    plt.plot([20,250], [20,250], linestyle='dashed', color='tab:red', linewidth=1)
    # plt.xlim([20,250]); plt.ylim([20,250])
    plt.xticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))
    plt.yticks(np.linspace(20,250,4), labels=np.linspace(20,250,4).astype(int))
    # plt.savefig('/home/niell_lab/Desktop/marmoset_latency_crossval.pdf', pad_inches=3)



    from scipy.stats import spearmanr

    spearmanr(cv_peakT[:,0], cv_peakT[:,1])

    from scipy.stats import linregress

    res = linregress(cv_peakT[:,0], cv_peakT[:,1])

    latency_sort = np.argsort(cv_peakT[:,0].flatten().copy())
    train_tseq = cv_norm_psth[:,0,:].copy()[latency_sort]
    test_tseq = cv_norm_psth[:,1,:].copy()[latency_sort]


    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(4,4), dpi=300)

    ax0_img = plot_tempseq(ax0, train_tseq)

    ax1_img = plot_tempseq(ax1, test_tseq)
    ax1.set_yticklabels([])

    ax0.set_title('train')
    ax1.set_title('test')

    fig.savefig('/home/niell_lab/Desktop/marmoset_crossval.pdf', pad_inches=3)



    uNum = 10

    unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
    unit_dict = dict(zip(unitlabels, list(totunits[uNum][0][0][0])))

    sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                'StimSU2','BaseMu','BaseMu2']
    sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

    rast = sacim_dict['StimRast2'].copy()

    trials = np.unique(rast[:,1]).astype(int)-1

    train_inds = np.array(sorted(np.random.choice(trials, size=int(np.floor(trials.size/2)), replace=False)))
    test_inds = trials.copy()
    test_inds = np.delete(test_inds, train_inds)

    all_sps = []
    for tnum in train_inds:
        sps = rast.copy()
        sps = sps[sps[:,1].astype(int)==int(tnum)]
        sps = sps[:,0]
        all_sps.extend(sps)
        
    # PSTH
    psth = calc_PSTH(np.array(all_sps), np.zeros(train_inds.size), bandwidth=5, win=400)
    psth = psth[200:800]
    # normalize
    norm_psth = normalize_psth(psth)



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


    # open in arrays
    sacc_psth = data['ISACMOD2']
    grat_psth = data['GSACMOD']
    sf_tuning = data['SFTUNE']
    tf_tuning = data['TFTUNE']
    ori_tuning = data['ORTUNE']
    bsln_fr = data['BASEMU2']
    peakT = data['PEAKIM2']
    animal = data['ANIMID']



    