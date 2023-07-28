
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import fmEphys as fme
import saccadeAnalysis as sacc


def figS3(data, good_inds, savepath):


    sacc.set_plt_params()
    props = sacc.propsdict()
    colors= props['colors']
    psth_bins = props['psth_bins']


    train_psth = np.zeros([len(data.index.values), 2001])
    test_psth = np.zeros([len(data.index.values), 2001])
    print('num cells = {}'.format(len(data.index.values)))

    for i, ind in tqdm(enumerate(data.index.values)):

        if data.loc[ind, 'pref_gazeshift_direction']=='left':
            fullT = data.loc[ind, 'FmLt_gazeshift_left_saccTimes_dHead1'].copy().astype(float)
        
        elif data.loc[ind, 'pref_gazeshift_direction']=='right':
            fullT = data.loc[ind, 'FmLt_gazeshift_right_saccTimes_dHead1'].copy().astype(float)
        
        else:
            print(data.loc[ind, 'pref_gazeshift_direction'])
        
        train_inds = np.random.choice(np.arange(0, fullT.size),
                                      size=int(np.floor(fullT.size/2)),
                                      replace=False)
        
        test_inds = np.arange(0, fullT.size)
        test_inds = np.delete(test_inds, train_inds)

        train = fullT[train_inds].copy()
        test = fullT[test_inds].copy()
        
        spikeT = data.loc[ind,'FmLt_spikeT']
        
        train_psth[i,:] = fme.calc_PSTH(spikeT, train)
        test_psth[i,:] = fme.calc_PSTH(spikeT, test)

    np.save(os.path.join(savepath, 'train_psth.npy'), train_psth)
    np.save(os.path.join(savepath, 'test_psth.npy'), test_psth)

    norm_train = np.zeros([len(good_inds),2001])
    norm_test = np.zeros([len(good_inds),2001])

    for i, ind in enumerate(good_inds):
        norm_train[i,:] = sacc.norm_PSTH(train_psth[ind,:])
        norm_test[i,:] = sacc.norm_PSTH(test_psth[ind,:])

    psth_bins = np.arange(-1,1.001,1/1000)

    train_peakT = np.zeros(np.size(norm_train,0))
    test_peakT = np.zeros(np.size(norm_test,0))
    for i in range(np.size(norm_train,0)):
        train_peakT[i], _ = sacc.calc_PSTH_latency(norm_train[i,:])
        test_peakT[i], _ = sacc.calc_PSTH_latency(norm_test[i,:])

    # sort peak times
    order = np.argsort(train_peakT)

    sort_train_psths = norm_train[order,:].copy()
    sort_test_psths = norm_test[order,:].copy()


    fig, [ax0,ax1] = plt.subplots(1,2,figsize=(4,4), dpi=300)

    ax0_img = sacc.plot_PSTH_heatmap(ax0, sort_train_psths)

    ax1_img = sacc.plot_PSTH_heatmap(ax1, sort_test_psths)
    ax1.set_yticklabels([])

    ax0.set_title('train')
    ax1.set_title('test')

    # fig.savefig('/home/niell_lab/Desktop/crossval.pdf')

    plt.figure(figsize=(2,2), dpi=300)

    plt.plot(train_peakT[(train_peakT>.025) * (train_peakT<.250)],
            test_peakT[(test_peakT>.025) * (test_peakT<.250)], 'k.', markersize=3)
    
    plt.xlabel('train latency (msec)')
    plt.ylabel('test latency (msecs)')
    plt.plot([0.02,.250], [0.02,.250], linestyle='dashed', color='tab:red', linewidth=1)
    plt.xlim([.02, .20]); plt.ylim([.02, .250])
    plt.xticks(np.linspace(0.020,.250,4),
               labels=np.linspace(20,250,4).astype(int))
    plt.yticks(np.linspace(0.020,.250,4),
               labels=np.linspace(20,250,4).astype(int))

    maxcc = np.zeros([len(sort_train_psths)])*np.nan
    for i in range(len(sort_train_psths)):
        
        train = sort_train_psths[i,:].copy()
        test = sort_test_psths[i,:].copy()
        
        r = np.corrcoef(train[1000:1250], test[1000:1250])
        maxcc[i] = r[0,1]**2

    # cross valudation correlation histogram
    fig, ax0 = plt.subplots(1,1,figsize=(2.5,1.5), dpi=300)

    weights = np.ones_like(maxcc) / float(len(maxcc))
    n,_,_ = ax0.hist(maxcc, color='grey', bins=np.linspace(-1,1,21), weights=weights)
    # ax0.set_xlabel('gaze shift cc');
    ax0.set_ylabel('frac. cells')
    ax0.set_xticks(np.arange(-1,1,3),labels=[])
    ax0.plot([0,0], [0, .22], color='k', linewidth=1, linestyle='dashed')
    ax0.set_ylim([0,.22])

    fig.tight_layout()
    fig.savefig('mouse_crossval_gazeshift_correlation.pdf')