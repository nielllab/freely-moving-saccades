


import os
from tqdm import tqdm

from scipy.io import loadmat

import numpy as np
import pandas as pd

import itertools
import scipy.stats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def main():



    psth_bins = np.arange(-200,401)

    # savepath = '/home/niell_lab/Desktop/'
    data = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_Final.mat')
    totdata = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_TotalInfo.mat')['TotalInfo']
    raw_sacc = np.load('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy')


    uNum = 0
    unitlabels = ['tagname','pathname','pathplot','isolation','depth','duration',
                'waveform','channel','shank','ISI','Hart','SacGrating','SacImage']
    unit_dict = dict(zip(unitlabels, list(totdata[uNum][0][0][0])))

    sacimlabels = ['EventList','StimWin','TrialType','StimRast','OriInd','StimTT',
                'StimUU','StimSU','OriInd2','StimRast2','StimTT2','StimUU2',
                'StimSU2','BaseMu','BaseMu2']
    sacim_dict = dict(zip(sacimlabels, list(unit_dict['SacImage'][0][0])))

    raw_sacc = np.load('/home/niell_lab/Desktop/marmoset_recalc_saccades.npy')






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


    ##### Recalculate PSTHS for all units #####






####### OPEN IN ARRAYS ###########

sacc_psth = data['ISACMOD2']
grat_psth = data['GSACMOD']
sf_tuning = data['SFTUNE']
tf_tuning = data['TFTUNE']
ori_tuning = data['ORTUNE']
bsln_fr = data['BASEMU2']
peakT = data['PEAKIM2']
animal = data['ANIMID']





n_cells = np.size(peakT,0)

norm_sacc_psth = np.zeros([n_cells, len(psth_bins)])
for ind in range(n_cells):
    norm_sacc_psth[ind,:] = normalize_psth(sacc_psth[ind].copy())





######   GRAT PREF #####



sacc_resp = np.zeros(n_cells)
mods = np.zeros(n_cells)
for ind in range(n_cells):
    mod = psth_modind(norm_sacc_psth[ind,:])
    mods[ind] = mod
    if mod > 0.1:
        sacc_resp[ind] = True


#### CLUSTERING ####


kord = [2,0,3,1]

color_list = [
    kcolors['early'],
          kcolors['late'],
          kcolors['biphasic'],
          kcolors['negative']
         ]


name_key_list = []
for i in [2,0,3,1]:
    name_key_list.append(['early','late','biphasic','negative'][i])
name_key_list
clusters = Z.copy()
for ki, k in enumerate(kord):
    print(names[ki], len(np.argwhere((clusters==k) * (grat_resp==1)).flatten()))
np.save('/home/niell_lab/Desktop/marmoset_clusters.npy', clusters)


### PLOT CLUSTERS ###







fig, ax = plt.subplots(1,1,figsize=(2.75,2), dpi=300)
for ki, k in enumerate(kord):
    inds = np.argwhere(clusters==k).flatten()
    for ind in inds:
        psth = norm_sacc_psth[ind,:]
        # ax.plot(psth_bins, psth, alpha=0.2)
    ax.plot(psth_bins, np.nanmean(norm_sacc_psth[inds,:], axis=0), color=color_list[ki], linewidth=2)
    ax.set_ylim([-.5,.7])
    ax.vlines(0, -1, 1, color='k', linestyle='dashed')
    
ax.set_xlim([-200,400])
ax.set_xticks(np.linspace(-200,400,4))
ax.set_xlabel('time (ms)')
ax.set_ylabel('norm sp/s')

fig.tight_layout()
fig.savefig(os.path.join(savepath, '6_all_clusters.pdf'))




fig, ax = plt.subplots(1,1,figsize=(2.5,2), dpi=300)
for ki, k in enumerate(kord):
    inds = np.argwhere(clusters==k).flatten()
    ax.scatter(proj[inds,0], proj[inds,1], s=1, c=color_list[ki])
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.axis('equal')
# ax.set_ylim([-3,5])
# ax.set_xlim([-6,8])

fig.savefig(os.path.join(savepath, '6_pca.pdf'))








def plot_cprop_scatter(ax, data, clusters, color_list):
    for ki, k in enumerate(kord):
        
        inds = np.argwhere(clusters==k).flatten()
        kdata = data[inds].copy()
        
        x_jitter = np.random.uniform(ki-0.2, ki+0.2, np.size(kdata,0))
        
        ax.plot(x_jitter, kdata, '.', color=color_list[ki], markersize=2)
        
        hline = np.nanmedian(kdata)
        err = np.std(kdata) / np.sqrt(np.size(kdata))
        
        ax.hlines(hline, ki-0.2, ki+0.2, color='k', linewidth=2)
        ax.vlines(ki, hline-err, hline+err, color='k', linewidth=2)
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(['early','late','biphasic','negative'])





def running_median(ax, x, y, n_bins=7):
    bins = np.linspace(np.min(x), np.max(x), n_bins)
    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.median, bins=bins)
    bin_std, _, _ = scipy.stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=bins)
    hist, _ = np.histogram(x[~np.isnan(x) & ~np.isnan(y)], bins=bins)
    tuning_err = bin_std / np.sqrt(hist)
    ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2), bin_means, '-', color='k')
    ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2), bin_means-tuning_err, bin_means+tuning_err, color='k', alpha=0.2)






grat_resp = np.zeros(n_cells)
for ind in range(n_cells):
    ffi = np.sqrt(ori_index[ind]**2 + sf_index[ind]**2)
    if ffi >= 0.2:
        grat_resp[ind] = True
grat_resp = grat_resp.astype(bool)



use = np.argwhere(grat_resp).flatten()

vcounts = np.unique(clusters[use], return_counts=True)
vcounts[1][kord]






use = np.argwhere((grat_resp) * (peakT.flatten()>25) * (peakT.flatten()<160)).flatten()

vcounts = np.unique(clusters[use], return_counts=True)
vcounts[1][kord]


######## TF and SF ######


combs = list(itertools.combinations(['early','late','biphasic','negative'], 2))

[['early','late','biphasic','negative'][k] for k in kord]

kname = 'negative'
kord[['early','late','biphasic','negative'].index(kname)]

pairwise_anova = {}

use = np.argwhere(grat_resp).flatten()

vals = []
for i, pair in enumerate(combs):
    
    n1 = kord[['early','late','biphasic','negative'].index(pair[0])]
    n2 = kord[['early','late','biphasic','negative'].index(pair[1])]
    
    v1 = sf_pref[use][clusters[use]==n1]
    v2 = sf_pref[use][clusters[use]==n2]
    v1 = v1[~np.isnan(v1)]
    v2 = v2[~np.isnan(v2)]
    
    f, p = scipy.stats.f_oneway(v1, v2)
    out = {'f':f, 'p':p}
    pairwise_anova['{}{} {:02f} vs {}{} {:02f}'.format(pair[0],n1,np.mean(v1),pair[1],n2,np.mean(v2))] = out
pairwise_anova







fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2,2, figsize=(6,4.5), dpi=300)

# use = np.argwhere((grat_resp)).flatten()
use = np.argwhere((grat_resp) * (peakT.flatten()>25) * (peakT.flatten()<160)).flatten() # 25; 175 ms

plot_cprop_scatter(ax0, tf_pref[use], clusters[use], color_list)
ax0.set_ylabel('temporal freq (cps)')

plot_cprop_scatter(ax2, sf_pref[use], clusters[use], color_list)
ax2.set_ylabel('spatial freq (cpd)')

for ki, k in enumerate(kord):
    inds = np.intersect1d(np.argwhere((clusters==k)).flatten(), use)
    
    ax1.plot(peakT.flatten()[inds], tf_pref[inds], '.', color=color_list[ki], markersize=3)
    ax3.plot(peakT.flatten()[inds], sf_pref[inds], '.', color=color_list[ki], markersize=3)
    
running_median(ax1, peakT.flatten()[use], tf_pref[use], n_bins=6)
running_median(ax3, peakT.flatten()[use], sf_pref[use], n_bins=6)

ax1.set_xlabel('saccade latency (ms)')
ax3.set_xlabel('saccade latency (ms)')
ax1.set_ylabel('temporal freq (cps)')
ax3.set_ylabel('spatial freq (cpd)')
    
ax1.set_xlim([25,175])
ax3.set_xlim([25,175])
ax0.set_ylim([0, 15]); ax1.set_ylim([0, 15])
ax2.set_ylim([1.5, 5]); ax3.set_ylim([1.5, 5])

fig.tight_layout()

# fig.savefig(os.path.join(savepath, '6_sf_tf_scatter.pdf'))





use = np.argwhere(grat_resp).flatten()

scipy.stats.spearmanr(peakT.flatten()[use], sf_pref[use])



use = np.argwhere((grat_resp) * (peakT.flatten()>25) * (peakT.flatten()<150)).flatten()
np.unique(clusters[use], return_counts=True)
latency_sort = np.argsort(peakT.flatten().copy())
tempseq = norm_sacc_psth.copy()[latency_sort]


def plot_tempseq(ax, seq):
    
    ax.set_xlabel('time (msec)')
    ax.set_ylim([np.size(seq,0),0])
    
    vmin = -0.75; vmax = 0.75
    img = ax.imshow(seq, cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    ax.set_xlim([0,601])
    
    ax.set_xticks(np.linspace(0,600,4))
    ax.set_xticklabels(labels=np.linspace(-200,400,4).astype(int))
    ax.vlines(200, 0, np.size(seq,0), color='k', linestyle='dashed', linewidth=1)
    
    ax.set_aspect(3)
    
    return img


fig, ax = plt.subplots(1,1, dpi=300, figsize=(5,5))
plot_tempseq(ax, tempseq)
ax.set_yticks(np.arange(0,np.size(tempseq,0),100))
fig.savefig(os.path.join(savepath, '6_tempseq.pdf'))



k_to_name = {
    0:'early',
    2:'late',
    3:'biphasic',
    1:'negative'
}




sorted_clusters = clusters.copy()[latency_sort]

tempseq_legend = np.zeros([n_cells, 1, 4])
for ki, k in enumerate(sorted_clusters):
    tempseq_legend[ki,:,:] = kcolors[k_to_name[k]]



fig, ax = plt.subplots(1,1,figsize=(0.5,1.5), dpi=300)
ax.imshow(tempseq_legend, aspect=.05)
ax.set_yticks([]); ax.set_xticks([])
ax.axes.spines.bottom.set_visible(False)
ax.axes.spines.right.set_visible(False)
ax.axes.spines.left.set_visible(False)
ax.axes.spines.top.set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(savepath, '6_tempseq_legend.pdf'))



fig, [ax_cellcounts, ax_baseline_fr] = plt.subplots(1,2, figsize=(5.5,2.5), dpi=300)

names = ['early','late','biphasic','negative']
print_names = ['early','late','biph','neg']

for ki, k in enumerate(kord):
    inds = np.argwhere((clusters==k)).flatten()
    ax_cellcounts.bar(ki, len(inds)/n_cells, color=color_list[ki])
    
ax_cellcounts.set_xticks(ticks=range(4))
ax_cellcounts.set_xticklabels(print_names, rotation=90)
ax_cellcounts.set_ylabel('frac. cells')

for ki, k in enumerate(kord):
    inds = np.argwhere((clusters==k)).flatten()
    bsln_plot_vals = bsln_fr[inds]
    
    err = np.nanstd(bsln_plot_vals) / np.sqrt(np.size(bsln_plot_vals))
    med = np.nanmedian(bsln_plot_vals)
    
    ax_baseline_fr.bar(ki, med, color=color_list[ki])
    ax_baseline_fr.plot([ki, ki], [med-err, med+err], 'k-')
    
ax_baseline_fr.set_xticks(range(4))
ax_baseline_fr.set_xticklabels(print_names, rotation=90)
ax_baseline_fr.set_ylabel('baseline (sp/s)')

fig.tight_layout()
fig.savefig(os.path.join(savepath, '6_ksize_bslnFr.pdf'))






df = pd.DataFrame(sf_pref, columns=['sf_pref'])
df['gazecluster_ind'] = clusters
df['animal'] = animal.T[0]

for i in df.index.values:
    df.at[i,'gazecluster'] = name_map_dict[df.loc[i,'gazecluster_ind']]

df['grating_responsive'] = grat_resp

df.to_json('/home/niell_lab/Desktop/marmoset_sf_label_responsive.json')

def add_jitter(ki, sz):
    return np.random.uniform(ki-0.2, ki+0.2, sz)

name_map_dict = dict(zip([2,0,3,1], ['early','late','biphasic','negative']))