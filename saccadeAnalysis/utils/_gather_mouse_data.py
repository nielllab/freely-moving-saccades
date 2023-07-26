

import os, sys, json, pickle
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

import fmEphys as fme
import saccadeAnalysis as sacc

def main(savepath, h5_path=None, session_dict=None):
    """
    Don't need to provide an h5 path. you can include dict of {session_name: [list of h5 paths]}
    and they will be stacked into a single dataset
    """

    sacc.set_plt_params()
    cmap = sacc.make_colors()


    if h5_path is not None:
        data = fme.read_group_h5(h5_path)

        if savepath is None:
            savepath = os.path.split(h5_path)

    elif session_dict is not None:

        data = sacc.stack_dataset(session_dict)

        _savepath_h5 = os.path.join(savepath,
                        'fmSaccades_dataset_{}.h5'.format(fme.fmt_now(c=True)))
        fme.write_group_h5(data, _savepath_h5)

    # Fix gratings spatial frequencies
    # for ind, row in df.iterrows():
    #     tuning = row['Gt_ori_tuning_tf'].copy().astype(float)
    #     new_tuning = np.roll(tuning, 1, axis=1)
    #     df.at[ind, 'Gt_ori_tuning_tf'] = new_tuning.astype(object)

    saccthresh = {
        'head_moved': 60,  # in deg/sec
        'gaze_stationary': 120,
        'gaze_moved': 240
    }

    # Cluster out the putative excitatory and inhibitory cells
    # based on spike waveform.
    data = sacc.putative_cell_type(data)



tuning = data['Gt_ori_tuning_tf'].iloc[0].copy()
tf = 1
plt.plot(np.arange(8)*45, tuning[:,0,tf], label='low sf')
plt.plot(np.arange(8)*45, tuning[:,1,tf], label='mid sf')
plt.plot(np.arange(8)*45, tuning[:,2,tf], label='high sf')
plt.legend()

### FmLt
for ind, row in data.iterrows():
    pref, nonpref, prefname, nonprefname = get_direction_pref(row['FmLt_gazeshift_left_saccPSTH_dHead1'], row['FmLt_gazeshift_right_saccPSTH_dHead1'])
    data.at[ind,'pref_gazeshift_direction'] = prefname
    data.at[ind,'nonpref_gazeshift_direction'] = nonprefname
    data.at[ind,'gazeshift_DSI'] = calc_psth_DSI(pref, nonpref)
    # norm gaze shifts
    data.at[ind, 'pref_gazeshift_psth'] = normalize_psth(pref).astype(object)
    data.at[ind, 'nonpref_gazeshift_psth'] = normalize_psth(nonpref, raw_pref=pref).astype(object)
    # raw gaze shifts and comp
    data.at[ind, 'pref_gazeshift_psth_raw'] = pref.copy().astype(object)
    data.at[ind, 'nonpref_gazeshift_psth_raw'] = nonpref.copy().astype(object)
    # compensatory
    data.at[ind, 'pref_comp_psth_raw'] = row['FmLt_comp_{}_saccPSTH_dHead1'.format(prefname)].copy().astype(object)
    data.at[ind, 'nonpref_comp_psth_raw'] = row['FmLt_comp_{}_saccPSTH_dHead1'.format(nonprefname)].copy().astype(object)
    # raw comp
    data.at[ind, 'pref_comp_psth'] = normalize_psth(row['FmLt_comp_{}_saccPSTH_dHead1'.format(prefname)], raw_pref=pref).astype(object)
    data.at[ind, 'nonpref_comp_psth'] = normalize_psth(row['FmLt_comp_{}_saccPSTH_dHead1'.format(nonprefname)], raw_pref=pref).astype(object)
### Hf
for ind, row in data.iterrows():
    # reversing checkerboard
    data.at[ind, 'norm_Rc_psth'] = normalize_psth(row['Rc_psth']).astype(object)
    # gratings
    data.at[ind, 'norm_gratings_psth'] = normalize_gt_psth(row['Gt_grating_psth']).astype(object)
    # sparse noise
    data.at[ind, 'norm_Sn_psth'] = normalize_psth(row['Sn_on_background_psth'], baseline_val=row['Sn_on_background_psth'][1000]).astype(object)
### FmDk
for ind, row in data.iterrows():
    pref = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(row['pref_gazeshift_direction'])]
    # gaze shifts
    data.at[ind, 'pref_dark_gazeshift_psth'] = normalize_psth(pref).astype(object)
    data.at[ind, 'nonpref_dark_gazeshift_psth'] = normalize_psth(row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(row['nonpref_gazeshift_direction'])], raw_pref=pref).astype(object)
    # compensatory
    data.at[ind, 'pref_dark_comp_psth'] = normalize_psth(row['FmDk_comp_{}_saccPSTH_dHead1'.format(row['pref_gazeshift_direction'])], raw_pref=pref).astype(object)
    data.at[ind, 'nonpref_dark_comp_psth'] = normalize_psth(row['FmDk_comp_{}_saccPSTH_dHead1'.format(row['nonpref_gazeshift_direction'])], raw_pref=pref).astype(object)
    # raw gaze shifts
    data.at[ind, 'pref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(row['pref_gazeshift_direction'])].astype(object)
    data.at[ind, 'nonpref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(row['nonpref_gazeshift_direction'])].astype(object)
    # compensatory
    data.at[ind, 'pref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead1'.format(row['pref_gazeshift_direction'])].astype(object)
    data.at[ind, 'nonpref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead1'.format(row['nonpref_gazeshift_direction'])].astype(object)
## Peak time
psth_bins = np.arange(-1,1.001,1/1000)
### FmLt
for ind, row in data.iterrows():
    raw_psth = row['pref_gazeshift_psth_raw']
    norm_psth = row['pref_gazeshift_psth']
    
    peakT, peak_val = calc_latency(norm_psth)
    
    data.at[ind, 'FmLt_gazeshift_baseline'] = raw_psth[0:800].astype(object)
    data.at[ind, 'FmLt_gazeshift_med_baseline'] = np.median(raw_psth[0:800])
    data.at[ind, 'FmLt_gazeshift_peak_val'] = peak_val
    data.at[ind, 'FmLt_gazeshift_peakT'] = peakT

# for ind, row in data.iterrows():
#     if row['FmLt_gazeshift_peakT']<0.033:
#         data.at[ind, 'movement'] = True
### FmDk
for ind, row in data.iterrows():
    raw_psth = row['pref_dark_gazeshift_psth_raw']
    norm_psth = row['pref_dark_gazeshift_psth']
    
    peakT, peak_val = calc_latency(norm_psth)
    
    data.at[ind, 'FmDk_gazeshift_baseline'] = raw_psth[0:800].astype(object)
    data.at[ind, 'FmDk_gazeshift_med_baseline'] = np.median(raw_psth[0:800])
    data.at[ind, 'FmDk_gazeshift_peak_val'] = peak_val
    data.at[ind, 'FmDk_gazeshift_peakT'] = peakT
## Clustering
### Eliminate unresponsive cells before clustering
for ind, row in data.iterrows():
    # firing rate
    sec = row['FmLt_eyeT'][-1].astype(float) - row['FmLt_eyeT'][0].astype(float)
    sp = len(row['FmLt_spikeT'])
    fm_fr = sp/sec
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
        
pca_input = flatten_series(data['pref_gazeshift_psth_for_kmeans'])[:,950:1300] # [data['gazeshift_responsive']]
pca_input.shape
### kmeans
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
Z = km.labels_

with open('/home/niell_lab/Desktop/dEye_PSTH_km_model1A-new.pickle', 'wb') as f:
    pickle.dump(km, f)
with open('/home/niell_lab/Desktop/dEye_PSTH_pca_model1A1-new.pickle', 'wb') as f:
    pickle.dump(pca, f)
np.save(file='/home/niell_lab/Desktop/dEye_PSTH_pca1-new.npy', arr=proj)
plt.scatter(proj[:,0], proj[:,1], c=Z)
# plt.scatter(proj[~use,0], proj[~use,1], c='k')
### Add to the df
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
plt.figure(figsize=(3,4))
plt.plot(psth_bins, np.median(flatten_series(data['pref_gazeshift_psth']),0))
plt.ylim([-1.5,1])
plt.xlim([-.3,.3])
### For ltdk data
keep_pcas = 4

with open('/home/niell_lab/Desktop/dEye_PSTH_km_model1A-new.pickle', 'rb') as f:
    km = pickle.load(f)
with open('/home/niell_lab/Desktop/dEye_PSTH_pca_model1A1-new.pickle', 'rb') as f:
    pca = pickle.load(f)

proj = pca.transform(pca_input)

gproj = proj[:,:keep_pcas]

Z = km.predict(gproj)

data['gazecluster_ind'] = -1
for i, ind in enumerate(data.index.values): # [data['gazeshift_responsive']==True]
    data.at[ind, 'gazecluster_ind'] = Z[i]
plt.subplots(2,3,figsize=(10,8))
for n, name in enumerate(range(0,5)):
    plt.subplot(2,3,n+1)
    plotvals = data['pref_gazeshift_psth'][data['gazeshift_responsive']][data['gazecluster_ind']==name] # pref_gazeshift_psth
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




## gratings
for ind, row in data.iterrows():
    sec = row['Gt_eyeT'][-1].astype(float) - row['Gt_eyeT'][0].astype(float)
    sp = len(row['Gt_spikeT'])
    data.at[ind, 'Gt_fr'] = sp/sec
    
    data.at[ind, 'raw_mod_for_Gt'] = gt_modind(row['Gt_grating_psth'])
    
    data.at[ind, 'norm_mod_for_Gt'] = gt_modind(row['norm_gratings_psth'])
    
    # psth = row['Gt_grating_psth']
    # baseline = np.median(psth[1:5].copy())
    # peak = np.median(psth[5:14].copy())
    # data.at[ind, 'raw_mod_for_Gt'] = peak - baseline
    
data['Gt_responsive'] = False
for ind, row in data.iterrows():
    if (row['raw_mod_for_Gt']>1) and (row['norm_mod_for_Gt']>0.1):
        data.at[ind, 'Gt_responsive'] = True

print(data['Gt_responsive'].sum())
print(data['Gt_responsive'].sum()/len(data.index.values))
### a few visualizations
from scipy import stats
def running_median(panel, x, y, n_bins=7):
    bins = np.linspace(np.min(x), np.max(x), n_bins)
    bin_means, bin_edges, bin_number = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.median, bins=bins)
    bin_std, _, _ = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=bins)
    hist, _ = np.histogram(x[~np.isnan(x) & ~np.isnan(y)], bins=bins)
    tuning_err = bin_std / np.sqrt(hist)
    panel.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2), bin_means, '-', color='k')
    panel.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2), bin_means-tuning_err, bin_means+tuning_err, color='k', alpha=0.2)
plotdata = data[data['gazeshift_responsive']][data['Gt_responsive']].copy()

fig, ax = plt.subplots(1,1, dpi=200, figsize=(3,2))
cmaps = [cat_cmap['early'],cat_cmap['late'],cat_cmap['biphasic'],cat_cmap['negative']]
for ki, k in enumerate([0,1,4,3]):
    
    sf_prefs = plotdata['sf_pref_cpd'][~pd.isnull(plotdata['sf_pref_cpd'])][~pd.isnull(plotdata['FmLt_gazeshift_peakT'])][plotdata['gazecluster_ind']==k].copy().to_numpy()
    peak_times = plotdata['FmLt_gazeshift_peakT'][~pd.isnull(plotdata['sf_pref_cpd'])][~pd.isnull(plotdata['FmLt_gazeshift_peakT'])][plotdata['gazecluster_ind']==k].copy().to_numpy()
    
    ax.scatter(peak_times, sf_prefs, color=cmaps[ki], s=3)
    
sf_prefs = plotdata['sf_pref_cpd'][plotdata['gazecluster_ind']!=2][~pd.isnull(plotdata['sf_pref_cpd'])][~pd.isnull(plotdata['FmLt_gazeshift_peakT'])].copy().to_numpy()
peak_times = plotdata['FmLt_gazeshift_peakT'][plotdata['gazecluster_ind']!=2][~pd.isnull(plotdata['sf_pref_cpd'])][~pd.isnull(plotdata['FmLt_gazeshift_peakT'])].copy().to_numpy()
    
running_median(ax, peak_times, sf_prefs, n_bins=7)

# all_peakT = []
# for i,x in enumerate(plotdata['sf_pref_cpd'][plotdata['gazecluster_ind']!=3]):
#     all_peakT.append(psth_bins[np.argmax(x[startwin:endwin])+startwin])
# all_peakT = np.array(all_peakT)
    
# sf_prefs = plotdata['sf_pref_cpd'][plotdata['gazecluster_ind']!=1].copy().to_numpy()

ax.set_ylabel('spatial freq (cpd)')
ax.set_xlabel('peak latency (sec)')

# ax.set_xlim([0,.175])
ax.set_xticks(np.linspace(0, 0.170, 7), labels=np.linspace(0, 170, 7).astype(int))

ax.set_ylim([0,0.2])

ax.vlines(0.033,0,.32,'k',linestyle='dashed', linewidth=1)
# plt.hlines(.33,0.015, .2,'k',linestyle='dashed', linewidth=1)

startwin = 1015
endwin = 1200
print(endwin-startwin)
plotdata = data[data['use']].copy()#[data1['use']].copy() #[data1['session']!='032022_J599LT_control_Rig2'].copy() # 
fig, ax = plt.subplots(1,1, dpi=300, figsize=(3,2))
cmaps = [cat_cmap['early'],cat_cmap['late'],cat_cmap['biphasic'],cat_cmap['negative']]
for ki, k in enumerate([0,2,4,1]):
    sf_prefs = plotdata['gazeshift_dirselind'][plotdata['gazecluster_ind']==k].copy().to_numpy()
    for i,x in enumerate(plotdata['pref_gazeshift_psth'][plotdata['gazecluster_ind']==k]): # gazecluster_ind
        peakT = psth_bins[np.argmax(x[startwin:endwin])+startwin]

        ax.plot(peakT, sf_prefs[i], '.', color=cmaps[ki], markersize=3)
all_peakT = []
for i,x in enumerate(plotdata['pref_gazeshift_psth'][plotdata['gazecluster_ind']!=3]):
    all_peakT.append(psth_bins[np.argmax(x[startwin:endwin])+startwin])
all_peakT = np.array(all_peakT)
    
sf_prefs = plotdata['gazeshift_dirselind'][plotdata['gazecluster_ind']!=3].copy().to_numpy()

all_peakT = all_peakT[~np.isnan(sf_prefs)]
sf_prefs = sf_prefs[~np.isnan(sf_prefs)]

running_median(ax, all_peakT, sf_prefs, n_bins=14)
# ax.set_ylim([0,0.2])
ax.set_ylabel('gaze direction selectivity')
ax.set_xlabel('peak latency (sec)')
plt.vlines(0.035,0,.8,'k',linestyle='dashed', linewidth=1)
plt.hlines(.33,0.015,.2,'k',linestyle='dashed', linewidth=1)

# dark modulation
for ind, row in data.iterrows():
    norm_psth = row['pref_dark_gazeshift_psth'].copy().astype(float)
    data.at[ind, 'norm_dark_modulation'] = psth_modind(norm_psth)
    
    raw_psth = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])].copy().astype(float)
    data.at[ind, 'dark_modulation'] = psth_modind(raw_psth)
cmaps = [cat_cmap['early'],cat_cmap['late'],cat_cmap['biphasic'],cat_cmap['negative']]

plotvals = data[data['gazeshift_responsive']][data['gazecluster_ind']!=4].copy()

for ind in plotvals.index.values:
    plt.plot(plotvals.loc[ind,'FmDk_gazeshift_peakT'], plotvals.loc[ind,'gazeshift_dirselind'], '.', color=cmaps[plotvals.loc[ind,'gazecluster_ind']])
plt.xlabel('peakT'); plt.ylabel('direction selectivity')
plt.vlines(.033, 0, 1, 'k')
plt.hlines(0, 0.010, .170, 'k')
movement_inds = data[data['movcluster1']=='movement'].index.values
ind = 292
row = data.loc[ind].copy()

fig, axs = plt.subplots(2,2,figsize=(3,3),dpi=200)

light_pref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
light_nonpref = row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]

dark_pref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['pref_gazeshift_direction'])]
dark_nonpref = row['FmDk_gazeshift_{}_saccPSTH_dHead'.format(row['nonpref_gazeshift_direction'])]

axs[0,0].plot(psth_bins, light_pref, 'k')
axs[0,1].plot(psth_bins, light_nonpref, 'k')
axs[1,0].plot(psth_bins, dark_pref, 'k')
axs[1,1].plot(psth_bins, dark_nonpref, 'k')

axs[0,0].set_title('light pref')
axs[0,1].set_title('light non-pref')
axs[1,0].set_title('dark pref')
axs[1,1].set_title('dark non-pref')

for i in range(2):
    for j in range(2):
        axs[i,j].set_xlim([-0.3,0.3])
        axs[i,j].set_ylim([0, np.max([light_pref, light_nonpref, dark_pref, dark_nonpref])*1.1])

fig.tight_layout()
cmaps = [cat_cmap['early'],cat_cmap['late'],cat_cmap['biphasic'],cat_cmap['negative']]

data['dark_responsive'] = False

for ki, k in enumerate([0,4,1,2]):
    
    plotvals = data[data['gazeshift_responsive']][data['gazecluster_ind']==k].copy()

    fig, [[ax0,ax1,ax2,ax3],[ax4,ax5,ax6,ax7]] = plt.subplots(2,4,figsize=(10,5), dpi=300)

    for ind in plotvals.index.values:
        row = plotvals.loc[ind].copy()
        
        fmt='.'; sz=2.5
        
        if (row['dark_modulation']>1) and (row['norm_dark_modulation']>0.1) and (row['FmDk_gazeshift_peakT']<=0.035): # (row['gazeshift_dirselind']>=.33) and
            data.at[ind, 'dark_responsive'] = True
            ax0.annotate(str(ind), xy=[row['dark_modulation'], row['norm_dark_modulation']], fontsize=6)
            ax1.annotate(str(ind), xy=[row['dark_modulation'], row['FmDk_gazeshift_peakT']], fontsize=6)
            ax2.annotate(str(ind), xy=[row['dark_modulation'], row['gazeshift_dirselind']], fontsize=6)
            
            ax3.annotate(str(ind), xy=[row['norm_dark_modulation'], row['FmDk_gazeshift_peakT']], fontsize=6)
            ax4.annotate(str(ind), xy=[row['norm_dark_modulation'], row['gazeshift_dirselind']], fontsize=6)
            
            ax5.annotate(str(ind), xy=[row['FmDk_gazeshift_peakT'], row['gazeshift_dirselind']], fontsize=6)
            
            ax6.plot(psth_bins, row['pref_dark_gazeshift_psth'], alpha=0.5)
            
            c='g'
            
        else:
            c=cmaps[ki]
            
        ax0.plot(row['dark_modulation'], row['norm_dark_modulation'],
                fmt, color=c, markersize=sz)
        ax1.plot(row['dark_modulation'], row['FmDk_gazeshift_peakT'],
                fmt, color=c, markersize=sz)
        ax2.plot(row['dark_modulation'], row['gazeshift_dirselind'],
                fmt, color=c, markersize=sz)
        ax3.plot(row['norm_dark_modulation'], row['FmDk_gazeshift_peakT'],
                fmt, color=c, markersize=sz)
        ax4.plot(row['norm_dark_modulation'], row['gazeshift_dirselind'],
                fmt, color=c, markersize=sz)
        ax5.plot(row['FmDk_gazeshift_peakT'], row['gazeshift_dirselind'],
                fmt, color=c, markersize=sz)
        
        
            

    ax0.set_xlabel('modulation (sp/sec)'); ax0.set_ylabel('norm. modulation')
    
    ax1.set_xlabel('modulation (sp/sec)'); ax1.set_ylabel('latency')
    
    ax2.set_xlabel('modulation (sp/sec)'); ax2.set_ylabel('DSI')
    
    ax3.set_xlabel('norm. modulation'); ax3.set_ylabel('latency')
    
    ax4.set_xlabel('norm. modulation'); ax4.set_ylabel('DSI')
    
    ax5.set_xlabel('latency'); ax5.set_ylabel('DSI')

    # (row['gazeshift_dirselind']>.33)
    # [plotvals['gazeshift_dirselind']>.33]
    
    kname = ['early','late','biphasic','negative'][ki]
    
    use_for_med = plotvals['pref_dark_gazeshift_psth'][plotvals['dark_modulation']>1][plotvals['norm_dark_modulation']>0.1][data['FmDk_gazeshift_peakT']<=0.035]#[plotvals['gazeshift_dirselind']>=.33]
    if len(use_for_med.index.values)>1:
        all_psth = flatten_series(use_for_med)
        ax6.plot(psth_bins, np.median(all_psth,0), 'k')
    ax6.set_title('dark pref. gaze shift')

    n_dark_resp = len(use_for_med.index.values)
    ax6.set_xlim([-.3,.3])
    
    ax0.set_title('{} n={}/{}'.format(kname, n_dark_resp, len(plotvals.index.values)))
    
    ax7.axis('off')
    
    fig.tight_layout()

    fig_label = 'modFr_lowLatency'
    
    fig.savefig('/home/niell_lab/Desktop/{}_{}_.png'.format(kname, fig_label))
    
# ax; ax.set_zlabel('')
# ax.set_xlim([0,17])
# ax.view_init(45, 80)
plotvals = data['pref_dark_gazeshift_psth'][data['norm_dark_modulation']>0.1][data['dark_modulation']>1].copy()

fig, [ax0, ax1] = plt.subplots(2,1,figsize=(3,4), dpi=300)

ax0.plot(psth_bins, np.median(flatten_series(plotvals[plotvals['gazecluster_ind']!=3]),0), label='dark mod')

ax1.plot(psth_bins, np.median(flatten_series(plotvals[plotvals['gazecluster_ind']!=3][plotvals['gazeshift_dirselind']>.33]),0), label='dark mod & 2:1 DSI')

for ind in plotvals[plotvals['norm_dark_modulation']>0.1][plotvals['dark_modulation']>0.1].index.values:
    plt.plot(psth_bins, plotvals.loc[ind, 'pref_gazeshift_psth'])
plt.xlim([-.3,.4])
for x in plotvals['pref_gazeshift_psth'][data['gazeshift_dirselind']>.6]:
    plt.plot(psth_bins, x)
plt.xlim([-.3,.4])
plt.vlines(0, 0, 1, 'k')
### Name clusters
np.min(np.median(flatten_series(data['pref_gazeshift_psth'][data['gazecluster_ind']==0]), axis=0))
np.min(np.median(flatten_series(data['pref_gazeshift_psth'][data['gazecluster_ind']==2]), axis=0))
for k in range(5):
    med_response = np.mean(flatten_series(data['pref_gazeshift_psth'][data['gazecluster_ind']==k]), axis=0)
    plt.plot(psth_bins, med_response, label=k)
plt.xlim(-0.2,0.4)
plt.vlines(0,-.5,.5 ,'k')
plt.legend()

def label_movcluster(psth, el_bound=0.08):
    """
    PSTH should be the neural response to eye movements
    between -0.0625 and 0.3125 sec, where 0 is the moment
    of the eye movement.
    """

    # find peaks and troughs in PSTH
    p, peak_props = find_peaks(psth, height=.30)
    t, trough_props = find_peaks(-psth, height=.20)

    # get the time index of the highest peaks
    if len(p) > 1:
        p = p[np.argmax(peak_props['peak_heights'])]
    if len(t) > 1:
        t = t[np.argmax(trough_props['peak_heights'])]
    if p.size == 0:
        p = np.nan
    if t.size == 0:
        t = np.nan
    if ~np.isnan(p):
        p = int(p)
    if ~np.isnan(t):
        t = int(t)

    # some filtering to choose the best position for the peak
    if ~np.isnan(p):
        has_peak = True
        peak_cent = p
    else:
        has_peak = False
        peak_cent = None
    if ~np.isnan(t):
        has_trough = True
        trough_cent = t
    else:
        has_trough = False
        trough_cent = None

    # now we decide which cluster each of these should be
    el_bound_ind = np.argmin(np.abs(psth_bins-el_bound))
    if has_peak and has_trough:
        return 'biphasic'
    elif has_trough and ~has_peak:
        return 'negative'
    elif peak_cent is not None and peak_cent <= el_bound_ind:
        return 'early'
    elif peak_cent is not None and peak_cent > el_bound_ind:
        return 'late'
    else:
        return 'unresponsive'
cluster_to_cell_type = {}
for l in range(5):
    med_response = np.median(flatten_series(data['pref_gazeshift_psth'][data['gazecluster_ind']==l]), axis=0)
    cluster_to_cell_type[l] = label_movcluster(med_response)
data['gazecluster'] = 'unresponsive'
for ind, row in data.iterrows():
    if row['gazeshift_responsive']:
        data.at[ind, 'gazecluster'] = cluster_to_cell_type[row['gazecluster_ind']]
data['gazecluster'].value_counts()
vc = data['gazecluster'].value_counts()
vc/np.sum(vc)
# Only for HFFM
## Gratings
for sf in ['low','mid','high']:
    data['norm_ori_tuning_'+sf] = data['Gt_ori_tuning_tf'].copy().astype(object)
for ind, row in data.iterrows():
    orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
    for sfnum in range(3):
        sf = ['low','mid','high'][sfnum]
        data.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
    mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
    mean_for_sf[mean_for_sf<0] = 0
    data.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)

data['osi_for_sf_pref'] = np.nan
data['dsi_for_sf_pref'] = np.nan
for ind, row in data.iterrows():
    if ~np.isnan(row['sf_pref']):
        best_sf_pref = int(np.round(row['sf_pref']))
        data.at[ind, 'osi_for_sf_pref'] = row[(['Gt_osi_low','Gt_osi_mid','Gt_osi_high'][best_sf_pref-1])]
        data.at[ind, 'dsi_for_sf_pref'] = row[(['Gt_dsi_low','Gt_dsi_mid','Gt_dsi_high'][best_sf_pref-1])]

data['osi_for_sf_pref'][data['osi_for_sf_pref']<0] = 0
data['dsi_for_sf_pref'][data['dsi_for_sf_pref']<0] = 0
for ind, row in data.iterrows():
    try:
        mean_for_sf = np.array([np.mean(data.at[ind,'norm_ori_tuning_low']), np.mean(data.at[ind,'norm_ori_tuning_mid']), np.mean(data.at[ind,'norm_ori_tuning_high'])])
        mean_for_sf[mean_for_sf<0] = 0
        data.at[ind, 'Gt_evoked_rate'] = np.max(mean_for_sf) - row['Gt_drift_spont']
    except:
        pass

for ind, row in data.iterrows():
    if type(row['Gt_ori_tuning_tf']) != float:
        tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
        tuning = tuning - row['Gt_drift_spont']
        tuning[tuning < 0] = 0
        mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
        tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
        data.at[ind, 'tf_pref'] = tf_pref

for ind, row in data.iterrows():
    tf = 2 + (6 * (row['tf_pref']-1))
    sf = 0.02 * 4 ** (row['sf_pref']-1)
    data.at[ind,'tf_pref_cps'] = tf
    data.at[ind,'sf_pref_cpd'] = sf
    data.at[ind,'grat_speed_dps'] = tf / sf


# Some light/dark calcs
model_dt = 0.025
for ind, row in data.iterrows():
    modelT = np.arange(0, np.nanmax(row['FmLt_eyeT']), model_dt)
    
    # timing is off sometimes... using eyeT instead of worldT to get maximum length
    # and they can be different by a few frames
    diff = len(modelT) - len(row['FmLt_rate'])
    if diff>0: # modelT is longer
        modelT = modelT[:-diff]
    elif diff<0: # modelT is shorted
        for i in range(np.abs(diff)):
            modelT = np.append(modelT, modelT[-1]+model_dt)
    model_gz = interp1d(row['FmLt_imuT'], row['FmLt_gyro_z'], bounds_error=False)(modelT)
    model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
    data.at[ind, 'FmLt_model_active'] = model_active.astype(object)
    data.at[ind, 'FmLt_modelT'] = modelT.astype(object)
    
    modelT = np.arange(0, np.nanmax(row['FmDk_eyeT']), model_dt)
    diff = len(modelT) - len(row['FmDk_rate'])
    if diff>0: # modelT is longer
        modelT = modelT[:-diff]
    elif diff<0: # modelT is shorted
        for i in range(np.abs(diff)):
            modelT = np.append(modelT, modelT[-1]+model_dt)
    model_gz = interp1d(row['FmDk_imuT'], row['FmDk_gyro_z'], bounds_error=False)(modelT)
    model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
    data.at[ind, 'FmDk_model_active'] = model_active.astype(object)
    data.at[ind, 'FmDk_modelT'] = modelT.astype(object)
data['FmLt_fr'] = ((data['FmLt_rate'].apply(np.sum)*0.025) / data['FmLt_eyeT'].apply(np.nanmax)).to_numpy()
data['FmDk_fr'] = ((data['FmDk_rate'].apply(np.sum)*0.025) / data['FmDk_eyeT'].apply(np.nanmax)).to_numpy()
FmLt_fr = np.zeros([len(data.index.values)])
FmDk_fr = np.zeros([len(data.index.values)])
for ind, row in data.iterrows():
    data.at[ind,'FmLt_active_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']>40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']>40])
    data.at[ind,'FmLt_inactive_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']<40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']<40])
    data.at[ind,'FmDk_active_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']>40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']>40])
    data.at[ind,'FmDk_inactive_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']<40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']<40])
## light/dark correlation
for ind, row in data[['pref_dark_gazeshift_psth','pref_gazeshift_psth']].iterrows():
    r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['pref_dark_gazeshift_psth'].astype(float)[1000:1250])
    data.at[ind, 'gaze_ltdk_maxcc'] = r[0,1]
# Head-fixed vs. gaze correlation
for ind, row in data[['norm_Rc_psth','norm_Sn_psth','pref_gazeshift_psth']].iterrows():
    if (np.sum(~np.isnan(row['norm_Rc_psth'].astype(float)[1000]))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
        r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Rc_psth'].astype(float)[1000:1250])
        data.at[ind, 'gaze_rc_maxcc'] = r[0,1]
    if (np.sum(~np.isnan(row['norm_Sn_psth'].astype(float)))>0) and (np.sum(~np.isnan(row['pref_gazeshift_psth'].astype(float)))>0):
        r = np.corrcoef(row['pref_gazeshift_psth'].astype(float)[1000:1250], row['norm_Sn_psth'].astype(float)[1000:1250])
        data.at[ind, 'gaze_sn_maxcc'] = r[0,1]
# Save the dataset as a pickle
data.to_pickle('/home/niell_lab/Data/freely_moving_ephys/batch_files/062022/ltdk_062022.pickle')


# Other
fig1 = plt.figure(constrained_layout=True, figsize=(9,7.6), dpi=300)
fig1spec = gridspec.GridSpec(nrows=5, ncols=3, figure=fig1, wspace=1.5, hspace=1.5)

fig1Cspec = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=fig1spec[0:2,1], wspace=0, hspace=0.01)
ax_theta = fig1.add_subplot(fig1Cspec[0,0])
ax_yaw = fig1.add_subplot(fig1Cspec[1,0])
ax_gaze = fig1.add_subplot(fig1Cspec[2,0])

fig1Dspec = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=fig1spec[0:2,2], wspace=0, hspace=0)
ax_dEyeHead = fig1.add_subplot(fig1Dspec[0,0])

fig1E2Fspec = gridspec.GridSpecFromSubplotSpec(3,3,subplot_spec=fig1spec[2:,0:2], wspace=0.15, hspace=-.05)

ax_pos_rasterG = fig1.add_subplot(fig1E2Fspec[0,0])
ax_biph_rasterG = fig1.add_subplot(fig1E2Fspec[0,1])
ax_neg_rasterG = fig1.add_subplot(fig1E2Fspec[0,2])

ax_pos_rasterC = fig1.add_subplot(fig1E2Fspec[1,0])
ax_biph_rasterC = fig1.add_subplot(fig1E2Fspec[1,1])
ax_neg_rasterC = fig1.add_subplot(fig1E2Fspec[1,2])

ax_pos_psth = fig1.add_subplot(fig1E2Fspec[2,0])
ax_biph_psth = fig1.add_subplot(fig1E2Fspec[2,1])
ax_neg_psth = fig1.add_subplot(fig1E2Fspec[2,2])

fig1Gspec = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=fig1spec[2:,2:], wspace=0.2, hspace=0.1)
ax_ex_gaze = fig1.add_subplot(fig1Gspec[0,0])
ax_ex_comp = fig1.add_subplot(fig1Gspec[1,0])

start = 2090 #2100
win = 60 # frames, not sec
ex_units = [171,112,126]
ex_units_direcprefs = ['left','left','right']

ylim_val = 36
theta_data = demo['FmLt_theta'][start:start+win]
theta_data = theta_data - np.nanmean(theta_data)
ax_theta.plot(theta_data, 'k-', linewidth=2, scaley=10)
ax_theta.set_xlim([0,60]); ax_theta.set_xticks(ticks=np.linspace(0,60,5), labels=np.linspace(0,1,5))
ax_theta.set_ylabel('theta (deg)')
ax_theta.set_ylim([-ylim_val,ylim_val])
ax_theta.axes.get_xaxis().set_visible(False)
ax_theta.axes.spines.bottom.set_visible(False)

pYaw = np.nancumsum(demo['FmLt_dHead'][start:start+win]*0.016)
pYaw = pYaw - np.nanmean(pYaw)
ax_yaw.plot(pYaw, 'k-', linewidth=2)
ax_yaw.set_xlim([0,60])
ax_yaw.set_xticks(ticks=np.linspace(0,60,5), labels=np.linspace(0,1,5))
ax_yaw.set_ylabel('yaw (deg)')
ax_yaw.axes.get_xaxis().set_visible(False)
ax_yaw.axes.spines.bottom.set_visible(False)
ax_yaw.set_ylim([-ylim_val,ylim_val])

ax_gaze.plot(pYaw + theta_data, 'k-', linewidth=2)
ax_gaze.set_xlim([0,60])
ax_gaze.set_xticks(ticks=np.linspace(0,60,5), labels=np.linspace(0,1000,5).astype(int))
ax_gaze.set_ylabel('gaze (deg)')
ax_gaze.set_ylim([-ylim_val,ylim_val])
ax_gaze.set_xlabel('time (msec)')

for i in plotinds:
    dGaze_i = np.abs(dHead_data[i]+dEye_data[i])
    if eyeT[i] in gazemovs:
        c = colors['gaze']
    elif eyeT[i] in comp:
        c = colors['comp']
    elif (np.abs(dHead_data[i])<60) or ((dGaze_i<240) and (dGaze_i>120)):
        c = 'dimgray'
    else:
        continue
    ax_dEyeHead.plot(dHead_data[i], dEye_data[i], '.', color=c, markersize=2)

ax_dEyeHead.set_aspect('equal','box')
ax_dEyeHead.set_xlim([-600,600])
ax_dEyeHead.set_ylim([-600,600])
ax_dEyeHead.set_xlabel('head velocity (deg/sec)')
ax_dEyeHead.set_ylabel('eye velocity (deg/sec)')
ax_dEyeHead.plot([-500,500],[500,-500], linestyle='dashed', color='k', linewidth=1)
# ax_dEyeHead.annotate('left', xy=[350,500], color='k')
# ax_dEyeHead.annotate('right', xy=[-550,-500], color='k')
# ax_dEyeHead.annotate('gaze shift', xy=[-620,470], color=colors['gaze'])
# ax_dEyeHead.annotate('compensated', xy=[-620,550], color=colors['comp'])
ax_dEyeHead.set_xticks(np.linspace(-600,600,5))
ax_dEyeHead.set_yticks(np.linspace(-600,600,5))

num_movements = 500
raster_panelsG = [ax_pos_rasterG, ax_biph_rasterG, ax_neg_rasterG]
raster_panelsC = [ax_pos_rasterC, ax_biph_rasterC, ax_neg_rasterC]
sdf_panels = [ax_pos_psth, ax_biph_psth, ax_neg_psth]
for i, u in enumerate(ex_units):
    row = hffm.iloc[u]
    rasterG = raster_panelsG[i]
    rasterC = raster_panelsC[i]
    sdf_panel = sdf_panels[i]
    LR = ex_units_direcprefs[i]
    
    rasterG.set_title(['positive','biphasic','negative'][i])

    gazeshifts = row['FmLt_gazeshift_{}_saccTimes_dHead'.format(LR)].copy()
    compmovs = np.hstack([row['FmLt_comp_left_saccTimes_dHead'], row['FmLt_comp_right_saccTimes_dHead']])
    
    plot_gs = np.random.choice(gazeshifts, size=num_movements, replace=False)
    plot_cp = np.random.choice(compmovs, size=num_movements, replace=False)

    for n, s in enumerate(plot_gs):
        sp = row['FmLt_spikeT']-s
        sp = sp[np.abs(sp)<=0.5]
        rasterG.plot(sp, np.ones(sp.size)*n, '|', color=colors['gaze'], markersize=0.25)

    for n, s in enumerate(plot_cp):
        sp = row['FmLt_spikeT']-s
        sp = sp[np.abs(sp)<=0.5]
        rasterC.plot(sp, np.ones(sp.size)*n, '|', color=colors['comp'], markersize=0.25)
    
    rasterG.set_ylim([num_movements, 0]); rasterC.set_ylim([num_movements,0])
    rasterG.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
    rasterC.vlines(0, 0, num_movements, color='k', linewidth=1, linestyle='dashed')
    if i == 0:
        rasterG.set_ylabel('gaze shifts'); rasterC.set_ylabel('compensatory')
        rasterG.set_yticks(np.linspace(0, 500, 3))
        rasterC.set_yticks(np.linspace(0, 500, 3))
    else:
        rasterG.set_yticks(np.linspace(0, 500, 3),labels=[])
        rasterC.set_yticks(np.linspace(0, 500, 3),labels=[])
    rasterG.set_xticks([]); rasterC.set_xticks([])
    rasterG.set_xlim([-.5,.5]); rasterC.set_xlim([-.5,.5])
    rasterG.axes.spines.bottom.set_visible(False); rasterC.axes.spines.bottom.set_visible(False)
    
    sdf_panel.plot(psth_bins, row['FmLt_comp_{}_saccPSTH_dHead'.format(LR)], color=colors['comp'])
    sdf_panel.plot(psth_bins, row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(LR)], color=colors['gaze'])
    max_fr = np.nanmax(row['FmLt_gazeshift_{}_saccPSTH_dHead'.format(LR)])*1.1
    sdf_panel.set_ylim([0,max_fr])
    sdf_panel.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
    sdf_panel.set_xlim([-.5,.5])
    if i == 0:
        sdf_panel.set_ylabel('spike rate (sp/sec)')
    sdf_panel.set_xlabel('time (msec)')
    sdf_panel.vlines(0, 0, max_fr, color='k', linewidth=1, linestyle='dashed')

possible_inds = hffm['pref_comp_psth'][hffm['fr']>2].index.values
np.random.seed(1)
example_inds = np.sort(np.random.choice(possible_inds, size=50, replace=False))

for ind in example_inds:
    ax_ex_gaze.plot(psth_bins, hffm.loc[ind,'pref_gazeshift_psth'].astype(float), linewidth=1, alpha=0.3)
    ax_ex_comp.plot(psth_bins, hffm.loc[ind,'pref_comp_psth'].astype(float), linewidth=1, alpha=0.3)
ax_ex_gaze.set_xlim([-.5,.5])
ax_ex_gaze.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
ax_ex_comp.set_xlim([-.5,.5])
ax_ex_comp.set_xticks(np.linspace(-.5,.5,3), labels=np.linspace(-500,500,3).astype(int))
ax_ex_gaze.set_ylim([-.7,1])
ax_ex_comp.set_ylim([-.7,1])
ax_ex_gaze.set_ylabel('norm. spike rate')
ax_ex_comp.set_ylabel('norm. spike rate')
ax_ex_comp.set_xlabel('time (msec)')
ax_ex_gaze.set_xlabel('time (msec)')

all_comp = flatten_series(hffm['pref_comp_psth'][hffm['gazecluster']!='unresponsive'][hffm['gazeshift_responsive']])
all_gaze = flatten_series(hffm['pref_gazeshift_psth'][hffm['gazecluster']!='unresponsive'][hffm['gazeshift_responsive']])

comp_mean = np.nanmedian(all_comp,0)
comp_std = np.std(all_comp,0) / np.sqrt(np.size(all_comp))

gaze_mean = np.nanmedian(all_gaze,0)
gaze_std = np.std(all_gaze,0) / np.sqrt(np.size(all_gaze))
ax_ex_comp.set_title('compensatory')
ax_ex_gaze.set_title('gaze-shifting')
ax_ex_comp.plot(psth_bins, comp_mean, color=colors['comp'], linewidth=3)
ax_ex_gaze.plot(psth_bins, gaze_mean, color=colors['gaze'], linewidth=3)

