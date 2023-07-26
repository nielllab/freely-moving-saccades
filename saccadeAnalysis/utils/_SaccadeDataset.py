

import os
import pickle

import numpy as np
import pandas as pd
import scipy.signal

import sklearn.cluster
import sklearn.decomposition

import fmEphys as fme
import saccadeAnalysis as sacc

class SaccadeDataset():
    def __init__(self):


        self.psth_bins = np.arange(-1,1.001,1/1000)

    def create_dataset(self, stype=None):

        if stype is not None:
            self.stype = stype

        if self.stype=='hffm':
            

        if self.stype=='ltdk':


        subdirs = list_subdirs(new_dir, givepath=True)
        usable_recordings = ['fm1','fm1_dark','fm_dark','hf1_wn','hf2_sparsenoiseflash','hf3_gratings','hf4_revchecker']
        subdirs = [p for p in subdirs if any(s in p for s in usable_recordings)]

    def read_dataset(self, h5_path, stype):
        """
        stype should be 'ltdk', 'hffm', or 'flhf'
        """
        self.stype = stype
        self.data = fme.read_group_h5(h5_path)


ellipse_json_path = find('*fm_eyecameracalc_props.json', new_dir)[0]
print(ellipse_json_path)
with open(ellipse_json_path) as f:
    ellipse_fit_params = json.load(f)
df['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
df['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']

df['original_session_path'] = new_dir
df['probe_name'] = probe

df['index'] = df.index.values
df.reset_index(inplace=True)





    def normalize_PSTHs(self, lcol, rcol):

        p, np, p_name, np_name = sacc.calc_PSTH_DS()

        dsi = sacc.calc_psth_DSI(p, np)

        return dsi, (p_name, p), (np_name, np)

    def saccade_normalization(self):
        """light vs. dark"""
            

        for ind, row in self.data.iterrows():

            # Determine the preferred gaze shift direction
            pref, nonpref, p_name, np_name = sacc.calc_PSTH_DS(row['FmLt_gazeshift_left_saccPSTH_dHead1'],
                                                               row['FmLt_gazeshift_right_saccPSTH_dHead1'])

            self.data.at[ind,'pref_gazeshift_direction'] = p_name
            self.data.at[ind,'nonpref_gazeshift_direction'] = np_name

            # Calculate an index of direction selectivity
            self.data.at[ind,'gazeshift_DSI'] = sacc.calc_PSTH_DSI(pref, nonpref)

            # Gaze shift PSTH normalization
            self.data.at[ind, 'pref_gazeshift_psth'] = sacc.norm_PSTH(pref).astype(object)

            self.data.at[ind, 'nonpref_gazeshift_psth'] = sacc.norm_PSTH(nonpref, raw_pref=pref).astype(object)

            # Raw gaze-shifting and compensatory saccades
            self.data.at[ind, 'pref_gazeshift_psth_raw'] = pref.copy().astype(object)
            self.data.at[ind, 'nonpref_gazeshift_psth_raw'] = nonpref.copy().astype(object)

            # Compensatory uses the direction selectivity of gaze shifts

            # preferred direction
            _usestr = 'FmLt_comp_{}_saccPSTH_dHead1'.format(p_name)
            # raw
            self.data.at[ind, 'pref_comp_psth_raw'] = row[_usestr].copy().astype(object)
            # normalized
            self.data.at[ind, 'pref_comp_psth'] = sacc.norm_PSTH(row[_usestr], raw_pref=pref).astype(object)

            # non-preferred
            _usestr = 'FmLt_comp_{}_saccPSTH_dHead1'.format(np_name)
            # raw
            self.data.at[ind, 'nonpref_comp_psth_raw'] = row[_usestr].copy().astype(object)
            # normalized
            self.data.at[ind, 'nonpref_comp_psth'] = sacc.norm_PSTH(row[_usestr], raw_pref=pref).astype(object)

    def peak_latency(self):

        for ind, row in self.data.iterrows():
            raw_psth = row['pref_gazeshift_psth_raw']
            norm_psth = row['pref_gazeshift_psth']
            
            peakT, peak_val = sacc.calc_latency(norm_psth)
            
            self.data.at[ind, 'FmLt_gazeshift_baseline'] = raw_psth[0:800].astype(object)
            self.data.at[ind, 'FmLt_gazeshift_med_baseline'] = np.median(raw_psth[0:800])
            self.data.at[ind, 'FmLt_gazeshift_peak_val'] = peak_val
            self.data.at[ind, 'FmLt_gazeshift_peakT'] = peakT

    def recalc_firing_rate(self, eyeT_str, spT_str):
        

        newFr = pd.Series([])

        for ind, row in self.data.iterrows():
            sec = row[eyeT_str][-1].astype(float) - row[eyeT_str][0].astype(float)
            sp = len(row[spT_str])
            fr = sp/sec
            newFr.at[ind] = fr

        return newFr

    def fmt_for_clustering(self):
        
        # Firing rates are not unique to the stimulus (they are the average
        # for the entire session) since Kilosort and Phy2. Usually want to do
        # this using the freely moving recording in the light condition.
        test_fr_using = 'Fm_fr'
        if (self.stype=='hffm') or (self.stype=='ltdk'):
            test_fr_using = 'Fm_fr'
            self.data[test_fr_using] = self.recalc_firing_rate('FmLt_eyeT', 'FmLt_spikeT')
        
        # Calulcate two measures of firing rate modulation following saccade: the raw
        # PSTH and the normalized PSTH. One will be a measure of modulation in units of
        # spikes/sec, and the other will be a percentage change
        for ind, row in self.data.iterrows():
            
            raw_psth = row['pref_gazeshift_psth_raw']
            self.data.at[ind, 'raw_mod_at_pref_peak'] = sacc.calc_PSTH_modind(raw_psth)
            
            norm_psth = row['pref_gazeshift_psth']
            self.data.at[ind, 'norm_mod_at_pref_peak'] = sacc.calc_PSTH_modind(norm_psth)

        self.data['gazeshift_responsive'] = False
        for ind, row in self.data.iterrows():

            # If modulated by at least 1 spike/sec AND 10% of pre-saccadic firing rate
            if (row['raw_mod_at_pref_peak']>1) and (row['norm_mod_at_pref_peak']>0.1):
                self.data.at[ind, 'gazeshift_responsive'] = True

                # TODO: Why was is commented out? Is it used for any dataset? Just hffm? not sure right now
                # elif (row['FmLt_gazeshift_peakT']<.035):
                #     data.at[ind, 'movement_responsive'] = True

        n_resp = self.data['gazeshift_responsive'].sum()
        n_count = len(self.data.index.values)
        print('{} of {} cells ({:1}%) and responsive'.format(n_resp, n_count, n_resp/n_count))

        # plt.subplots(5,5,figsize=(15,15))
        # plot_inds = np.random.randint(0, len(data.index.values), size=25)
        # for i, ind in enumerate(plot_inds):
        #     if data.loc[ind,'gazeshift_responsive']==True:
        #         color = 'b'
        #     else:
        #         color = 'r'
        #     psth = data.loc[ind, 'FmLt_gazeshift_{}_saccPSTH_dHead1'.format(data.loc[ind, 'pref_gazeshift_direction'])]
        #     plt.subplot(5,5,i+1)
        #     plt.plot(psth_bins, psth, color=color)
        #     plt.title('r={:.2f}, n={:.2f}'.format(data.loc[ind,'raw_mod_at_pref_peak'], data.loc[ind,'norm_mod_at_pref_peak']))
        #     plt.xlim([-.5,.5]); plt.ylim([0, np.max(psth)*1.2])

        for ind, row in self.data.iterrows():
            if self.data.loc[ind,'gazeshift_responsive']==True:
                self.data.at[ind,'pref_gazeshift_psth_for_kmeans'] = self.data.loc[ind,'pref_gazeshift_psth'].copy().astype(object)
            elif self.data.loc[ind,'gazeshift_responsive']==False:
                self.data.at[ind,'pref_gazeshift_psth_for_kmeans'] = np.zeros([2001]).astype(object)

        # Create a numpy array of the kmeans input. Index into 950:1300 (t=0 is index 1000, so
        # this is t= -50:300 in units of ms relative to saccade onset)
        pca_input = fme.flatten_series(self.data['pref_gazeshift_psth_for_kmeans'])[:,950:1300]

        return pca_input

    def make_clustering_model(self, pca_input, n_pcas=10, req_explvar=0.95,
                              model_savepath=None):
        n_pcas = 10

        pca = sklearn.decomposition.PCA(n_components=n_pcas)
        pca.fit(pca_input)

        explvar = pca.explained_variance_ratio_

        proj = pca.transform(pca_input)

        keep_pcas = int(np.argwhere(np.cumsum(explvar)>req_explvar)[0])
        print('using best {} PCs'.format(keep_pcas))

        gproj = proj[:,:keep_pcas]

        km = sklearn.cluster.KMeans(n_clusters=5)
        km.fit_predict(gproj)
        Z = km.labels_

        if model_savepath is not None:
            
            _date, _time = fme.fmt_now()

            km_savename = 'KMeans_model_{}-{}.pickle'.format(_date, _time)
            km_savepath = os.path.join(model_savepath, km_savename)

            with open(km_savepath, 'wb') as f:
                pickle.dump(km, f)

            pca_savename = 'PCA_model_{}-{}.pickle'.format(_date, _time)
            pca_savepath = os.path.join(model_savepath, pca_savename)

            with open(pca_savepath, 'wb') as f:
                pickle.dump(pca, f)

            pcaproj_savename = 'PCA_projection_{}-{}.npy'.format(_date, _time)
            pcaproj_savepath = os.path.join(model_savepath, pcaproj_savename)

            np.save(file=pcaproj_savepath, arr=proj)

        # Add cluster labels to object data.
        # Initialize the col values to -1. Then, fill them in.
        self.data['gazecluster_ind'] = -1
        for i, ind in enumerate(self.data.index.values):
            self.data.at[ind, 'gazecluster_ind'] = Z[i]

        ax0.set_ylabel('PC1')
        ax0.set_ylabel('PC2')
        ax0.scatter(proj[:,0], proj[:,1], c=Z)

        ax1.plot(np.arange(1, n_pcas+1), explvar, 'k')
        ax1.hlines(explvar, 1, n_pcas+1, color='tab:red')
        ax1.set_ylabel('explained variance')
        ax1.set_xlabel('number PCs')

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

    def apply_clustering_model(self, pca_input, pca_model_path, km_model_path, keep_pcas=4):
        """
        when you want to apply an existing model to novel data
        for ltdk, apply the clustering from hffm to ltdk
        for flhf, build a model using both hffm and flhf. then apply it to flhf

        have to set # PCs since checking the explained variance doesn't
        make sense for an existing model
        """

        with open(pca_model_path, 'rb') as f:
            km = pickle.load(f)

        with open(km_model_path, 'rb') as f:
            pca = pickle.load(f)

        proj = pca.transform(pca_input)

        gproj = proj[:,:keep_pcas]

        Z = km.predict(gproj)

        self.data['gazecluster_ind'] = -1
        for i, ind in enumerate(self.data.index.values):
            self.data.at[ind, 'gazecluster_ind'] = Z[i]


    def hffm(self):
        """head-fixed vs. freely moving"""

    def flhf(self):
        """Flashed head-fixed stimuli"""




    def main():
        self.normalize_PSTHs(lcol='FmLt_gazeshift_left_saccPSTH_dHead1',
                             rcol='FmLt_gazeshift_right_saccPSTH_dHead1')

        data.at[ind,'pref_gazeshift_direction'] = prefname
        data.at[ind,'nonpref_gazeshift_direction'] = nonprefname
        data.at[ind,'gazeshift_DSI'] = calc_psth_DSI(pref, nonpref)



        for ind, row in self.data.iterrows():
            pref, nonpref, prefname, nonprefname = get_direction_pref(row['FmLt_gazeshift_left_saccPSTH_dHead1'],
                                                row['FmLt_gazeshift_right_saccPSTH_dHead1'])

            




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



    def load_cluster_models(self, pca=None, kmeans=None):




    def label_movcluster(psth, el_bound=0.08):
        """
        PSTH should be the neural response to eye movements
        between -0.0625 and 0.3125 sec, where 0 is the moment
        of the eye movement.
        """

        # find peaks and troughs in PSTH
        p, peak_props = scipy.signal.find_peaks(psth, height=.30)
        t, trough_props = scipy.signal.find_peaks(-psth, height=.20)

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