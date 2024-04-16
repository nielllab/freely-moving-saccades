"""
FreelyMovingEphys/projects/ephys/population.py
"""
import pandas as pd
import numpy as np
import os, platform, json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
import scipy.interpolate
from matplotlib import cm
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.rcParams.update({'font.size': 25})

from src.utils.path import find
from src.utils.auxiliary import flatten_series

def to_color(r,g,b):
    return (r/255, g/255, b/255)

class Population:
    def __init__(self, savepath, metadata_path=None):
        self.metadata_path = metadata_path
        self.savepath = savepath
        self.samprate = 30000
        self.model_dt = 0.025
        lower = -0.5; upper = 1.5; grat_dt = 0.1
        gratbins = np.arange(lower,upper+grat_dt,grat_dt)
        self.grat_psth_x = gratbins[0:-1]+ grat_dt/2
        self.trange = np.arange(-1, 1.1, self.model_dt)
        self.trange_x = 0.5*(self.trange[0:-1]+ self.trange[1:])
        self.cmap_orientation = ['#fec44f','#ec7014','#993404','#000000'] # [low, mid, high, spont]
        self.cmap_movclusts = {
            'all': 'k',
            'movement': to_color(230,135,45), # orange
            'early': to_color(44,140,109), # green
            'late': to_color(46,131,232), # blue
            'biphasic': to_color(222,190,43), # yellow
            'negative': to_color(111,61,175), # purple
            'unresponsive': 'dimgray'
        }
        self.cmap_celltype = ['sandybrown', 'olivedrab'] # [exc, inh]
        self.cmap_sacc = ['steelblue','coral'] # [right, left]
        self.cmap_special2 = ['dimgray','deepskyblue']

        self.high_sacc_thresh = 300 # deg/sec
        self.low_sacc_thresh = 180 # deg/sec
        self.gaze_sacc_thresh = 60 # deg/sec

        self.trange_win = [37,53]
        self.trange_twin = self.trange_x[self.trange_win[0]:self.trange_win[1]]

    def gather_data(self, csv_filepath=None, path_list=None, probe_list=None, ltdk_bools=None,
                    opticflow_bools=None):
        if csv_filepath==None:
            csv_filepath = self.metadata_path
        if path_list is not None:
            goodsessions = path_list
            use_path_list = True
            probenames_for_goodsessions = probe_list
            use_in_dark_analysis = ltdk_bools
            use_in_optic_flow = opticflow_bools
        elif path_list is None:
            use_path_list = False
            # open the csv file of metadata and pull out all of the desired data paths
            if type(csv_filepath) == str:
                csv = pd.read_csv(csv_filepath)
                for_data_pool = csv[csv['good_experiment'] == any(['TRUE' or True or 'True'])]
            elif type(csv_filepath) == pd.Series:
                for_data_pool = csv_filepath
            goodsessions = []; probenames_for_goodsessions = []; layer5_depth_for_goodsessions = []; use_in_dark_analysis = []; use_in_optic_flow = []
            # get all of the best freely moving recordings of a session into a dictionary
            if not (for_data_pool['computer']=='local').all() and not (for_data_pool['drive']=='local').all():
                goodlightrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else 'fm1' for i in for_data_pool['best_light_fm']]))
                gooddarkrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else None for i in for_data_pool['best_dark_fm']]))
            else:
                goodlightrecs = dict(zip(pd.read_csv(self.metadata_path)['animal_dirpath'].to_list(), [i if i !='' else 'fm1' for i in for_data_pool['best_light_fm']]))
                gooddarkrecs = dict(zip(pd.read_csv(self.metadata_path)['animal_dirpath'].to_list(), [i if i !='' else None for i in for_data_pool['best_dark_fm']]))
            # change paths to work with linux
            if platform.system() == 'Linux':
                for ind, row in for_data_pool.iterrows():
                    if row['computer'] != 'local' and row['drive'] != 'local':
                        drive = [row['drive'] if row['drive'] == 'nlab-nas' else row['drive'].capitalize()][0]
                        for_data_pool.loc[ind,'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(), drive] + list(filter(None, row['animal_dirpath'].replace('\\','/').split('/')))[2:])))
            for ind, row in for_data_pool.iterrows():
                goodsessions.append(row['animal_dirpath'])
                probenames_for_goodsessions.append(row['probe_name'])
                layer5_depth_for_goodsessions.append(row['overwrite_layer5center'])
                use_in_dark_analysis.append(row['use_in_dark_analysis'])
                use_in_optic_flow.append(row['use_in_optic_flow'])
        # get the .h5 files from each day
        # this will be a list of lists, where each list inside of the main list has all the data of a single session
        sessions = [find('*_ephys_props.h5',session) for session in goodsessions]
        # read the data in and append them into one shared df
        all_data = pd.DataFrame([])
        ind = 0
        sessions = [i for i in sessions if i != []]
        for session in tqdm(sessions):
            session_data = pd.DataFrame([])
            for recording in session:
                rec_data = pd.read_hdf(recording)
                # get name of the current recording (i.e. 'FmLt' or 'Wn')
                rec_type = '_'.join(([col for col in rec_data.columns.values if 'contrast_tuning_bins' in col][0]).split('_')[:-3])
                # rename spike time columns so that data is retained for each of the seperate trials
                rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT', 'spikeTraw':rec_type+'_spikeTraw','rate':rec_type+'_rate','n_spikes':rec_type+'_n_spikes'})
                # add a column for which fm recording should be prefered
                if not use_path_list:
                    for key,val in goodlightrecs.items():
                        if key in rec_data['session'].iloc[0]:
                            rec_data['best_light_fm'] = val
                    for key,val in gooddarkrecs.items():
                        if key in rec_data['session'].iloc[0]:
                            rec_data['best_dark_fm'] = val
                # get column names
                column_names = list(session_data.columns.values) + list(rec_data.columns.values)
                # new columns for same unit within a session
                session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True)
                # add the list of column names from all sessions plus the current recording
                session_data.columns = column_names
                # remove duplicate columns (i.e. shared metadata)
                session_data = session_data.loc[:,~session_data.columns.duplicated()]
            # add probe name as new col
            animal = goodsessions[ind]
            ellipse_json_path = find('*fm_eyecameracalc_props.json', animal)
            if ellipse_json_path != []:
                with open(ellipse_json_path[0]) as f:
                    ellipse_fit_params = json.load(f)
                session_data['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
                session_data['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']
            # add the session path to each row
            session_data['original_session_path'] = goodsessions[ind]
            # add probe name
            session_data['probe_name'] = probenames_for_goodsessions[ind]
            session_data['use_in_optic_flow'] = use_in_optic_flow[ind]
            if not use_path_list:
                session_data['use_in_dark_analysis'] = use_in_dark_analysis[ind]
                # replace LFP power profile estimate of laminar depth with value entered into spreadsheet
                manual_depth_entry = layer5_depth_for_goodsessions[ind]
                if 'Wn_layer5cent_from_lfp' in session_data.columns.values:
                    if type(session_data['Wn_layer5cent_from_lfp'].iloc[0]) != float and type(manual_depth_entry) != float and manual_depth_entry not in ['?','','FALSE',False]:
                        num_sh = len(session_data['Wn_layer5cent_from_lfp'].iloc[0])
                        for i, row in session_data.iterrows():
                            session_data.at[i, 'Wn_layer5cent_from_lfp'] = list(np.ones([num_sh]).astype(int)*int(manual_depth_entry))
            else:
                session_data['use_in_dark_analysis'] = use_in_dark_analysis[ind]
            ind += 1
            # new rows for units from different mice or sessions
            all_data = pd.concat([all_data, session_data], axis=0)
        fm2_light = [c for c in all_data.columns.values if 'fm2_light' in c]
        fm1_dark = [c for c in all_data.columns.values if 'fm1_dark' in c]
        dark_dict = dict(zip(fm1_dark, [i.replace('fm1_dark', 'FmDk') for i in fm1_dark]))
        light_dict = dict(zip(fm2_light, [i.replace('fm2_light_', 'FmLt_') for i in fm2_light]))
        all_data = all_data.rename(dark_dict, axis=1).rename(light_dict, axis=1)
        # drop empty data without session name
        for ind, row in all_data.iterrows():
            if type(row['session']) != str:
                all_data = all_data.drop(ind, axis=0)
        # combine columns where one property of the unit is spread across multiple columns because of renaming scheme
        for col in list(all_data.loc[:,all_data.columns.duplicated()].columns.values):
            all_data[col] = all_data[col].iloc[:,0].combine_first(all_data[col].iloc[:,1])
        # and drop the duplicates that have only partial data (all the data will now be in another column)
        self.data = all_data.loc[:,~all_data.columns.duplicated()]
        
        # clean up index
        self.data['index'] = self.data.index.values
        self.data.reset_index(inplace=True)

    def save(self, fname, savedir=None):
        if savedir is None:
            savedir = self.savepath
        pickle_path = os.path.join(savedir, fname+'.pickle')
        if os.path.isfile(pickle_path):
            os.remove(pickle_path)
        print('saving to '+pickle_path)
        self.data.to_pickle(pickle_path)

    def load(self, fname, savedir=None):
        if savedir is None:
            savedir = self.savepath
        pickle_path = os.path.join(savedir, fname+'.pickle')
        print('reading from '+pickle_path)
        self.data = pd.read_pickle(pickle_path)

    def add_available_optic_flow_data(self, use_lag=2):
        """
        use_lag is the index--not the value
        so use_lag=2 is time lag 0msec
        """
        self.data['has_optic_flow'] = False
        self.data['has_topdown_optic_flow'] = False
        self.data['FmLt_flowvec_scale'] = None

        movement_state_list = ['full','active_gyro','inactive_gyro','running_forward','running_backward','fine_motion','immobile']
        
        dummy_vec = np.zeros([40,60,2])
        vec_series = pd.Series([])
        dummy_amp = np.zeros([40,60])
        amp_series = pd.Series([])
        for i in range(len(self.data)):
            vec_series[i] = dummy_vec.astype(object)
            amp_series[i] = dummy_amp.astype(object)

        for movement_state in movement_state_list:
            self.data['FmLt_optic_flow_'+movement_state+'_vec'] = vec_series.astype(object)
            self.data['FmLt_optic_flow_'+movement_state+'_amp'] = amp_series.astype(object)
        
        recordings = self.data['original_session_path'].unique()
        recordings = [os.path.join(x, 'fm1') for x in recordings]
        for i, recording_path in enumerate(recordings):
            flow_files = find('*optic_flow.npz', recording_path)
            if flow_files:
                print('reading '+flow_files[0])
                flow_data = np.load(flow_files[0])
                # optic flow w/ topdown tracking, or only using gyro?
                if 'running_forward_vec' in flow_data.files:
                    topdown_flow = True
                else:
                    topdown_flow = False

                if topdown_flow:
                    movement_state_dict = {'full_vec':flow_data['full_vec'],'full_amp':flow_data['full_amp'],
                                        'active_gyro_vec':flow_data['active_gyro_vec'],'active_gyro_amp':flow_data['active_gyro_amp'],
                                        'inactive_gyro_vec':flow_data['inactive_gyro_vec'],'inactive_gyro_amp':flow_data['inactive_gyro_amp'],
                                        'running_forward_vec':flow_data['running_forward_vec'],'running_forward_amp':flow_data['running_forward_amp'],
                                        'running_backward_vec':flow_data['running_backward_vec'],'running_backward_amp':flow_data['running_backward_amp'],
                                        'fine_motion_vec':flow_data['fine_motion_vec'],'fine_motion_amp':flow_data['fine_motion_amp'],
                                        'immobile_vec':flow_data['immobile_vec'],'immobile_amp':flow_data['immobile_amp']}
                    movement_state_list = ['full','active_gyro','inactive_gyro','running_forward','running_backward','fine_motion','immobile']
                elif not topdown_flow:
                    movement_state_dict = {'full_vec':flow_data['full_vec'],'full_amp':flow_data['full_amp'],
                                        'active_gyro_vec':flow_data['active_gyro_vec'],'active_gyro_amp':flow_data['active_gyro_amp'],
                                        'inactive_gyro_vec':flow_data['inactive_gyro_vec'],'inactive_gyro_amp':flow_data['inactive_gyro_amp']}
                    movement_state_list = ['full','active_gyro','inactive_gyro']

                origsess, _ = os.path.split(recording_path)

                for movement_state in movement_state_list:
                    flow_vec = movement_state_dict[movement_state+'_vec'] # shape is [unit, lag, x, y, U/V]
                    flow_amp = movement_state_dict[movement_state+'_amp'] # shape is [unit, lag, x, y]
                    if movement_state=='full':
                        vec_scale = np.zeros(np.size(flow_vec, 0))
                    flow_arr_ind = 0
                    use_inds = [os.path.samefile(p, origsess) for p in self.data['original_session_path']]
                    for ind, _ in self.data[use_inds].iterrows():
                        self.data.at[ind, 'FmLt_optic_flow_'+movement_state+'_vec'] = flow_vec[flow_arr_ind, use_lag].astype(object)
                        self.data.at[ind, 'FmLt_optic_flow_'+movement_state+'_amp'] = flow_amp[flow_arr_ind, use_lag].astype(object)
                        if movement_state=='full':
                            vec_scale[flow_arr_ind] = np.max(np.sqrt((flow_vec[flow_arr_ind, use_lag, :, 0].flatten()**2) + (flow_vec[flow_arr_ind, use_lag, :, 1].flatten()**2))) # U**2 + V**2
                        flow_arr_ind += 1
                    if movement_state=='full':
                        max_vec_scale = np.max(vec_scale.flatten())
                        for ind, _ in self.data[use_inds].iterrows():
                            self.data.at[ind, 'FmLt_flowvec_scale'] = max_vec_scale
                            self.data.at[ind, 'has_optic_flow'] = True
                            self.data.at[ind, 'has_topdown_optic_flow'] = topdown_flow
            
    def optic_flow_vec(self, panel, movstate, do_norm=False):
        fv = self.current_row['FmLt_optic_flow_'+movstate+'_vec'].astype(float) # shape is [x, y, U/V]
        fa = self.current_row['FmLt_optic_flow_'+movstate+'_amp'].astype(float) # shape is [x, y]

        nx = 5 # binning for plotting flow vectors
        fv_scale = self.current_row['FmLt_flowvec_scale']
        flow_w = np.size(fv, 1)
        flow_h = np.size(fv, 0)

        X,Y = np.meshgrid(np.arange(0,flow_w),np.arange(0,flow_h))

        if do_norm:
            norm_amp = (fa/np.max(fa))

            U = fv[:,:,0] * norm_amp
            V = fv[:,:,1] * norm_amp

        elif not do_norm:
            U = fv[:,:,0]
            V = fv[:,:,1]

        panel.quiver(X[::nx,::nx], -Y[::nx,::nx], U[::nx,::nx], -V[::nx,::nx], scale=fv_scale)
        panel.axis('off')
        panel.axis('equal')
        panel.set_title(movstate)

    def optic_flow_amp(self, panel, movstate):
        fa = self.current_row['FmLt_optic_flow_'+movstate+'_amp'].astype(float) # shape is [x, y]

        panel.imshow(fa, cmap='Reds')
        panel.axis('off')
        panel.axis('equal')
        # panel.set_title(movstate + ' amp')

    def add_avalible_RcSn_psth(self):
        self.data['has_hfpsth'] = False

        # set up and create empty columns
        dummy_psth = np.zeros([2001])*np.nan
        psth_series = pd.Series([])
        for i in range(len(self.data)):
            psth_series[i] = dummy_psth.astype(object)
        for col in ['Rc_psth','Sn_on_all_psth','Sn_on_darkstim_psth','Sn_on_lightstim_psth',
                    'Sn_on_background_psth','Sn_off_all_psth','Sn_off_darkstim_psth',
                    'Sn_off_lightstim_psth','Sn_off_background_psth']:
            self.data[col] = psth_series.copy().astype(object)
        dummy_vals = np.zeros([2])*np.nan
        dummy_series = pd.Series([])
        for i in range(len(self.data)):
            dummy_series[i] = dummy_vals.astype(object)
        self.data['Wn_rf_on_cent'] = dummy_series.copy().astype(object)
        self.data['Wn_rf_off_cent'] = dummy_series.copy().astype(object)

        # add it for each unit
        sessions = self.data['original_session_path'].unique()
        for session_path in sessions:
            psth_files = find('addtlhf_props1.npz', session_path)
            if psth_files:
                print('reading '+psth_files[0])
                psth_data = np.load(psth_files[0])
                # reversing checkerboard
                rc_psth = psth_data['rc'] # shape is [unit#, time]
                # sparse noise
                sn_on_psth = psth_data['sn_on'] # shape is [unit#, time, all/l2d/d2l/only_global]
                sn_off_psth = psth_data['sn_off'] # shape is [unit#, time, all/l2d/d2l/only_global]
                # receptive field centers
                rf_xy = psth_data['rf'] # shape is [unit#, x/y]
                
                # just the current session
                use_inds = [os.path.samefile(p, session_path) for p in self.data['original_session_path']]
                for i, ind in enumerate(self.data[use_inds].index.values):
                    self.data.at[ind, 'has_hfpsth'] = True
                    self.data.at[ind, 'Rc_psth'] = rc_psth[i,:]
                    self.data.at[ind, 'Wn_rf_on_cent'] = rf_xy[i,:2]
                    self.data.at[ind, 'Wn_rf_off_cent'] = rf_xy[i,2:]

                    self.data.at[ind, 'Sn_on_all_psth'] = sn_on_psth[i,:,0]
                    self.data.at[ind, 'Sn_on_darkstim_psth'] = sn_on_psth[i,:,1]
                    self.data.at[ind, 'Sn_on_lightstim_psth'] = sn_on_psth[i,:,2]
                    self.data.at[ind, 'Sn_on_background_psth'] = sn_on_psth[i,:,3]

                    self.data.at[ind, 'Sn_off_all_psth'] = sn_off_psth[i,:,0]
                    self.data.at[ind, 'Sn_off_darkstim_psth'] = sn_off_psth[i,:,1]
                    self.data.at[ind, 'Sn_off_lightstim_psth'] = sn_off_psth[i,:,2]
                    self.data.at[ind, 'Sn_off_background_psth'] = sn_off_psth[i,:,3]
            else:
                print('no files found')

    def rc_psth(self, panel, tightx=True):
        panel.plot(self.trange_x, self.current_row['Rc_psth'], color='k')
        panel.set_ylim([0,np.nanmax(self.current_row['Rc_psth'])*1.2])
        panel.set_title('RevChecker PSTH')
        panel.set_ylabel('sec')
        panel.set_xlabel('sp/sec')
        if tightx:
            panel.set_xlim([-0.2,0.4])

    def sn_psth(self, panel, tightx=True):
        panel.plot(self.trange_x, self.current_row['Sn_l2d_psth'], color='k')
        panel.plot(self.trange_x, self.current_row['Sn_d2l_psth'], color='lightgray')
        panel.set_ylim([0,np.nanmax(self.current_row['Rc_psth'])*1.2])
        panel.set_title('SparseNoise PSTH')
        panel.set_ylabel('sec')
        panel.set_xlabel('sp/sec')
        if tightx:
            panel.set_xlim([-0.2,0.4])

    def tuning_modulation_index(self, tuning):
        tuning = tuning[~np.isnan(tuning)]
        modind = (np.max(tuning) - np.min(tuning)) / (np.max(tuning) + np.min(tuning))
        return modind

    def saccade_modulation_index(self, saccavg):
        t0ind = (np.abs(self.trange-0)).argmin()
        t100ind = t0ind+4
        baseline = np.nanmean(saccavg[0:int(t100ind-((1/4)*t100ind))])
        r0 = np.round((saccavg[t0ind] - baseline) / (saccavg[t0ind] + baseline), 3)
        r100 = np.round((saccavg[t100ind] - baseline) / (saccavg[t100ind] + baseline), 3)
        return r0, r100

    def waveform(self, panel):
        wv = self.current_row['waveform']
        panel.plot(np.arange(len(wv))*1000/self.samprate, wv, color='k')
        panel.set_ylabel('millivolts')
        panel.set_xlabel('msec')
        panel.set_title(self.current_row['KSLabel']+' cont='+str(np.round(self.current_row['ContamPct'],3)), fontsize=20)

    def tuning_curve(self, panel, varcent_name, tuning_name, err_name, title, xlabel):
        var_cent = self.current_row[varcent_name]
        tuning = self.current_row[tuning_name]
        tuning_err = self.current_row[err_name]
        panel.errorbar(var_cent,tuning[:],yerr=tuning_err[:], color='k')
        modind = self.tuning_modulation_index(tuning)
        panel.set_title(title+'\nmod.ind.='+str(modind), fontsize=20)
        panel.set_xlabel(xlabel); panel.set_ylabel('sp/sec')
        panel.set_ylim(0, np.nanmax(tuning[:]*1.2))
        return modind

    def grat_stim_tuning(self, panel, tf_sel='mean'):
        if tf_sel=='mean':
            raw_tuning = np.mean(self.current_row['Gt_ori_tuning_tf'],2)
        elif tf_sel=='low':
            raw_tuning = self.current_row['Gt_ori_tuning_tf'][:,:,0]
        elif tf_sel=='high':
            raw_tuning = self.current_row['Gt_ori_tuning_tf'][:,:,1]
        drift_spont = self.current_row['Gt_drift_spont']
        tuning = raw_tuning - drift_spont # subtract off spont rate
        tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
        th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
        osi = np.zeros([3])
        dsi = np.zeros([3])
        for sf in range(3):
            R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
            th_ortho = (th_pref[sf]+2)%8 # get ortho position
            R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5 # ortho firing rate (average between two peaks)
            # orientaiton selectivity index
            osi[sf] = (R_pref - R_ortho) / (R_pref + R_ortho)
            # direction selectivity index
            th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
            R_null = tuning[th_null, sf] # tuning value at that peak
            dsi[sf] = (R_pref - R_null) / (R_pref + R_null)
        panel.set_title(tf_sel+' tf\n OSI l='+str(np.round(osi[0],3))+'m='+str(np.round(osi[1],3))+'h='+str(np.round(osi[2],3))+'\n DSI l='+str(np.round(dsi[0],3))+'m='+str(np.round(dsi[1],3))+'h='+str(np.round(dsi[2],3)), fontsize=20)
        panel.plot(np.arange(8)*45, raw_tuning[:,0], label='low sf', color=self.cmap_orientation[0])
        panel.plot(np.arange(8)*45, raw_tuning[:,1], label='mid sf', color=self.cmap_orientation[1])
        panel.plot(np.arange(8)*45, raw_tuning[:,2], label='high sf', color=self.cmap_orientation[2])
        panel.plot([0,315],[drift_spont,drift_spont],':',label='spont', color=self.cmap_orientation[3])
        if tf_sel=='mean':
            panel.legend()
        panel.set_ylim([0,np.nanmax(self.current_row['Gt_ori_tuning_tf'][:,:,:])*1.2])
        if tf_sel=='mean':
            self.data.at[self.current_index, 'Gt_osi_low'] = osi[0]; self.data.at[self.current_index, 'Gt_osi_mid'] = osi[1]; self.data.at[self.current_index, 'Gt_osi_high'] = osi[2]
            self.data.at[self.current_index, 'Gt_dsi_low'] = dsi[0]; self.data.at[self.current_index, 'Gt_dsi_mid'] = dsi[1]; self.data.at[self.current_index, 'Gt_dsi_high'] = dsi[2]

    def revchecker_laminar_depth(self, panel):
        if np.size(self.current_row['Rc_response_by_channel'],0) == 64:
            shank_channels = [c for c in range(np.size(self.current_row['Rc_response_by_channel'], 0)) if int(np.floor(c/32)) == int(np.floor(int(self.current_row['ch'])/32))]
            whole_shank = self.current_row['Rc_response_by_channel'][shank_channels]
            shank_num = [0 if np.max(shank_channels) < 40 else 1][0]
            colors = plt.cm.jet(np.linspace(0,1,32))
            for ch_num in range(len(shank_channels)):
                panel.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
            panel.plot(whole_shank[self.current_row['Rc_layer4cent'][shank_num]], color=colors[self.current_row['Rc_layer4cent'][shank_num]], label='layer4', linewidth=4) # layer 4
        elif np.size(self.current_row['Rc_response_by_channel'],0) == 16:
            whole_shank = self.current_row['Rc_response_by_channel']
            colors = plt.cm.jet(np.linspace(0,1,16))
            shank_num = 0
            for ch_num in range(16):
                panel.plot(self.current_row['Rc_response_by_channel'][ch_num], color=colors[ch_num], alpha=0.3, linewidth=1) # all other channels
            panel.plot(whole_shank[self.current_row['Rc_layer4cent']], color=colors[self.current_row['Rc_layer4cent']], label='layer4', linewidth=1) # layer 4
        elif np.size(self.current_row['Rc_response_by_channel'],0) == 128:
            shank_channels = [c for c in range(np.size(self.current_row['Rc_response_by_channel'], 0)) if int(np.floor(c/32)) == int(np.floor(int(self.current_row['ch'])/32))]
            whole_shank = self.current_row['Rc_response_by_channel'][shank_channels]
            shank_num = int(np.floor(int(self.current_row['ch'])/32))
            colors = plt.cm.jet(np.linspace(0,1,32))
            for ch_num in range(len(shank_channels)):
                panel.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
            panel.plot(whole_shank[self.current_row['Rc_layer4cent'][shank_num]], color=colors[self.current_row['Rc_layer4cent'][shank_num]], label='layer4', linewidth=4) # layer 4
        else:
            print('unrecognized probe count in LFP plots during unit summary! index='+str(self.current_index))
        self.current_row['ch'] = int(self.current_row['ch'])
        panel.plot(self.current_row['Rc_response_by_channel'][self.current_row['ch']%32], color=colors[self.current_row['ch']%32], label='this channel', linewidth=4) # current channel
        depth_to_layer4 = 0 # could be 350um, but currently, everything will stay relative to layer4 since we don't know angle of probe & other factors
        if self.current_row['probe_name'] == 'DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25
        if shank_num == 0:
            position_of_ch = int(self.current_row['Rc_relative_depth'][0][self.current_row['ch']])
            self.data.at[self.current_index, 'Rc_ch_lfp_relative_depth'] = position_of_ch
            depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
            self.data.at[self.current_index, 'Rc_depth_from_layer4'] = depth_from_surface
            panel.set_title('ch='+str(self.current_row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
        elif shank_num == 1:
            position_of_ch = int(self.current_row['Rc_relative_depth'][1][self.current_row['ch']-32])
            self.data.at[self.current_index, 'Rc_ch_lfp_relative_depth'] = position_of_ch
            depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
            self.data.at[self.current_index, 'Rc_depth_from_layer4'] = depth_from_surface
            panel.set_title('ch='+str(self.current_row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
        elif shank_num == 2:
            position_of_ch = int(self.current_row['Rc_relative_depth'][1][self.current_row['ch']-64])
            self.data.at[self.current_index, 'Rc_ch_lfp_relative_depth'] = position_of_ch
            depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
            self.data.at[self.current_index, 'Rc_depth_from_layer4'] = depth_from_surface
            panel.set_title('ch='+str(self.current_row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
        elif shank_num == 3:
            position_of_ch = int(self.current_row['Rc_relative_depth'][1][self.current_row['ch']-96])
            self.data.at[self.current_index, 'Rc_ch_lfp_relative_depth'] = position_of_ch
            depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
            self.data.at[self.current_index, 'Rc_depth_from_layer4'] = depth_from_surface
            panel.set_title('ch='+str(self.current_row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
        panel.legend(); panel.axvline(x=(0.1*30000), color='k', linewidth=1)
        panel.set_xticks(np.arange(0,18000,18000/8))
        panel.set_xticklabels(np.arange(-100,500,75))
        panel.set_xlabel('msec'); panel.set_ylabel('uvolts')

    def grat_psth(self, panel):
        lower = -0.5; upper = 1.5; dt = 0.1
        bins = np.arange(lower,upper+dt,dt)
        psth = self.current_row['Gt_grating_psth']
        panel.plot(bins[0:-1]+ dt/2,psth, color='k')
        panel.set_title('gratings psth', fontsize=20)
        panel.set_xlabel('time'); panel.set_ylabel('sp/sec')
        panel.set_ylim([0,np.nanmax(psth)*1.2])

    def lfp_laminar_depth(self, panel):
        power_profiles = self.current_row['Wn_lfp_power']
        ch_shank = int(np.floor(self.current_row['ch']/32))
        ch_shank_profile = power_profiles[ch_shank]
        ch_power = ch_shank_profile[int(self.current_row['ch']%32)]
        layer5cent = self.current_row['Wn_layer5cent_from_lfp'][ch_shank]
        
        if self.current_row['probe_name'] == 'DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25

        ch_depth = ch_spacing*(self.current_row['ch']%32)-(layer5cent*ch_spacing)
        num_sites = 32
        panel.plot(ch_shank_profile,range(0,num_sites),color='k')
        panel.plot(ch_shank_profile[layer5cent]+0.01,layer5cent,'r*',markersize=12)
        panel.hlines(y=self.current_row['ch']%32, xmin=ch_power, xmax=1, colors='g', linewidth=5)
        panel.set_ylim([33,-1])
        panel.set_yticks(list(range(-1,num_sites+1)))
        panel.set_yticklabels(ch_spacing*np.arange(num_sites+2)-(layer5cent*ch_spacing))
        panel.set_title('shank='+str(ch_shank)+' site='+str(self.current_row['ch']%32)+'\n depth='+str(ch_depth), fontsize=20)
        self.data.at[self.current_index, 'Wn_depth_from_layer5'] = ch_depth

    def sta(self, panel, sta_name, title):
        # wnsta = np.reshape(self.current_row[sta_name],tuple(self.current_row[shape_name]))
        wnsta = self.current_row[sta_name]
        sta_range = np.max(np.abs(wnsta))*1.2
        sta_range = (0.25 if sta_range<0.25 else sta_range)
        panel.set_title(title, fontsize=20)
        panel.imshow(wnsta, vmin=-sta_range, vmax=sta_range, cmap='seismic')
        panel.axis('off')

    def stv(self, panel, stv_name, title):
        # wnstv = np.reshape(self.current_row[stv_name],tuple(self.current_row[shape_name]))
        wnstv = self.current_row[stv_name]
        panel.imshow(wnstv, vmin=-1, vmax=1, cmap='cividis')
        panel.set_title(title, fontsize=20)
        panel.axis('off')

    def movement_psth(self, panel, rightsacc, leftsacc, title, show_legend=False):
        rightavg = self.current_row[rightsacc]; leftavg = self.current_row[leftsacc]
        panel.set_title(title, fontsize=20)
        modind_right = self.saccade_modulation_index(rightavg)
        modind_left = self.saccade_modulation_index(leftavg)
        panel.plot(self.trange_x, rightavg[:], color=self.cmap_sacc[0])
        panel.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]), color=self.cmap_sacc[0], xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
        panel.plot(self.trange_x, leftavg[:], color=self.cmap_sacc[1])
        panel.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]), color=self.cmap_sacc[1], xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
        if show_legend:
            panel.legend(['right','left'], loc=1)
        maxval = np.max(np.maximum(rightavg[:], leftavg[:]))*1.2
        panel.set_ylim([0, maxval])
        panel.set_xlim([-0.5, 0.6])
        return modind_right, modind_left

    def is_empty_index(self, attr, savekey):
        for ind, val in self.data[attr].items():
            self.data.at[ind, savekey] = (True if ~np.isnan(val).all() else False)

    def is_empty_cell(self, row, name):
        if name in row.index.values and type(row[name]) != float and row[name] == []:
            return True
        else:
            return False

    def summarize_units(self, use_pop_outputs=False):
        pdf = PdfPages(os.path.join(self.savepath, 'unit_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))

        if 'FmDk_theta' in self.data.columns:
            self.is_empty_index('FmDk_theta', 'has_dark')
        else:
            self.data['has_dark'] = False
        
        if 'Wn_contrast_tuning' in self.data.columns:
            self.is_empty_index('Wn_contrast_tuning', 'has_hf')
        else:
            self.data['has_hf'] = False

        if 'has_optic_flow' not in self.data.columns:
            self.data['has_optic_flow'] = False

        print('num units=' + str(len(self.data)))

        for index, row in tqdm(self.data.iterrows()):
            self.current_index = index
            self.current_row = row

            # set up page
            self.figure = plt.figure(constrained_layout=True, figsize=(30,22))
            self.spec = gridspec.GridSpec(ncols=7, nrows=10, figure=self.figure)

            # page title
            title = self.figure.add_subplot(self.spec[0,0])
            title.axis('off')
            title.annotate(str(self.current_row['session'])+'_unit'+str(self.current_row['index']), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20)

            # unit waveform
            unitfig_waveform = self.figure.add_subplot(self.spec[0,1])
            self.waveform(panel=unitfig_waveform)

            # whitenoise contrast tuning curve
            fig_contrast_tuning = self.figure.add_subplot(self.spec[0,2])
            if self.current_row['has_hf']:
                Wn_contrast_modind = self.tuning_curve(panel=fig_contrast_tuning,
                                                  varcent_name='Wn_contrast_tuning_bins',
                                                  tuning_name='Wn_contrast_tuning',
                                                  err_name='Wn_contrast_tuning_err',
                                                  title='Wn contrast',
                                                  xlabel='contrast a.u.')
                self.data.at[self.current_index, 'Wn_contrast_modind'] = Wn_contrast_modind
            else:
                fig_contrast_tuning.axis('off')

            # gratings psth
            fig_grat_psth = self.figure.add_subplot(self.spec[0,3])
            if self.current_row['has_hf']:
                self.grat_psth(panel=fig_grat_psth)
            else:
                fig_grat_psth.axis('off')

            # laminar depth relative to cortex layer 4
            # based on revchecker stim
            fig_revchecker_depth = self.figure.add_subplot(self.spec[0,4])
            if self.current_row['has_hf']:
                self.revchecker_laminar_depth(panel=fig_revchecker_depth)
            else:
                fig_revchecker_depth.axis('off')

            # laminar depth relative to cortex layer 5
            # based on whitenoise stim, but the data exist for all stim except for fm
            fig_lfp_depth = self.figure.add_subplot(self.spec[6:8,4])
            if self.current_row['has_hf']:
                self.lfp_laminar_depth(panel=fig_lfp_depth)
            else:
                fig_lfp_depth.axis('off')

            # whitenoise sta
            fig_wn_sta = self.figure.add_subplot(self.spec[1,0])
            if self.current_row['has_hf']:
                self.sta(panel=fig_wn_sta,
                         sta_name='Wn_spike_triggered_average',
                         title='Wn STA')
            else:
                fig_wn_sta.axis('off')

            # whitenoise stv
            fig_wn_stv = self.figure.add_subplot(self.spec[1,1])
            if self.current_row['has_hf']:
                self.stv(panel=fig_wn_stv,
                         stv_name='Wn_spike_triggered_variance',
                         title='Wn STV')
            else:
                fig_wn_stv.axis('off')

            # whitenoise eye movement psth
            fig_wn_eye_psth = self.figure.add_subplot(self.spec[1,2])
            if self.current_row['has_hf']:
                wn_eye_psth_right_modind, wn_eye_psth_left_modind = self.movement_psth(panel=fig_wn_eye_psth,
                                        rightsacc='Wn_rightsacc_avg',
                                        leftsacc='Wn_leftsacc_avg',
                                        title='Wn left/right saccades')
                self.data.at[index, 'Wn_rightsacc_modind_t0'] = wn_eye_psth_right_modind[0]
                self.data.at[index, 'Wn_leftsacc_modind_t0'] = wn_eye_psth_left_modind[0]
                self.data.at[index, 'Wn_rightsacc_modind_t100'] = wn_eye_psth_right_modind[1]
                self.data.at[index, 'Wn_leftsacc_modind_t100'] = wn_eye_psth_left_modind[1]
            else:
                fig_wn_eye_psth.axis('off')

            # whitenoise pupil radius tuning curve
            fig_wn_pupilradius_tuning = self.figure.add_subplot(self.spec[1,3])
            if self.current_row['has_hf']:
                wn_pupilradius_modind = self.tuning_curve(panel=fig_wn_pupilradius_tuning,
                                        varcent_name='Wn_pupilradius_tuning_bins',
                                        tuning_name='Wn_pupilradius_tuning',
                                        err_name='Wn_pupilradius_tuning_err',
                                        title='Wn pupil radius',
                                        xlabel='pxls')
                self.data.at[self.current_index, 'Wn_pupilradius_modind'] = wn_pupilradius_modind
            else:
                fig_wn_pupilradius_tuning.axis('off')

            # whitenoise running speed tuning curve
            fig_speed_tuning = self.figure.add_subplot(self.spec[1,4])
            if self.current_row['has_hf']:
                wn_speed_modind = self.tuning_curve(panel=fig_speed_tuning,
                                        varcent_name='Wn_ballspeed_tuning_bins',
                                        tuning_name='Wn_ballspeed_tuning',
                                        err_name='Wn_ballspeed_tuning_err',
                                        title='Wn ball speed',
                                        xlabel='cm/sec')
                self.data.at[self.current_index, 'Wn_ballspeed_modind'] = wn_speed_modind
            else:
                fig_speed_tuning.axis('off')

            # FmLt sta
            fig_FmLt_sta = self.figure.add_subplot(self.spec[2,0])
            self.sta(panel=fig_FmLt_sta,
                     sta_name='FmLt_spike_triggered_average',
                     title='FmLt STA')

            # FmLt stv
            fig_FmLt_stv = self.figure.add_subplot(self.spec[2,1])
            self.stv(panel=fig_FmLt_stv,
                     stv_name='FmLt_spike_triggered_variance',
                     title='FmLt STV')

            # FmLt gyro z tuning curve
            fig_FmLt_gyro_z_tuning = self.figure.add_subplot(self.spec[2,2])
            FmLt_gyro_z_tuning_modind = self.tuning_curve(panel=fig_FmLt_gyro_z_tuning,
                                    varcent_name='FmLt_gyroz_tuning_bins',
                                    tuning_name='FmLt_gyroz_tuning',
                                    err_name='FmLt_gyroz_tuning_err',
                                    title='FmLt gyro z',
                                    xlabel='deg/sec')
            self.data.at[self.current_index, 'FmLt_gyroz_modind'] = FmLt_gyro_z_tuning_modind

            # FmLt gyro x tuning curve
            fig_FmLt_gyro_x_tuning = self.figure.add_subplot(self.spec[2,3])
            FmLt_gyro_x_tuning_modind = self.tuning_curve(panel=fig_FmLt_gyro_x_tuning,
                                    varcent_name='FmLt_gyrox_tuning_bins',
                                    tuning_name='FmLt_gyrox_tuning',
                                    err_name='FmLt_gyrox_tuning_err',
                                    title='FmLt gyro x',
                                    xlabel='deg/sec')
            self.data.at[self.current_index, 'FmLt_gyrox_modind'] = FmLt_gyro_x_tuning_modind

            # FmLt gyro y tuning curve
            fig_FmLt_gyro_y_tuning = self.figure.add_subplot(self.spec[2,4])
            FmLt_gyro_y_tuning_modind = self.tuning_curve(panel=fig_FmLt_gyro_y_tuning,
                                    varcent_name='FmLt_gyroy_tuning_bins',
                                    tuning_name='FmLt_gyroy_tuning',
                                    err_name='FmLt_gyroy_tuning_err',
                                    title='FmLt gyro y',
                                    xlabel='deg/sec')
            self.data.at[self.current_index, 'FmLt_gyroy_modind'] = FmLt_gyro_y_tuning_modind
            
            if 'FmLt_glm_rf' in self.data.columns:
                if type(self.current_row['FmLt_glm_rf']) != float:
                    # FmLt glm receptive field at five lags
                    glm = self.current_row['FmLt_glm_rf']
                    glm_cc = self.current_row['FmLt_glm_cc']
                    lag_list = [-4,-2,0,2,4]
                    crange = np.max(np.abs(glm))
                    for glm_lag in range(5):
                        unitfig_glm = self.figure.add_subplot(self.spec[3,glm_lag])
                        unitfig_glm.imshow(glm[glm_lag], vmin=-crange, vmax=crange, cmap='seismic')
                        unitfig_glm.set_title('FmLt GLM RF\n(lag='+str(lag_list[glm_lag])+' cc='+str(np.round(glm_cc[glm_lag],2))+')', fontsize=20)
                        unitfig_glm.axis('off')

            # FmLt gaze shift dEye psth
            fig_FmLt_gaze_dEye = self.figure.add_subplot(self.spec[4,1])
            FmLt_gaze_dEye_right_modind, FmLt_gaze_dEye_left_modind = self.movement_psth(panel=fig_FmLt_gaze_dEye,
                                    rightsacc='FmLt_rightsacc_avg_gaze_shift_dEye',
                                    leftsacc='FmLt_leftsacc_avg_gaze_shift_dEye',
                                    title='FmLt gaze shift dEye')
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_gaze_shift_dEye_modind_t0'] = FmLt_gaze_dEye_right_modind[0]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_gaze_shift_dEye_modind_t0'] = FmLt_gaze_dEye_left_modind[0]
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_gaze_shift_dEye_modind_t100'] = FmLt_gaze_dEye_right_modind[1]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_gaze_shift_dEye_modind_t100'] = FmLt_gaze_dEye_left_modind[1]
            
            # FmLt comp dEye psth
            fig_FmLt_comp_dEye = self.figure.add_subplot(self.spec[4,2])
            FmLt_comp_dEye_right_modind, FmLt_comp_dEye_left_modind = self.movement_psth(panel=fig_FmLt_comp_dEye,
                                    rightsacc='FmLt_rightsacc_avg_comp_dEye',
                                    leftsacc='FmLt_leftsacc_avg_comp_dEye',
                                    title='FmLt comp dEye')
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_comp_dEye_modind_t0'] = FmLt_comp_dEye_right_modind[0]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_comp_dEye_modind_t0'] = FmLt_comp_dEye_left_modind[0]
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_comp_dEye_modind_t100'] = FmLt_comp_dEye_right_modind[1]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_comp_dEye_modind_t100'] = FmLt_comp_dEye_left_modind[1]

            # FmLt gaze shift dHead psth
            fig_FmLt_gaze_dHead = self.figure.add_subplot(self.spec[4,3])
            FmLt_gaze_dHead_right_modind, FmLt_gaze_dHead_left_modind = self.movement_psth(panel=fig_FmLt_gaze_dHead,
                                    rightsacc='FmLt_rightsacc_avg_gaze_shift_dHead',
                                    leftsacc='FmLt_leftsacc_avg_gaze_shift_dHead',
                                    title='FmLt gaze shift dHead')
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_gaze_shift_dHead_modind_t0'] = FmLt_gaze_dHead_right_modind[0]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_gaze_shift_dHead_modind_t0'] = FmLt_gaze_dHead_left_modind[0]
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_gaze_shift_dHead_modind_t100'] = FmLt_gaze_dHead_right_modind[1]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_gaze_shift_dHead_modind_t100'] = FmLt_gaze_dHead_left_modind[1]
            
            # FmLt comp dHead psth
            fig_FmLt_comp_dHead = self.figure.add_subplot(self.spec[4,4])
            FmLt_comp_dHead_right_modind, FmLt_comp_dHead_left_modind = self.movement_psth(panel=fig_FmLt_comp_dHead,
                                    rightsacc='FmLt_rightsacc_avg_comp_dHead',
                                    leftsacc='FmLt_leftsacc_avg_comp_dHead',
                                    title='FmLt comp dHead')
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_comp_dHead_modind_t0'] = FmLt_comp_dHead_right_modind[0]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_comp_dHead_modind_t0'] = FmLt_comp_dHead_left_modind[0]
            self.data.at[self.current_index, 'FmLt_rightsacc_avg_comp_dHead_modind_t100'] = FmLt_comp_dHead_right_modind[1]
            self.data.at[self.current_index, 'FmLt_leftsacc_avg_comp_dHead_modind_t100'] = FmLt_comp_dHead_left_modind[1]

            fig_mean_grat_ori_tuning = self.figure.add_subplot(self.spec[6,0])
            if self.current_row['has_hf']:
                self.grat_stim_tuning(panel=fig_mean_grat_ori_tuning,
                                        tf_sel='mean')
            else:
                fig_mean_grat_ori_tuning.axis('off')
            
            fig_low_grat_ori_tuning = self.figure.add_subplot(self.spec[6,1])
            if self.current_row['has_hf']:
                self.grat_stim_tuning(panel=fig_low_grat_ori_tuning,
                                        tf_sel='low')
            else:
                fig_low_grat_ori_tuning.axis('off')

            fig_high_grat_ori_tuning = self.figure.add_subplot(self.spec[6,2])
            if self.current_row['has_hf']:
                self.grat_stim_tuning(panel=fig_high_grat_ori_tuning,
                                        tf_sel='high')
            else:
                fig_high_grat_ori_tuning.axis('off')

            # FmLt all dEye psth
            fig_FmLt_all_dEye = self.figure.add_subplot(self.spec[4,0])
            FmLt_all_dEye_right_modind, FmLt_all_dEye_left_modind = self.movement_psth(panel=fig_FmLt_all_dEye,
                                    rightsacc='FmLt_rightsacc_avg',
                                    leftsacc='FmLt_leftsacc_avg',
                                    title='FmLt all dEye',
                                    show_legend=True)
            self.data.at[self.current_index, 'FmLt_rightsacc_modind_t0'] = FmLt_all_dEye_right_modind[0]
            self.data.at[self.current_index, 'FmLt_leftsacc_modind_t0'] = FmLt_all_dEye_left_modind[0]
            self.data.at[self.current_index, 'FmLt_rightsacc_modind_t100'] = FmLt_all_dEye_right_modind[1]
            self.data.at[self.current_index, 'FmLt_leftsacc_modind_t100'] = FmLt_all_dEye_left_modind[1]

            # FmLt pupil radius tuning
            fig_FmLt_pupilradius_tuning = self.figure.add_subplot(self.spec[5,0])
            FmLt_pupilradius_modind = self.tuning_curve(panel=fig_FmLt_pupilradius_tuning,
                                    varcent_name='FmLt_pupilradius_tuning_bins',
                                    tuning_name='FmLt_pupilradius_tuning',
                                    err_name='FmLt_pupilradius_tuning_err',
                                    title='FmLt pupil radius',
                                    xlabel='pupil radius')
            self.data.at[self.current_index, 'FmLt_pupilradius_modind'] = FmLt_pupilradius_modind

            # FmLt theta tuning
            fig_FmLt_theta_tuning = self.figure.add_subplot(self.spec[5,1])
            FmLt_theta_modind = self.tuning_curve(panel=fig_FmLt_theta_tuning,
                                    varcent_name='FmLt_theta_tuning_bins',
                                    tuning_name='FmLt_theta_tuning',
                                    err_name='FmLt_theta_tuning_err',
                                    title='FmLt theta',
                                    xlabel='deg')
            self.data.at[self.current_index, 'FmLt_theta_modind'] = FmLt_theta_modind

            # FmLt phi tuning
            fig_FmLt_phi_tuning = self.figure.add_subplot(self.spec[5,2])
            FmLt_phi_modind = self.tuning_curve(panel=fig_FmLt_phi_tuning,
                                    varcent_name='FmLt_phi_tuning_bins',
                                    tuning_name='FmLt_phi_tuning',
                                    err_name='FmLt_phi_tuning_err',
                                    title='FmLt phi',
                                    xlabel='deg')
            self.data.at[self.current_index, 'FmLt_phi_modind'] = FmLt_phi_modind

            # FmLt roll tuning
            fig_FmLt_roll_tuning = self.figure.add_subplot(self.spec[5,3])
            FmLt_roll_modind = self.tuning_curve(panel=fig_FmLt_roll_tuning,
                                    varcent_name='FmLt_roll_tuning_bins',
                                    tuning_name='FmLt_roll_tuning',
                                    err_name='FmLt_roll_tuning_err',
                                    title='FmLt roll',
                                    xlabel='deg')
            self.data.at[self.current_index, 'FmLt_roll_modind'] = FmLt_roll_modind

            # FmLt pitch tuning
            fig_FmLt_pitch_tuning = self.figure.add_subplot(self.spec[5,4])
            FmLt_pitch_modind = self.tuning_curve(panel=fig_FmLt_pitch_tuning,
                                    varcent_name='FmLt_pitch_tuning_bins',
                                    tuning_name='FmLt_pitch_tuning',
                                    err_name='FmLt_pitch_tuning_err',
                                    title='FmLt pitch',
                                    xlabel='deg')
            self.data.at[self.current_index, 'FmLt_pitch_modind'] = FmLt_pitch_modind

            if self.current_row['has_optic_flow']:
                fig_flow_full_vec = self.figure.add_subplot(self.spec[0,6])
                fig_flow_full_amp = self.figure.add_subplot(self.spec[0,5])

                fig_flow_ag_vec = self.figure.add_subplot(self.spec[1,6])
                fig_flow_ag_amp = self.figure.add_subplot(self.spec[1,5])

                fig_flow_ig_vec = self.figure.add_subplot(self.spec[2,6])
                fig_flow_ig_amp = self.figure.add_subplot(self.spec[2,5])

                if self.current_row['has_topdown_optic_flow']:
                    fig_flow_rf_vec = self.figure.add_subplot(self.spec[3,6])
                    fig_flow_rf_amp = self.figure.add_subplot(self.spec[3,5])

                    fig_flow_rb_vec = self.figure.add_subplot(self.spec[4,6])
                    fig_flow_rb_amp = self.figure.add_subplot(self.spec[4,5])

                    fig_flow_fm_vec = self.figure.add_subplot(self.spec[5,6])
                    fig_flow_fm_amp = self.figure.add_subplot(self.spec[5,5])

                    fig_flow_im_vec = self.figure.add_subplot(self.spec[6,6])
                    fig_flow_im_amp = self.figure.add_subplot(self.spec[6,5])

                    movstates = ['full','active_gyro','inactive_gyro','running_forward','running_backward','fine_motion','immobile']
                    statevecs = [fig_flow_full_vec, fig_flow_ag_vec, fig_flow_ig_vec, fig_flow_rf_vec, fig_flow_rb_vec, fig_flow_fm_vec, fig_flow_im_vec]
                    stateamps = [fig_flow_full_amp, fig_flow_ag_amp, fig_flow_ig_amp, fig_flow_rf_amp, fig_flow_rb_amp, fig_flow_fm_amp, fig_flow_im_amp]

                elif not self.current_row['has_topdown_optic_flow']:
                    movstates = ['full','active_gyro','inactive_gyro']
                    statevecs = [fig_flow_full_vec, fig_flow_ag_vec, fig_flow_ig_vec]
                    stateamps = [fig_flow_full_amp, fig_flow_ag_amp, fig_flow_ig_amp]

                for i in range(len(movstates)):
                    self.optic_flow_vec(panel=statevecs[i],
                                        movstate=movstates[i])
                    self.optic_flow_amp(panel=stateamps[i],
                                    movstate=movstates[i])

            if use_pop_outputs:
                self.modulation_scatters()

            # set up panels for dark figures
            fig_fmdark_gyro_z_tuning = self.figure.add_subplot(self.spec[7,0])
            fig_fmdark_gyro_x_tuning = self.figure.add_subplot(self.spec[7,1])
            fig_fmdark_gyro_y_tuning = self.figure.add_subplot(self.spec[7,2])
            fig_fmdark_gaze_dEye = self.figure.add_subplot(self.spec[8,1])
            fig_fmdark_comp_dEye = self.figure.add_subplot(self.spec[8,2])
            fig_fmdark_gaze_dHead = self.figure.add_subplot(self.spec[8,3])
            fig_fmdark_comp_dHead = self.figure.add_subplot(self.spec[8,4])
            fig_fmdark_all_dEye = self.figure.add_subplot(self.spec[8,0])
            fig_fmdark_pupilradius_tuning = self.figure.add_subplot(self.spec[9,0])
            fig_fmdark_theta_tuning = self.figure.add_subplot(self.spec[9,1])
            fig_fmdark_phi_tuning = self.figure.add_subplot(self.spec[9,2])
            fig_fmdark_roll_tuning = self.figure.add_subplot(self.spec[9,3])
            fig_fmdark_pitch_tuning = self.figure.add_subplot(self.spec[9,4])

            if not self.current_row['has_dark']:
                # set up empty axes
                fig_fmdark_gyro_z_tuning.axis('off')
                fig_fmdark_gyro_x_tuning.axis('off')
                fig_fmdark_gyro_y_tuning.axis('off')
                fig_fmdark_gaze_dEye.axis('off')
                fig_fmdark_comp_dEye.axis('off')
                fig_fmdark_gaze_dHead.axis('off')
                fig_fmdark_comp_dHead.axis('off')
                fig_fmdark_all_dEye.axis('off')
                fig_fmdark_pupilradius_tuning.axis('off')
                fig_fmdark_theta_tuning.axis('off')
                fig_fmdark_phi_tuning.axis('off')
                fig_fmdark_roll_tuning.axis('off')
                fig_fmdark_pitch_tuning.axis('off')

            elif self.current_row['has_dark']:
                # fm dark gyro z tuning curve
                fmdark_gyro_z_tuning_modind = self.tuning_curve(panel=fig_fmdark_gyro_z_tuning,
                                        varcent_name='FmDk_gyroz_tuning_bins',
                                        tuning_name='FmDk_gyroz_tuning',
                                        err_name='FmDk_gyroz_tuning_err',
                                        title='FmDk gyro z',
                                        xlabel='deg/sec')
                self.data.at[self.current_index, 'FmDk_gyroz_modind'] = fmdark_gyro_z_tuning_modind

                # fm dark gyro x tuning curve
                fmdark_gyro_x_tuning_modind = self.tuning_curve(panel=fig_fmdark_gyro_x_tuning,
                                        varcent_name='FmDk_gyrox_tuning_bins',
                                        tuning_name='FmDk_gyrox_tuning',
                                        err_name='FmDk_gyrox_tuning_err',
                                        title='FmDk gyro x',
                                        xlabel='deg/sec')
                self.data.at[self.current_index, 'FmDk_gyrox_modind'] = fmdark_gyro_x_tuning_modind

                # fm dark gyro y tuning curve
                fmdark_gyro_y_tuning_modind = self.tuning_curve(panel=fig_fmdark_gyro_y_tuning,
                                        varcent_name='FmDk_gyroy_tuning_bins',
                                        tuning_name='FmDk_gyroy_tuning',
                                        err_name='FmDk_gyroy_tuning_err',
                                        title='FmDk gyro y',
                                        xlabel='deg/sec')
                self.data.at[self.current_index, 'FmDk_gyroy_modind'] = fmdark_gyro_y_tuning_modind

                # fm dark gaze shift dEye psth
                fmdark_gaze_dEye_right_modind, fmdark_gaze_dEye_left_modind = self.movement_psth(panel=fig_fmdark_gaze_dEye,
                                        rightsacc='FmDk_rightsacc_avg_gaze_shift_dEye',
                                        leftsacc='FmDk_leftsacc_avg_gaze_shift_dEye',
                                        title='FmDk gaze shift dEye')
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_gaze_shift_dEye_modind_t0'] = fmdark_gaze_dEye_right_modind[0]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_gaze_shift_dEye_modind_t0'] = fmdark_gaze_dEye_left_modind[0]
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_gaze_shift_dEye_modind_t100'] = fmdark_gaze_dEye_right_modind[1]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_gaze_shift_dEye_modind_t100'] = fmdark_gaze_dEye_left_modind[1]
                
                # fm dark comp dEye psth
                fmdark_comp_dEye_right_modind, fmdark_comp_dEye_left_modind = self.movement_psth(panel=fig_fmdark_comp_dEye,
                                        rightsacc='FmDk_rightsacc_avg_comp_dEye',
                                        leftsacc='FmDk_leftsacc_avg_comp_dEye',
                                        title='FmDk comp dEye')
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_comp_dEye_modind_t0'] = fmdark_comp_dEye_right_modind[0]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_comp_dEye_modind_t0'] = fmdark_comp_dEye_left_modind[0]
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_comp_dEye_modind_t100'] = fmdark_comp_dEye_right_modind[1]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_comp_dEye_modind_t100'] = fmdark_comp_dEye_left_modind[1]

                # fm dark gaze shift dHead psth
                fmdark_gaze_dHead_right_modind, fmdark_gaze_dHead_left_modind = self.movement_psth(panel=fig_fmdark_gaze_dHead,
                                        rightsacc='FmDk_rightsacc_avg_gaze_shift_dHead',
                                        leftsacc='FmDk_leftsacc_avg_gaze_shift_dHead',
                                        title='FmDk gaze shift dHead')
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_gaze_shift_dHead_modind_t0'] = fmdark_gaze_dHead_right_modind[0]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_gaze_shift_dHead_modind_t0'] = fmdark_gaze_dHead_left_modind[0]
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_gaze_shift_dHead_modind_t100'] = fmdark_gaze_dHead_right_modind[1]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_gaze_shift_dHead_modind_t100'] = fmdark_gaze_dHead_left_modind[1]
                
                # fm dark comp dHead psth
                fmdark_comp_dHead_right_modind, fmdark_comp_dHead_left_modind = self.movement_psth(panel=fig_fmdark_comp_dHead,
                                        rightsacc='FmDk_rightsacc_avg_comp_dHead',
                                        leftsacc='FmDk_leftsacc_avg_comp_dHead',
                                        title='FmDk comp dHead')
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_comp_dHead_modind_t0'] = fmdark_comp_dHead_right_modind[0]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_comp_dHead_modind_t0'] = fmdark_comp_dHead_left_modind[0]
                self.data.at[self.current_index, 'FmDk_rightsacc_avg_comp_dHead_modind_t100'] = fmdark_comp_dHead_right_modind[1]
                self.data.at[self.current_index, 'FmDk_leftsacc_avg_comp_dHead_modind_t100'] = fmdark_comp_dHead_left_modind[1]

                # fm dark all dEye psth
                fmdark_all_dEye_right_modind, fmdark_all_dEye_left_modind = self.movement_psth(panel=fig_fmdark_all_dEye,
                                        rightsacc='FmDk_rightsacc_avg',
                                        leftsacc='FmDk_leftsacc_avg',
                                        title='FmDk all dEye')
                self.data.at[self.current_index, 'FmDk_rightsacc_modind_t0'] = fmdark_all_dEye_right_modind[0]
                self.data.at[self.current_index, 'FmDk_leftsacc_modind_t0'] = fmdark_all_dEye_left_modind[0]
                self.data.at[self.current_index, 'FmDk_rightsacc_modind_t100'] = fmdark_all_dEye_right_modind[1]
                self.data.at[self.current_index, 'FmDk_leftsacc_modind_t100'] = fmdark_all_dEye_left_modind[1]

                # fm dark pupil radius tuning
                fmdark_pupilradius_modind = self.tuning_curve(panel=fig_fmdark_pupilradius_tuning,
                                        varcent_name='FmDk_pupilradius_tuning_bins',
                                        tuning_name='FmDk_pupilradius_tuning',
                                        err_name='FmDk_pupilradius_tuning_err',
                                        title='FmDk pupil radius',
                                        xlabel='pxls')
                self.data.at[self.current_index, 'FmDk_pupilradius_modind'] = fmdark_pupilradius_modind

                # fm dark theta tuning
                fmdark_theta_modind = self.tuning_curve(panel=fig_fmdark_theta_tuning,
                                        varcent_name='FmDk_theta_tuning_bins',
                                        tuning_name='FmDk_theta_tuning',
                                        err_name='FmDk_theta_tuning_err',
                                        title='FmDk theta',
                                        xlabel='deg')
                self.data.at[self.current_index, 'FmDk_theta_modind'] = fmdark_theta_modind

                # fm dark phi tuning
                fmdark_phi_modind = self.tuning_curve(panel=fig_fmdark_phi_tuning,
                                        varcent_name='FmDk_phi_tuning_bins',
                                        tuning_name='FmDk_phi_tuning',
                                        err_name='FmDk_phi_tuning_err',
                                        title='FmDk phi',
                                        xlabel='deg')
                self.data.at[self.current_index, 'FmDk_phi_modind'] = fmdark_phi_modind

                # fm dark roll tuning
                fmdark_roll_modind = self.tuning_curve(panel=fig_fmdark_roll_tuning,
                                        varcent_name='FmDk_roll_tuning_bins',
                                        tuning_name='FmDk_roll_tuning',
                                        err_name='FmDk_roll_tuning_err',
                                        title='FmDk roll',
                                        xlabel='deg')
                self.data.at[self.current_index, 'FmDk_roll_modind'] = fmdark_roll_modind
                
                # fm dark pitch tuning
                fmdark_pitch_modind = self.tuning_curve(panel=fig_fmdark_pitch_tuning,
                                        varcent_name='FmDk_pitch_tuning_bins',
                                        tuning_name='FmDk_pitch_tuning',
                                        err_name='FmDk_pitch_tuning_err',
                                        title='FmDk pitch',
                                        xlabel='deg')
                self.data.at[self.current_index, 'FmDk_pitch_modind'] = fmdark_pitch_modind

            plt.tight_layout()
            pdf.savefig(self.figure)
            plt.close()
        
        print('saving unit summary pdf')
        pdf.close()

    def get_animal_activity(self):
        active_time_by_session = dict()
        dark_len = []; light_len = []
        sessions = [x for x in self.data['session'].unique() if str(x) != 'nan']
        for session in sessions:
            session_data = self.data[self.data['session']==session]
            # find active times
            if 'FmLt_eyeT' in session_data.columns.values and type(session_data['FmLt_eyeT'].iloc[0]) != float:
                # light setup
                fm_light_eyeT = np.array(session_data['FmLt_eyeT'].iloc[0])
                fm_light_gz = session_data['FmLt_gyro_z'].iloc[0]
                fm_light_accT = session_data['FmLt_imuT'].iloc[0]
                light_model_t = np.arange(0,np.nanmax(fm_light_eyeT),self.model_dt)
                light_model_gz = interp1d(fm_light_accT,(fm_light_gz-np.mean(fm_light_gz))*7.5,bounds_error=False)(light_model_t)
                light_model_active = np.convolve(np.abs(light_model_gz),np.ones(np.int(1/self.model_dt)),'same')
                light_active = light_model_active>40

                n_units = len(session_data)
                light_model_nsp = np.zeros((n_units, len(light_model_t)))
                bins = np.append(light_model_t, light_model_t[-1]+self.model_dt)
                i = 0
                for ind, row in session_data.iterrows():
                    light_model_nsp[i,:], bins = np.histogram(row['FmLt_spikeT'], bins)
                    unit_active_spikes = light_model_nsp[i, light_active]
                    unit_stationary_spikes = light_model_nsp[i, ~light_active]
                    self.data.at[ind,'FmLt_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*self.model_dt)
                    self.data.at[ind,'FmLt_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*self.model_dt)
                    i += 1

                active_time_by_session.setdefault('light', {})[session] = np.sum(light_active) / len(light_active)
                light_len.append(len(light_active))

            if 'FmDk_eyeT' in session_data.columns.values and type(session_data['FmDk_eyeT'].iloc[0]) != float:
                del unit_active_spikes, unit_stationary_spikes

                # dark setup
                FmDk_eyeT = np.array(session_data['FmDk_eyeT'].iloc[0])
                FmDk_gz = session_data['FmDk_gyro_z'].iloc[0]
                FmDk_accT = session_data['FmDk_imuT'].iloc[0]
                dark_model_t = np.arange(0,np.nanmax(FmDk_eyeT),self.model_dt)
                dark_model_gz = interp1d(FmDk_accT,(FmDk_gz-np.mean(FmDk_gz))*7.5,bounds_error=False)(dark_model_t)
                dark_model_active = np.convolve(np.abs(dark_model_gz),np.ones(np.int(1/self.model_dt)),'same')
                dark_active = dark_model_active>40

                n_units = len(session_data)
                dark_model_nsp = np.zeros((n_units, len(dark_model_t)))
                bins = np.append(dark_model_t, dark_model_t[-1]+self.model_dt)
                i = 0
                for ind, row in session_data.iterrows():
                    dark_model_nsp[i,:], bins = np.histogram(row['FmDk_spikeT'], bins)
                    unit_active_spikes = dark_model_nsp[i, dark_active]
                    unit_stationary_spikes = dark_model_nsp[i, ~dark_active]
                    self.data.at[ind,'FmDk_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*self.model_dt)
                    self.data.at[ind,'FmDk_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*self.model_dt)
                    i += 1

                active_time_by_session.setdefault('dark', {})[session] = np.sum(dark_active) / len(dark_active)
                dark_len.append(len(dark_active))

        return active_time_by_session, light_len, dark_len

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = np.nanargmin(np.abs(array - value))
        return array[idx]

    def summarize_sessions(self, do_session_props=False):
        pdf = PdfPages(os.path.join(self.savepath, 'session_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))

        if 'FmDk_theta' in self.data.columns:
            self.data['has_dark'] = ~self.data['FmDk_theta'].isna()
        else:
            self.data['has_dark'] = False
        
        if 'Wn_contrast_tuning' in self.data.columns:
            self.data['has_hf'] = ~self.data['Wn_contrast_tuning'].isna()
        else:
            self.data['has_hf'] = False

        if do_session_props:
            print('session property comparisons')
            
            if self.data['has_dark'].sum() > 0 and self.data['has_hf'].sum() > 0:
                active_time_by_session, light_len, dark_len = self.get_animal_activity()

                # fraction active time: light vs dark
                light = np.array([val for key,val in active_time_by_session['light'].items()])
                light_err = np.std(light) / np.sqrt(len(light))
                dark = np.array([val for key,val in active_time_by_session['dark'].items()])
                dark_err = np.std(dark) / np.sqrt(len(dark))
                fig, ax = plt.subplots(1,1,figsize=(3,5))
                plt.bar(0, np.mean(light), yerr=light_err, width=0.5, color='yellow')
                plt.plot(np.zeros(len(light)), light, 'o', color='tab:gray')
                plt.bar(1, np.mean(dark), yerr=dark_err, width=0.5, color='cadetblue')
                plt.plot(np.ones(len(dark)), dark, 'o', color='tab:gray')
                ax.set_xticks([0,1])
                ax.set_xticklabels(['light','dark'])
                plt.ylim([0,1])
                plt.ylabel('fraction of time spent active')
                plt.tight_layout(); pdf.savefig(); plt.close()

                # fraction active time: light vs dark (broken up by session)
                dark_active_times = [active_frac for session, active_frac in active_time_by_session['dark'].items()]
                dark_session_names = [session for session, active_frac in active_time_by_session['dark'].items()]
                fig, ax = plt.subplots(1,1, figsize=(5,10))
                plt.bar(np.arange(0, len(dark_session_names)), dark_active_times, color='cadetblue')
                ax.set_xticks(np.arange(0, len(dark_session_names)))
                ax.set_xticklabels(dark_session_names, rotation=90)
                plt.ylabel('frac active time')
                plt.tight_layout(); pdf.savefig(); plt.close()

                light_active_times = [active_frac for session, active_frac in active_time_by_session['light'].items()]
                light_session_names = [session for session, active_frac in active_time_by_session['light'].items()]
                fig, ax = plt.subplots(1,1, figsize=(12,10))
                plt.bar(np.arange(0, len(light_session_names)), light_active_times, color='khaki')
                ax.set_xticks(np.arange(len(light_session_names)))
                ax.set_xticklabels(light_session_names, rotation=90)
                plt.ylabel('frac active time'); plt.ylim([0,1])
                plt.tight_layout(); pdf.savefig(); plt.close()

                # minutes active or stationary: light vs dark
                total_min = [(i*self.model_dt)/60 for i in light_len]
                frac_active = [active_frac for session, active_frac in active_time_by_session['light'].items()]
                light_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
                light_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]
                light_session_names = [session for session, active_frac in active_time_by_session['light'].items()]
                fig, ax = plt.subplots(1,1, figsize=(12,10))
                plt.bar(np.arange(0, len(light_session_names)), light_active_min, color='salmon', label='active')
                plt.bar(np.arange(0, len(light_session_names)), light_stationary_min, bottom=light_active_min, color='gray', label='stationary')
                ax.set_xticks(np.arange(len(light_session_names)))
                ax.set_xticklabels(light_session_names, rotation=90)
                plt.legend()
                plt.ylabel('recording time (min)')
                plt.tight_layout(); pdf.savefig(); plt.close()

                total_min = [(i*self.model_dt)/60 for i in dark_len]
                frac_active = [active_frac for session, active_frac in active_time_by_session['dark'].items()]
                dark_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
                dark_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]
                dark_session_names = [session for session, active_frac in active_time_by_session['dark'].items()]
                fig, ax = plt.subplots(1,1, figsize=(12,10))
                plt.bar(np.arange(0, len(dark_session_names)), dark_active_min, color='salmon', label='active')
                plt.bar(np.arange(0, len(dark_session_names)), dark_stationary_min, bottom=dark_active_min, color='gray', label='stationary')
                ax.set_xticks(np.arange(len(dark_session_names)))
                ax.set_xticklabels(dark_session_names, rotation=90)
                plt.legend()
                plt.ylabel('recording time (min)')
                plt.tight_layout(); pdf.savefig(); plt.close()

            movement_count_dict = dict()
            session_stim_list = []
            if self.data['has_dark'].sum() > 0:
                session_stim_list.append('FmDk')
            if self.data['has_hf'].sum() > 0:
                session_stim_list.append('FmLt')

            for base in session_stim_list:
                for movement in ['eye_gaze_shifting', 'eye_comp']:
                    sessions = [i for i in self.data['session'].unique() if type(i) != float]
                    n_sessions = len(self.data['session'].unique())
                    trange = np.arange(-1,1.1,0.025)
                    for session_num, session_name in enumerate(sessions):
                        row = self.data[self.data['session']==session_name].iloc[0]

                        eyeT = np.array(row[base+'_eyeT'])
                        dEye = row[base+'_dEye_dps']
                        dhead = row[base+'_dHead']
                        dgz = dEye + dhead

                        if movement=='eye_gaze_shifting':
                            sthresh = 5
                            rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
                            leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
                        elif movement=='eye_comp':
                            sthresh = 3
                            rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
                            leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
                        elif movement=='head_gaze_shifting':
                            sthresh = 3
                            rightsacc = eyeT[(np.append(dhead,0)>sthresh) & (np.append(dgz,0)>sthresh)]
                            leftsacc = eyeT[(np.append(dhead,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
                        elif movement=='head_comp':
                            sthresh = 3
                            rightsacc = eyeT[(np.append(dhead,0)>sthresh) & (np.append(dgz,0)<1)]
                            leftsacc = eyeT[(np.append(dhead,0)<-sthresh) & (np.append(dgz,0)>-1)]

                        deye_mov_right = np.zeros([len(rightsacc), len(trange)]); deye_mov_left = np.zeros([len(leftsacc), len(trange)])
                        dgz_mov_right = np.zeros([len(rightsacc), len(trange)]); dgz_mov_left = np.zeros([len(leftsacc), len(trange)])
                        dhead_mov_right = np.zeros([len(rightsacc), len(trange)]); dhead_mov_left = np.zeros([len(leftsacc), len(trange)])

                        for sind in range(len(rightsacc)):
                            s = rightsacc[sind]
                            mov_ind = np.where([eyeT==self.find_nearest(eyeT, s)])[1]
                            trange_inds = list(mov_ind + np.arange(-42,42))
                            if np.max(trange_inds) < len(dEye):
                                deye_mov_right[sind,:] = dEye[np.array(trange_inds)]
                            if np.max(trange_inds) < len(dgz):
                                dgz_mov_right[sind,:] = dgz[np.array(trange_inds)]
                            if np.max(trange_inds) < len(dhead):
                                dhead_mov_right[sind,:] = dhead[np.array(trange_inds)]
                        for sind in range(len(leftsacc)):
                            s = leftsacc[sind]
                            mov_ind = np.where([eyeT==self.find_nearest(eyeT, s)])[1]
                            trange_inds = list(mov_ind + np.arange(-42,42))
                            if np.max(trange_inds) < len(dEye):
                                deye_mov_left[sind,:] = dEye[np.array(trange_inds)]
                            if np.max(trange_inds) < len(dgz):
                                dgz_mov_left[sind,:] = dgz[np.array(trange_inds)]
                            if np.max(trange_inds) < len(dhead):
                                dhead_mov_left[sind,:] = dhead[np.array(trange_inds)]

                        movement_count_dict.setdefault(base, {}).setdefault(movement, {}).setdefault(session_name, {})['right'] = len(rightsacc)
                        movement_count_dict.setdefault(base, {}).setdefault(movement, {}).setdefault(session_name, {})['left'] = len(leftsacc)

            if np.sum(self.data['has_dark']) > 0:
                right_gaze = [val['right'] for key,val in movement_count_dict['FmLt']['eye_gaze_shifting'].items()]
                left_gaze = [val['left'] for key,val in movement_count_dict['FmLt']['eye_gaze_shifting'].items()]

                right_comp = [val['right'] for key,val in movement_count_dict['FmLt']['eye_comp'].items()]
                left_comp = [val['left'] for key,val in movement_count_dict['FmLt']['eye_comp'].items()]

                right_gaze_dark = [val['right'] for key,val in movement_count_dict['FmDk']['eye_gaze_shifting'].items()]
                left_gaze_dark = [val['left'] for key,val in movement_count_dict['FmDk']['eye_gaze_shifting'].items()]

                right_comp_dark = [val['right'] for key,val in movement_count_dict['FmDk']['eye_comp'].items()]
                left_comp_dark = [val['left'] for key,val in movement_count_dict['FmDk']['eye_comp'].items()]

                # number of eye movements during recording: light vs dark (broken up by session)            
            # number of eye movements during recording: light vs dark (broken up by session)            
                # number of eye movements during recording: light vs dark (broken up by session)            
                x = np.arange(len(['gaze-shifting', 'compensatory']))
                width = 0.35

                fig, ax = plt.subplots(figsize=(4,7))

                ax.bar(x - width/2, np.mean(right_gaze), width, color='lightcoral')
                ax.bar(x - width/2, np.mean(left_gaze), width, bottom=np.mean(right_gaze), color='lightsteelblue')
                plt.plot(np.ones(len(right_gaze))*(0 - width/2), np.add(right_gaze, left_gaze), '.', color='gray')

                ax.bar(x + width/2, np.mean(right_gaze_dark), width, color='lightcoral')
                ax.bar(x + width/2, np.mean(left_gaze_dark), width, bottom=np.mean(right_gaze_dark), color='lightsteelblue')
                plt.plot(np.ones(len(right_gaze_dark))*(0 + width/2), np.add(right_gaze_dark, left_gaze_dark), '.', color='gray')

                ax.bar(x - width/2, np.mean(right_comp), width, color='lightcoral')
                ax.bar(x - width/2, np.mean(left_comp), width, bottom=np.mean(right_comp), color='lightsteelblue')
                plt.plot(np.ones(len(right_comp))*(1 - width/2), np.add(right_comp, left_comp), '.', color='gray')

                ax.bar(x + width/2, np.mean(right_comp_dark), width, color='lightcoral')
                ax.bar(x + width/2, np.mean(left_comp_dark), width, bottom=np.mean(right_comp_dark), color='lightsteelblue')
                plt.plot(np.ones(len(right_comp_dark))*(1 + width/2), np.add(right_comp_dark, left_comp_dark), '.', color='gray')

                ax.set_xticks(x)
                ax.set_xticklabels(['gaze-shifting', 'compensatory'])
                plt.ylim([0,3700]); plt.ylabel('number of eye movements')
                plt.tight_layout(); pdf.savefig(); plt.close()

                total_min = [(i*self.model_dt)/60 for i in light_len]
                frac_active = [active_frac for session, active_frac in active_time_by_session['light'].items()]
                light_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
                light_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]

                # number of eye movements per minute of active time: light vs dark (broken up by session)
                fig = plt.subplots(2,1,figsize=(10,15))
                ax = plt.subplot(2,1,1)
                ax.bar(light_session_names, np.add(right_gaze, left_gaze) / light_active_min)
                ax.set_xticklabels(light_session_names, rotation=90); plt.ylim([0,220]); plt.ylabel('eye movements per min during active periods'); plt.title('light stim')
                ax = plt.subplot(2,1,2)
                ax.bar(dark_session_names, np.add(right_gaze_dark, left_gaze_dark) / dark_active_min, width=0.3)
                ax.set_xticklabels(dark_session_names, rotation=90); plt.ylim([0,220]); plt.ylabel('eye movements per min during active periods'); plt.title('dark stim')
                plt.tight_layout(); pdf.savefig(); plt.close()

        session_data = self.data.set_index('session')
        unique_inds = sorted(list(set(session_data.index.values)))
        
        for unique_ind in tqdm(unique_inds):
            uniquedf = session_data.loc[unique_ind]

            fmt_m = str(np.round(uniquedf['best_ellipse_fit_m'].iloc[0],4))
            fmt_r = str(np.round(uniquedf['best_ellipse_fit_r'].iloc[0],4))

            plt.subplots(5,5,figsize=(40,30))

            plt.subplot(5,5,1)
            plt.title(unique_ind+' eye fit: m='+fmt_m+' r='+fmt_r, fontsize=20)
            dEye = uniquedf['FmLt_dEye_dps'].iloc[0]
            dHead = uniquedf['FmLt_dHead'].iloc[0]
            eyeT = uniquedf['FmLt_eyeT'].iloc[0]
            plt.plot(dEye[::10], dHead[::10], 'k.')
            plt.xlabel('dEye (deg/sec)', fontsize=20); plt.ylabel('dHead (deg/sec)', fontsize=20)
            plt.xlim((-700,700)); plt.ylim((-700,700))
            plt.plot([-700,700],[700,-700], 'r:')

            imuT = uniquedf['FmLt_imuT'].iloc[0]
            roll = uniquedf['FmLt_roll'].iloc[0]
            pitch = uniquedf['FmLt_pitch'].iloc[0]

            centered_roll = roll - np.mean(roll)
            roll_interp = interp1d(imuT, centered_roll, bounds_error=False)(eyeT)

            centered_pitch = pitch - np.mean(pitch)
            pitch_interp = interp1d(imuT, centered_pitch, bounds_error=False)(eyeT)

            th = uniquedf['FmLt_theta'].iloc[0]
            phi = uniquedf['FmLt_phi'].iloc[0]
            plt.subplot(5,5,2)
            plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch (deg)', fontsize=20); plt.ylabel('theta (deg)', fontsize=20)
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            
            plt.subplot(5,5,3)
            plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll (deg)', fontsize=20); plt.ylabel('phi (deg)', fontsize=20)
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
            
            # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(5,5,4)
            plt.hist(uniquedf['FmLt_theta'].iloc[0], range=[-45,45], alpha=0.5); plt.xlabel('FmLt theta (deg)', fontsize=20)
            # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(5,5,5)
            plt.hist(uniquedf['FmLt_phi'].iloc[0], range=[-45,45], alpha=0.5); plt.xlabel('FmLt phi (deg)', fontsize=20)
            # histogram of gyro z (resonable range?)
            plt.subplot(5,5,6)
            plt.hist(uniquedf['FmLt_gyro_z'].iloc[0], range=[-400,400], alpha=0.5); plt.xlabel('FmLt gyro z (deg)', fontsize=20)
            # plot of contrast response functions on same panel scaled to max 30sp/sec
            # plot of average contrast reponse function across units
            plt.subplot(5,5,7)
            if uniquedf['has_hf'].iloc[0]:
                for ind, row in uniquedf.iterrows():
                    plt.errorbar(row['Wn_contrast_tuning_bins'], row['Wn_contrast_tuning'], yerr=row['Wn_contrast_tuning_err'], alpha=0.5, linewidth=4)
                plt.ylim(0,30); plt.xlabel('contrast a.u.', fontsize=20); plt.ylabel('sp/sec', fontsize=20); plt.title('hf contrast tuning', fontsize=20)
                plt.errorbar(uniquedf['Wn_contrast_tuning_bins'].iloc[0], np.mean(uniquedf['Wn_contrast_tuning'], axis=0),yerr=np.mean(uniquedf['Wn_contrast_tuning_err'],axis=0), color='k', linewidth=6)
                # lfp traces as separate shanks
                colors = plt.cm.jet(np.linspace(0,1,32))
                num_channels = np.size(uniquedf['Rc_response_by_channel'].iloc[0],0)
                if num_channels == 64:
                    for ch_num in np.arange(0,64):
                        if ch_num<=31:
                            plt.subplot(5,5,8)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                            plt.title('shank0', fontsize=20); plt.axvline(x=(0.1*30000))
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                            plt.ylim([-1200,400]); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                        if ch_num>31:
                            plt.subplot(5,5,9)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                            plt.title('shank1', fontsize=20); plt.axvline(x=(0.1*30000))
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                            plt.ylim([-1200,400]); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                    plt.subplot(5,5,10); plt.axis('off')
                    plt.subplot(5,5,11); plt.axis('off')
                elif num_channels == 128:
                    for ch_num in np.arange(0,128):
                        if ch_num < 32:
                            plt.subplot(5,5,8)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                            plt.title('shank0'); plt.axvline(x=(0.1*30000)); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        elif 32 <= ch_num < 64:
                            plt.subplot(5,5,9)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                            plt.title('shank1'); plt.axvline(x=(0.1*30000)); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        elif 64 <= ch_num < 10:
                            plt.subplot(5,5,10)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num-64], linewidth=1)
                            plt.title('shank2'); plt.axvline(x=(0.1*30000)); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        elif 96 <= ch_num < 128:
                            plt.subplot(5,5,11)
                            plt.plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num], color=colors[ch_num-96], linewidth=1)
                            plt.title('shank3'); plt.axvline(x=(0.1*30000)); plt.xlabel('msec', fontsize=20); plt.ylabel('uvolts', fontsize=20)
                            plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
            
            # fm spike raster
            plt.subplot(5,5,12)
            i = 0
            for ind, row in uniquedf.iterrows():
                plt.vlines(row['FmLt_spikeT'],i-0.25,i+0.25)
                plt.xlim(0, 10); plt.xlabel('sec', fontsize=20); plt.ylabel('unit #', fontsize=20)
                i = i+1

            if uniquedf['has_hf'].iloc[0]:
                plt.subplot(5,5,13)
                try:
                    lower = -0.5; upper = 1.5; dt = 0.1
                    bins = np.arange(lower,upper+dt,dt)
                    psth_list = []
                    for ind, row in uniquedf.iterrows():
                        plt.plot(bins[0:-1]+dt/2,row['Gt_grating_psth'])
                        psth_list.append(row['Gt_grating_psth'])
                    avg_psth = np.mean(np.array(psth_list), axis=0)
                    plt.plot(bins[0:-1]+dt/2,avg_psth,color='k',linewidth=6)
                    plt.title('gratings psth', fontsize=20); plt.xlabel('sec', fontsize=20); plt.ylabel('sp/sec', fontsize=20)
                    plt.ylim([0,np.nanmax(avg_psth)*1.5])
                except:
                    pass

                lfp_power_profile = uniquedf['Wn_lfp_power'].iloc[0]
                layer5_cent = uniquedf['Wn_layer5cent_from_lfp'].iloc[0]
                if type(lfp_power_profile) == list:
                    if uniquedf['probe_name'].iloc[0] == 'DB_P64-8':
                        ch_spacing = 25/2
                    else:
                        ch_spacing = 25
                    if '64' in uniquedf['probe_name'].iloc[0]:
                        norm_profile_sh0 = lfp_power_profile[0]
                        layer5_cent_sh0 = layer5_cent[0]
                        norm_profile_sh1 = lfp_power_profile[1]
                        layer5_cent_sh1 = layer5_cent[1]
                        plt.subplot(5,5,14)
                        plt.plot(norm_profile_sh0,range(0,32))
                        plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                        plt.title('shank0', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,15)
                        plt.plot(norm_profile_sh1,range(0,32))
                        plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                        plt.title('shank1', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,16); plt.axis('off')
                        plt.subplot(5,5,17); plt.axis('off')
                    if '16' in uniquedf['probe_name'].iloc[0]:
                        norm_profile_sh0 = lfp_power_profile[0]
                        layer5_cent_sh0 = layer5_cent[0]
                        plt.subplot(5,5,14)
                        plt.tight_layout()
                        plt.plot(norm_profile_sh0,range(0,16))
                        plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                        plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
                        plt.title('shank0', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,15); plt.axis('off')
                        plt.subplot(5,5,16); plt.axis('off')
                        plt.subplot(5,5,17); plt.axis('off')
                    if '128' in uniquedf['probe_name'].iloc[0]:
                        norm_profile_sh0 = lfp_power_profile[0]
                        layer5_cent_sh0 = layer5_cent[0]
                        norm_profile_sh1 = lfp_power_profile[1]
                        layer5_cent_sh1 = layer5_cent[1]
                        norm_profile_sh2 = lfp_power_profile[2]
                        layer5_cent_sh2 = layer5_cent[2]
                        norm_profile_sh3 = lfp_power_profile[3]
                        layer5_cent_sh3 = layer5_cent[3]
                        plt.subplot(5,5,14)
                        plt.plot(norm_profile_sh0,range(0,32))
                        plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                        plt.title('shank0', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,15)
                        plt.plot(norm_profile_sh1,range(0,32))
                        plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                        plt.title('shank1', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,16)
                        plt.plot(norm_profile_sh2,range(0,32))
                        plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
                        plt.title('shank2', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)
                        plt.subplot(5,5,17)
                        plt.plot(norm_profile_sh3,range(0,32))
                        plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
                        plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
                        plt.title('shank3', fontsize=20); plt.ylabel('depth relative to layer 5', fontsize=20); plt.xlabel('norm mua power', fontsize=20)

            if not uniquedf['has_dark'].iloc[0]:
                plt.subplot(5,5,18); plt.axis('off')
                plt.subplot(5,5,19); plt.axis('off')
                plt.subplot(5,5,20); plt.axis('off')

            elif uniquedf['has_dark'].iloc[0]:
                plt.subplot(5,5,18)

                imuT_dk = uniquedf['FmDk_imuT'].iloc[0]
                roll_dk = uniquedf['FmDk_roll'].iloc[0]
                pitch_dk = uniquedf['FmDk_pitch'].iloc[0]
                dHead_dk = uniquedf['FmDk_dHead'].iloc[0]

                theta_dk = uniquedf['FmDk_theta'].iloc[0]
                phi_dk = uniquedf['FmDk_phi'].iloc[0]
                eyeT_dk = uniquedf['FmDk_eyeT'].iloc[0]
                imuT_dk = uniquedf['FmDk_imuT'].iloc[0]
                dEye_dk = uniquedf['FmDk_dEye_dps'].iloc[0]

                centered_roll_dk = roll_dk - np.mean(roll_dk)
                roll_dk_interp = interp1d(imuT_dk, centered_roll_dk, bounds_error=False)(eyeT_dk)

                centered_pitch_dk = pitch_dk - np.mean(pitch_dk)
                pitch_dk_interp = interp1d(imuT_dk, centered_pitch_dk, bounds_error=False)(eyeT_dk)


                plt.plot(dEye_dk[::10], dHead_dk[::10], 'k.')
                plt.xlabel('dark dEye (deg)', fontsize=20); plt.ylabel('dark dHead (deg)', fontsize=20)
                plt.xlim((-700,700)); plt.ylim((-700,700))
                plt.plot([-700,700],[700,-700], 'r:')

                plt.subplot(5,5,19)
                plt.plot(pitch_dk_interp[::100], theta_dk[::100], '.'); plt.xlabel('dark pitch (deg)', fontsize=20)
                plt.ylabel('dark theta (deg)', fontsize=20)
                plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
                
                plt.subplot(5,5,20)
                plt.plot(roll_dk_interp[::100], phi_dk[::100], '.'); plt.xlabel('dark roll (deg)', fontsize=20)
                plt.ylabel('dark phi (deg)', fontsize=20)
                plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')

            plt.tight_layout(); pdf.savefig(); plt.close()

        pdf.close()

    def putative_celltype(self):
        self.data['norm_waveform'] = self.data['waveform']
        for ind, row in self.data.iterrows():
            if type(row['waveform']) == list:
                starting_val = np.mean(row['waveform'][:6])
                center_waveform = [i-starting_val for i in row['waveform']]
                norm_waveform = center_waveform / -np.min(center_waveform)
                self.data.at[ind, 'waveform_trough_width'] = len(norm_waveform[norm_waveform < -0.2])
                self.data.at[ind, 'AHP'] = norm_waveform[27]
                self.data.at[ind, 'waveform_peak'] = norm_waveform[18]
                self.data.at[ind, 'norm_waveform'] = norm_waveform

        km_labels = KMeans(n_clusters=2).fit(list(self.data['norm_waveform'][self.data['waveform_peak'] < 0].to_numpy())).labels_
        # make inhibitory is always group 0
        # excitatory should always have a smaller mean waveform trough
        # if it's larger, flip the kmeans labels
        if np.mean(self.data['waveform_trough_width'][self.data['waveform_peak']<0][km_labels==0]) > np.mean(self.data['waveform_trough_width'][self.data['waveform_peak']<0][km_labels==1]):
            km_labels = [0 if i==1 else 1 for i in km_labels]
        
        count = 0
        for ind, row in self.data.iterrows():
            if row['waveform_peak'] < 0 and row['AHP'] < 0.7:
                self.data.at[ind, 'waveform_km_label'] = km_labels[count]
                count = count+1

        # make new column of strings for excitatory vs inhibitory clusters
        for ind, row in self.data.iterrows():
            if row['waveform_km_label'] == 0:
                self.data.at[ind, 'exc_or_inh'] = 'inh'
            elif row['waveform_km_label'] == 1:
                self.data.at[ind, 'exc_or_inh'] = 'exc'

    def plot_running_average(self, xvar, yvar, n, filter_for=None, force_range=None, along_y=False,
                            use_median=False, abs=False, show_legend=False):
        fig = plt.subplot(n[0],n[1],n[2])
        if force_range is None:
            force_range = np.arange(0,0.40,0.05)
        for count, exc_or_inh in enumerate(['inh','exc']):
            if exc_or_inh == 'inh':
                c = 'g'
            elif exc_or_inh == 'exc':
                c = 'b'
            x = self.data[xvar][self.data['exc_or_inh']==exc_or_inh]
            if abs==True:
                x = np.abs(x)
            y = self.data[yvar][self.data['exc_or_inh']==exc_or_inh]
            if filter_for is not None:
                for key, val in filter_for.items():
                    x = x[self.data[key]==val]
                    y = y[self.data[key]==val]
            x = x.to_numpy().astype(float)
            y = y.to_numpy().astype(float)
            if use_median == False:
                stat2use = np.nanmean
            elif use_median == True:
                stat2use = np.nanmedian
            if along_y == False:
                bin_means, bin_edges, bin_number = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
                bin_std, _, _ = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=force_range)
                hist, _ = np.histogram(x[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
            elif along_y == True:
                bin_means, bin_edges, bin_number = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
                bin_std, _, _ = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=force_range)
                hist, _ = np.histogram(y[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
            tuning_err = bin_std / np.sqrt(hist)
            if along_y == False:
                plt.plot(x, y, c+'.', markersize=2)
                plt.plot(bin_edges[:-1], bin_means, c+'-')
                plt.fill_between(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
                num_outliers = len([i for i in x if i>np.max(force_range) or i<np.min(force_range)])
                plt.xlim([np.min(force_range), np.max(force_range)])
            elif along_y == True:
                plt.plot(x, y, c+'.', markersize=2)
                plt.plot(bin_means, bin_edges[:-1], c+'-')
                plt.fill_betweenx(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
                num_outliers = len([i for i in y if i>np.max(force_range) or i<np.min(force_range)])
                plt.ylim([np.max(force_range), np.min(force_range)])
        plt.title('excluded='+str(num_outliers)+' pts in data='+str(np.sum(~pd.isnull(self.data[xvar]) & ~pd.isnull(self.data[yvar])))+' abs='+str(abs))
        if show_legend:
            plt.legend(handles=[self.bluepatch, self.greenpatch])
        return fig

    def neural_response_to_contrast(self):
        for ind, row in self.data.iterrows():
            tuning = row['Wn_contrast_tuning']
            if type(tuning) == np.ndarray or type(tuning) == list:
                tuning = [x for x in tuning if x != None]
                # thresh out units which have a small response to contrast, even if the modulation index is large
                self.data.at[ind, 'responsive_to_contrast'] = np.abs(row['Wn_contrast_modind']) > 0.33
            else:
                self.data.at[ind, 'responsive_to_contrast'] = False
        if np.sum(self.data['has_hf'])>1:
            self.depth_range = [np.max(self.data['Wn_depth_from_layer5'][self.data['responsive_to_contrast']==True])+50, np.min(self.data['Wn_depth_from_layer5'][self.data['responsive_to_contrast']==True])+50]

        for i, x in self.data['Wn_contrast_tuning'].items():
            if type(x) == str:
                x = np.array([np.nan if i=='nan' else i for i in list(x.split(' ')[1:-2])])
            if type(x) != float:
                self.data.at[i, 'Wn_spont_rate'] = x[0]
                self.data.at[i, 'Wn_max_contrast_rate'] = x[-1]
                self.data.at[i, 'Wn_evoked_rate'] = x[-1] - x[0]
            else:
                self.data.at[i, 'Wn_spont_rate'] = np.nan
                self.data.at[i, 'Wn_max_contrast_rate'] = np.nan
                self.data.at[i, 'Wn_evoked_rate'] = np.nan

    def neural_response_to_gratings(self):
        for sf in ['low','mid','high']:
            self.data['norm_ori_tuning_'+sf] = self.data['Gt_ori_tuning_tf'].copy().astype(object)
        for ind, row in self.data.iterrows():
            try:
                orientations = np.nanmean(np.array(row['Gt_ori_tuning_tf'], dtype=np.float),2)
                for sfnum in range(3):
                    sf = ['low','mid','high'][sfnum]
                    self.data.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['Gt_drift_spont']
                mean_for_sf = np.array([np.mean(self.data.at[ind,'norm_ori_tuning_low']), np.mean(self.data.at[ind,'norm_ori_tuning_mid']), np.mean(self.data.at[ind,'norm_ori_tuning_high'])])
                mean_for_sf[mean_for_sf<0] = 0
                self.data.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)
                self.data.at[ind,'responsive_to_gratings'] = (True if np.max(mean_for_sf)>2 else False)
            except:
                for sfnum in range(3):
                    sf = ['low','mid','high'][sfnum]
                    self.data.at[ind,'norm_ori_tuning_'+sf] = None
                self.data.at[ind,'responsive_to_gratings'] = False
                self.data.at[ind,'sf_pref'] = np.nan

        self.data['osi_for_sf_pref'] = np.nan
        self.data['dsi_for_sf_pref'] = np.nan
        for ind, row in self.data.iterrows():
            if ~np.isnan(row['sf_pref']):
                best_sf_pref = int(np.round(row['sf_pref']))
                self.data.at[ind, 'osi_for_sf_pref'] = row[(['Gt_osi_low','Gt_osi_mid','Gt_osi_high'][best_sf_pref-1])]
                self.data.at[ind, 'dsi_for_sf_pref'] = row[(['Gt_dsi_low','Gt_dsi_mid','Gt_dsi_high'][best_sf_pref-1])]

        self.data['osi_for_sf_pref'][self.data['osi_for_sf_pref']<0] = 0
        self.data['dsi_for_sf_pref'][self.data['dsi_for_sf_pref']<0] = 0
                
        for ind, row in self.data.iterrows():
            try:
                mean_for_sf = np.array([np.mean(self.data.at[ind,'norm_ori_tuning_low']), np.mean(self.data.at[ind,'norm_ori_tuning_mid']), np.mean(self.data.at[ind,'norm_ori_tuning_high'])])
                mean_for_sf[mean_for_sf<0] = 0
                self.data.at[ind, 'Gt_evoked_rate'] = np.max(mean_for_sf) - row['Gt_drift_spont']
            except:
                pass

        for ind, row in self.data.iterrows():
            if type(row['Gt_ori_tuning_tf']) != float:
                tuning = np.nanmean(row['Gt_ori_tuning_tf'],1)
                tuning = tuning - row['Gt_drift_spont']
                tuning[tuning < 0] = 0
                mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
                tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
                self.data.at[ind, 'tf_pref'] = tf_pref

        self.data['tf_pref_cps'] = 2 + (6 * (row['tf_pref']-1))
        self.data['sf_pref_cpd'] = 0.02 * 4 ** (row['sf_pref']-1)
        self.data['grat_speed_dps'] = self.data['tf_pref_cps'] / self.data['sf_pref_cpd']

    def spike_rate_by_stim(self):
        self.get_animal_activity()
        for ind, row in self.data.iterrows():
            if self.is_empty_cell(row, 'FmLt_spikeT'):
                self.data.at[ind,'FmLt_rec_rate'] = len(row['FmLt_spikeT']) / (row['FmLt_spikeT'][-1] - row['FmLt_spikeT'][0])
            if self.is_empty_cell(row, 'FmDk_spikeT'):
                self.data.at[ind,'FmDk_rec_rate'] = len(row['FmDk_spikeT']) / (row['FmDk_spikeT'][-1] - row['FmDk_spikeT'][0])
            if self.is_empty_cell(row, 'Gt_spikeT'):
                self.data.at[ind,'Gt_rec_rate'] = len(row['Gt_spikeT']) / (row['Gt_spikeT'][-1] - row['Gt_spikeT'][0])
            if self.is_empty_cell(row, 'Wn_spikeT'):
                self.data.at[ind,'Wn_rec_rate'] = len(row['Wn_spikeT']) / (row['Wn_spikeT'][-1] - row['Wn_spikeT'][0])

    def neural_response_to_movement(self):

        for ind, row in self.data.iterrows():
            if type(row['FmLt_spikeT']) != float:
                self.data.at[ind, 'fires_2sp_sec'] = (True if (len(row['FmLt_spikeT'])/np.nanmax(row['FmLt_eyeT']))>2 else False)
            else:
                self.data.at[ind, 'fires_2sp_sec'] = False

    def get_peak_trough(self, wv, baseline):
        wv = [i-baseline for i in wv]
        wv_flip = [-i for i in wv]
        peaks, peak_props = find_peaks(wv, height=0.18)
        troughs, trough_props = find_peaks(wv_flip, height=0.18)
        if len(peaks) > 1:
            peaks = peaks[np.argmax(peak_props['peak_heights'])]
        if len(troughs) > 1:
            troughs = troughs[np.argmax(trough_props['peak_heights'])]
        if peaks.size == 0:
            peaks = np.nan
        if troughs.size == 0:
            troughs = np.nan
        if ~np.isnan(peaks):
            peaks = int(peaks)
        if ~np.isnan(troughs):
            troughs = int(troughs)
        return peaks, troughs

    def get_cluster_props(self, p, t):
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
        if has_peak and has_trough:
            return 'biphasic'
        elif has_trough and ~has_peak:
            return 'negative'
        elif peak_cent is not None and peak_cent <= 5.39:
            return 'early'
        elif peak_cent is not None and peak_cent > 5.39:
            return 'late'
        else:
            return 'unresponsive'

    def comparative_z_score(self, a, b):
        return [(np.max(np.abs(a))-np.mean(a)) / np.std(a), (np.max(np.abs(b))-np.mean(b)) / np.std(b)]

    def split_cluster_by_compensatory_modulation(self, cluster_name, save_key):
        this_cluster = flatten_series(self.data['nonpref_comp_psth'][self.data['sacccluster']==cluster_name])
        km_labels = KMeans(n_clusters=2).fit(this_cluster).labels_
        cluster0mean = np.nanmean(this_cluster[km_labels==0], 0)
        cluster1mean = np.nanmean(this_cluster[km_labels==1], 0)
        comp_responsive = np.argmax(self.comparative_z_score(cluster0mean, cluster1mean))
        inds = self.data['nonpref_comp_psth'][self.data['sacccluster']==cluster_name].index.values
        for i in range(np.size(this_cluster, 0)):
            real_ind = inds[i]
            self.data.at[real_ind, save_key] = (True if km_labels[i]==comp_responsive else False)

    def modind(self, a, b):
        """
        value of 1 means a is more significant
        value of -1 means b is more significant
        """
        mi = (a - b) / (a + b)
        return mi

    def calc_direction_pref_index(self):
        for ind, row in self.data[self.data['fr']>2].iterrows():
            if row['sacccluster'] in ['early','late','biphasic','negative']:
                if row['pref_gazeshift_direction']=='L':
                    pref_gaze = row['pref_gazeshift_psth'][self.trange_win[0]:self.trange_win[1]]
                    nonpref_gaze = row['nonpref_gazeshift_psth'][self.trange_win[0]:self.trange_win[1]]
                    pref_comp = row['pref_comp_psth'][self.trange_win[0]:self.trange_win[1]]
                    nonpref_comp= row['nonpref_comp_psth'][self.trange_win[0]:self.trange_win[1]]
                elif row['pref_gazeshift_direction']=='R':
                    pref_gaze = row['pref_gazeshift_psth'][self.trange_win[0]:self.trange_win[1]]
                    nonpref_gaze = row['nonpref_gazeshift_psth'][self.trange_win[0]:self.trange_win[1]]
                    pref_comp = row['pref_comp_psth'][self.trange_win[0]:self.trange_win[1]]
                    nonpref_comp= row['nonpref_comp_psth'][self.trange_win[0]:self.trange_win[1]]
                
                if row['pref_gazeshift_direction']=='L':
                    left_gaze = pref_gaze; right_gaze = nonpref_gaze
                    left_comp = pref_comp; right_comp = nonpref_comp
                elif row['pref_gazeshift_direction']=='R':
                    right_gaze = pref_gaze; left_gaze = nonpref_gaze
                    right_comp = pref_comp; left_comp = nonpref_comp
                
                self.data.at[ind, 'gaze_sacc_rlMI'] = self.modind(np.max(right_gaze), np.max(left_gaze))
                self.data.at[ind, 'comp_sacc_rlMI'] = self.modind(np.max(right_comp), np.max(left_comp))
                self.data.at[ind, 'gaze_sacc_pnpMI'] = self.modind(np.max(pref_gaze), np.max(nonpref_gaze))
                self.data.at[ind, 'comp_sacc_pnpMI'] = self.modind(np.max(pref_comp), np.max(nonpref_comp))
            
        if self.exptype=='hffm' and row['responsive_to_gratings']:
            ori = row['Gt_ori_tuning_mean_tf']
            best_sf = ori[:,np.argmax(np.mean(ori,0))]
            left_grat = best_sf[0]
            right_grat = best_sf[4]
            self.data.at[ind, 'gratings_rlMI'] = self.modind(right_grat, left_grat)

    def label_movcluster(self, psth, baseline, hthresh=0.17, el_bound=0.075):
        """
        PSTH should be the neural response to eye movements
        between -0.0625 and 0.3125 sec, where 0 is the moment
        of the eye movement.
        """
        # baseline subtract PSTH
        psth = psth - baseline
        # flip sign of the PSTH so that we can find the troughts also
        revpsth = -psth.copy()
        
        # find peaks and troughs in PSTH
        p, peak_props = find_peaks(psth, height=hthresh)
        t, trough_props = find_peaks(revpsth, height=hthresh)
        
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
        el_bound_ind = np.argmin(np.abs(self.trange_twin-el_bound))
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

    def deye_clustering(self):
        zwin = [35,55]
        for ind, row in self.data.iterrows():
            # direction preference
            left_deflection = row['FmLt_leftsacc_avg_gaze_shift_dHead']
            right_deflection = row['FmLt_rightsacc_avg_gaze_shift_dHead']
            left_right_index = np.argmax(np.abs(self.comparative_z_score(left_deflection[zwin[0]:zwin[1]], right_deflection[zwin[0]:zwin[1]])))
            saccade_direction_pref = ['L','R'][left_right_index]
            self.data.at[ind, 'pref_gazeshift_direction'] = saccade_direction_pref; self.data.at[ind, 'pref_gazeshift_direction_ind'] = left_right_index
            self.data.at[ind, 'pref_gazeshift_raw_psth'] = ([row['FmLt_leftsacc_avg_gaze_shift_dHead'], row['FmLt_rightsacc_avg_gaze_shift_dHead']][int(left_right_index)]).astype(object)
            # direction preference for compensatory movements
            left_deflection = row['FmLt_leftsacc_avg_comp_dHead']
            right_deflection = row['FmLt_rightsacc_avg_comp_dHead']
            left_right_index = np.argmax(np.abs(self.comparative_z_score(left_deflection[zwin[0]:zwin[1]], right_deflection[zwin[0]:zwin[1]])))
            saccade_direction_pref = ['L','R'][left_right_index]
            self.data.at[ind, 'pref_comp_direction'] = saccade_direction_pref; self.data.at[ind, 'pref_comp_direction_ind'] = left_right_index
        for ind, row in self.data.iterrows():
            # more compensatory or more gaze-shifting?
            comp_deflection = [row['FmLt_leftsacc_avg_comp_dHead'],row['FmLt_rightsacc_avg_comp_dHead']][int(row['pref_comp_direction_ind'])]
            gazeshift_deflection = [row['FmLt_leftsacc_avg_gaze_shift_dHead'],row['FmLt_rightsacc_avg_gaze_shift_dHead']][int(row['pref_gazeshift_direction_ind'])]
            comp_gazeshift_zscores = [(np.max(np.abs(comp_deflection))-np.mean(comp_deflection)) / np.std(comp_deflection), (np.max(np.abs(gazeshift_deflection))-np.mean(gazeshift_deflection)) / np.std(gazeshift_deflection)]
            comp_gazeshift_index = np.argmax(np.abs(comp_gazeshift_zscores))
            is_more_gazeshift = ['comp','gaze_shift'][comp_gazeshift_index]
            self.data.at[ind, 'is_more_gazeshift'] = is_more_gazeshift
            
        self.data['pref_gazeshift_psth_FmDk'] = np.array(np.nan).astype(object)
        self.data['pref_comp_psth_FmDk'] = np.array(np.nan).astype(object)
        self.data['nonpref_gazeshift_psth_FmDk'] = np.array(np.nan).astype(object)
        self.data['nonpref_comp_psth_FmDk'] = np.array(np.nan).astype(object)

        for ind, row in self.data.iterrows():
            deflection_at_pref_direction = [row['FmLt_leftsacc_avg_gaze_shift_dHead'],row['FmLt_rightsacc_avg_gaze_shift_dHead']][int(row['pref_gazeshift_direction_ind'])]
            norm_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
            self.data.at[ind, 'pref_gazeshift_psth'] = norm_deflection.astype(object)

            deflection_at_pref_direction = [row['FmLt_leftsacc_avg_comp_dHead'],row['FmLt_rightsacc_avg_comp_dHead']][int(row['pref_gazeshift_direction_ind'])]
            norm_comp_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
            self.data.at[ind, 'pref_comp_psth'] = norm_comp_deflection.astype(object)

            deflection_at_pref_direction = [row['FmLt_leftsacc_avg_gaze_shift_dHead'],row['FmLt_rightsacc_avg_gaze_shift_dHead']][1-int(row['pref_gazeshift_direction_ind'])]
            norm_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
            self.data.at[ind, 'nonpref_gazeshift_psth'] = norm_deflection.astype(object)

            deflection_at_pref_direction = [row['FmLt_leftsacc_avg_comp_dHead'],row['FmLt_rightsacc_avg_comp_dHead']][1-int(row['pref_gazeshift_direction_ind'])]
            norm_comp_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
            self.data.at[ind, 'nonpref_comp_psth'] = norm_comp_deflection.astype(object)

            if np.sum(self.data['has_dark'])==len(self.data):
                dark_gaze_shift = [row['FmDk_leftsacc_avg_gaze_shift_dHead'],row['FmDk_rightsacc_avg_gaze_shift_dHead']][int(row['pref_gazeshift_direction_ind'])]
                dark_gaze_shift_norm = (dark_gaze_shift-np.nanmean(dark_gaze_shift)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
                dark_comp = [row['FmDk_leftsacc_avg_comp_dHead'],row['FmDk_rightsacc_avg_comp_dHead']][int(row['pref_gazeshift_direction_ind'])]
                dark_comp_norm = (dark_comp-np.nanmean(dark_comp)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))

                dark_gaze_shift_opp = [row['FmDk_leftsacc_avg_gaze_shift_dHead'],row['FmDk_rightsacc_avg_gaze_shift_dHead']][1-int(row['pref_gazeshift_direction_ind'])]
                dark_gaze_shift_norm_opp = (dark_gaze_shift_opp-np.nanmean(dark_gaze_shift_opp)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))
                dark_comp_opp = [row['FmDk_leftsacc_avg_comp_dHead'],row['FmDk_rightsacc_avg_comp_dHead']][1-int(row['pref_gazeshift_direction_ind'])]
                dark_comp_norm_opp = (dark_comp_opp-np.nanmean(dark_comp_opp)) / np.nanmax(np.abs(row['pref_gazeshift_raw_psth']))

                self.data.at[ind, 'pref_gazeshift_psth_FmDk'] = dark_gaze_shift_norm.astype(object)
                self.data.at[ind, 'pref_comp_psth_FmDk'] = dark_comp_norm.astype(object)
                self.data.at[ind, 'nonpref_gazeshift_psth_FmDk'] = dark_gaze_shift_norm_opp.astype(object)
                self.data.at[ind, 'nonpref_comp_psth_FmDk'] = dark_comp_norm_opp.astype(object)

        unit_modulations = flatten_series(self.data['pref_gazeshift_psth'])[:,self.trange_win[0]:self.trange_win[1]]
        pcas_cutoff = 7

        if self.exptype=='hffm':
            pca = PCA(n_components=15)
            pca.fit(unit_modulations)

            explvar = pca.explained_variance_

            # plt.figure()
            # plt.plot(explvar)
            # plt.xlabel('PCAs'); plt.ylabel('explained variance')
            # plt.vlines(pcas_cutoff, 0, explvar[0], colors='k', linestyle='dotted')
            # plt.tight_layout(); self.poppdf.savefig(); plt.close()

            proj = pca.transform(unit_modulations)
            gproj = proj[:,:pcas_cutoff]

            km = KMeans(n_clusters=5)
            km.fit_predict(gproj)
            Z = km.labels_

            with open(os.path.join(self.savepath,'dEye_PSTH_km_model.pickle'), 'wb') as f:
                pickle.dump(km, f)
            with open(os.path.join(self.savepath,'dEye_PSTH_pca_model.pickle'), 'wb') as f:
                pickle.dump(pca, f)

            np.save(file=os.path.join(self.savepath,'dEye_PSTH_pca.npy'), arr=proj)
        
        elif self.exptype=='ltdk':
            # load in the clustering model from the hf/fm experiment data
            with open(os.path.join(self.savepath,'dEye_PSTH_km_model.pickle'), 'rb') as f:
                km = pickle.load(f)
            with open(os.path.join(self.savepath,'dEye_PSTH_pca_model.pickle'), 'rb') as f:
                pca = pickle.load(f)
            
            proj = pca.transform(unit_modulations)
            gproj = proj[:,:pcas_cutoff]

            Z = km.predict(gproj)

        self.data['mov_kmclust'] = Z
        
        clustermeans = np.zeros([5,83])
        for l in range(5):
            clustermeans[l] = np.nanmean(flatten_series(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==l]),0)
        cluster_to_cell_type = dict()
        for l in range(5):
            mean_response = clustermeans[l, self.trange_win[0]:self.trange_win[1]]
            mean_baseline = np.nanmean(clustermeans[l, :self.trange_win[0]])
            cluster_to_cell_type[l] = self.label_movcluster(mean_response, mean_baseline)
        for ind, row in self.data.iterrows():
            self.data.at[ind, 'sacccluster'] = cluster_to_cell_type[row['mov_kmclust']]

        # plt.subplots(2,3, figsize=(15,10))
        mean_cluster = dict()
        for label in range(5):
            # plt.subplot(2,3,label+1)
            # plt.title('cluster='+str(label)+' count='+str(len(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==label].dropna())))
            # inhibitory = flatten_series(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==label][self.data['exc_or_inh']=='inh'])
            # for i in range(len(inhibitory)):
            #     plt.plot(self.trange_x, inhibitory[i], 'g', alpha=0.1, linewidth=1)
            # excitatory = flatten_series(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==label][self.data['exc_or_inh']=='exc'])
            # for i in range(len(excitatory)):
            #     plt.plot(self.trange_x, excitatory[i], 'b', alpha=0.1, linewidth=1)
            all_units = flatten_series(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==label])
            # plt.plot(self.trange_x, np.nanmean(all_units, axis=0), 'k', linewidth=3)
            # plt.xlim([-0.5,0.75]); plt.ylabel('norm spike rate'); plt.xlabel('sec')
            mean_cluster[label] = np.nanmean(all_units, axis=0)
        # plt.legend(handles=[self.bluepatch, self.greenpatch])
        # plt.tight_layout(); self.poppdf.savefig(); plt.close()

        cluster_to_cell_type = dict()
        for cluster_num, orig_cluster in mean_cluster.items():
            cluster = flatten_series(self.data['pref_gazeshift_psth'][self.data['mov_kmclust']==cluster_num])
            cluster_mean = np.nanmean(cluster, 0)
            baseline = np.nanmean(cluster_mean[:30])
            p, t = self.get_peak_trough(cluster_mean[38:50], baseline)
            cluster_to_cell_type[cluster_num] = self.get_cluster_props(p, t)
        for ind, row in self.data.iterrows():
            self.data.at[ind, 'sacccluster_no_movement'] = cluster_to_cell_type[row['mov_kmclust']]

        self.calc_direction_pref_index()
        self.data['sacccluster'] = self.data['sacccluster_no_movement']
        for ind, row in self.data.iterrows():
            if np.abs(row['gaze_sacc_rlMI']) >= 0.33:
                self.data.at[ind, 'sacccluster'] = 'movement'

        self.split_cluster_by_compensatory_modulation('biphasic', 'biphasic_comp_responsive')
        self.split_cluster_by_compensatory_modulation('early', 'early_comp_responsive')
        self.split_cluster_by_compensatory_modulation('late', 'late_comp_responsive')
        self.split_cluster_by_compensatory_modulation('negative', 'negative_comp_responsive')

        for ind, row in self.data[self.data['has_hf']].iterrows():
            ori_tuning = np.mean(row['Gt_ori_tuning_tf'],2) # [orientation, sf, tf]
            drift_spont = row['Gt_drift_spont']
            tuning = ori_tuning - drift_spont # subtract off spont rate
            tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
            th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
            prefered_direction = np.zeros(3)
            prefered_orientation = np.zeros(3)
            best_tuning_for_sf = np.zeros(3)
            for sf in range(3):
                R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
                th_ortho = (th_pref[sf]+2)%8 # get ortho position
                R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5  # ortho firing rate (average between two peaks)
                th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
                R_null = tuning[th_null, sf] # tuning value at that peak
                prefered_direction[sf] = (np.arange(8)*45)[th_null]
                prefered_orientation[sf] = (np.arange(8)*45)[th_pref[sf]]
                best_tuning_for_sf[sf] = R_pref
            best_sf_ind = np.argmax(best_tuning_for_sf)
            self.data.at[ind, 'Gt_best_direction'] = prefered_direction[best_sf_ind]
            self.data.at[ind, 'Gt_best_orientation'] = prefered_orientation[best_sf_ind]
        
    def position_around_saccade(self, movement):
        sessions = [i for i in self.data['session'].unique() if type(i) != float]
        n_sessions = len(self.data['session'].unique())
        plt.subplots(n_sessions,4,figsize=(20,30))
        count = 1
        for session_num in tqdm(range(len(sessions))):
            session = sessions[session_num]
            # get 0th index of units in this session (all units have identical info for these columns)
            row = self.data[self.data['session']==session].iloc[0]
                
            eyeT = np.array(row['FmLt_eyeT'])
            dEye = row['FmLt_dEye_dps']
            dhead = row['FmLt_dHead']
            dgz = dEye + dhead
            
            if movement=='eye_gaze_shifting':
                sthresh = 5
                rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
                leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
            elif movement=='eye_comp':
                sthresh = 3
                rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
                leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
            elif movement=='head_gaze_shifting':
                sthresh = 3
                rightsacc = eyeT[(np.append(dhead,0)>sthresh) & (np.append(dgz,0)>sthresh)]
                leftsacc = eyeT[(np.append(dhead,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
            elif movement=='head_comp':
                sthresh = 3
                rightsacc = eyeT[(np.append(dhead,0)>sthresh) & (np.append(dgz,0)<1)]
                leftsacc = eyeT[(np.append(dhead,0)<-sthresh) & (np.append(dgz,0)>-1)]

            deye_mov_right = np.zeros([len(rightsacc), len(self.trange)]); deye_mov_left = np.zeros([len(leftsacc), len(self.trange)])
            dgz_mov_right = np.zeros([len(rightsacc), len(self.trange)]); dgz_mov_left = np.zeros([len(leftsacc), len(self.trange)])
            dhead_mov_right = np.zeros([len(rightsacc), len(self.trange)]); dhead_mov_left = np.zeros([len(leftsacc), len(self.trange)])

            for sind in range(len(rightsacc)):
                s = rightsacc[sind]
                mov_ind = np.where([eyeT==self.find_nearest(eyeT, s)])[1]
                trange_inds = list(mov_ind + np.arange(-42,42))
                if np.max(trange_inds) < len(dEye):
                    deye_mov_right[sind,:] = dEye[np.array(trange_inds)]
                if np.max(trange_inds) < len(dgz):
                    dgz_mov_right[sind,:] = dgz[np.array(trange_inds)]
                if np.max(trange_inds) < len(dhead):
                    dhead_mov_right[sind,:] = dhead[np.array(trange_inds)]
            for sind in range(len(leftsacc)):
                s = leftsacc[sind]
                mov_ind = np.where([eyeT==self.find_nearest(eyeT, s)])[1]
                trange_inds = list(mov_ind + np.arange(-42,42))
                if np.max(trange_inds) < len(dEye):
                    deye_mov_left[sind,:] = dEye[np.array(trange_inds)]
                if np.max(trange_inds) < len(dgz):
                    dgz_mov_left[sind,:] = dgz[np.array(trange_inds)]
                if np.max(trange_inds) < len(dhead):
                    dhead_mov_left[sind,:] = dhead[np.array(trange_inds)]

    def set_activity_thresh(self, method='min_active', light_val=14, dark_val=7):
        """ Set threshold for how active an animal is before the session can be included in population analysis.
        Could add method for deye_count (number of eye movements) at some point?
        """
        active_time_by_session, light_len, dark_len = self.get_animal_activity()

        light_cols = [col for col in self.data.columns.values if 'FmLt' in col]
        light_frac_active = active_time_by_session['light']
        light_total_min = dict(zip(light_frac_active.keys(), [(i*self.model_dt)/60 for i in light_len]))
        light_min_active = dict({(session,frac*light_total_min[session]) for session,frac in light_frac_active.items()})

        dark_cols = [col for col in self.data.columns.values if 'FmDk' in col]
        dark_frac_active = active_time_by_session['dark']
        dark_total_min = dict(zip(dark_frac_active.keys(), [(i*self.model_dt)/60 for i in dark_len]))
        dark_min_active = dict({(session,frac*dark_total_min[session]) for session,frac in dark_frac_active.items()})

        # get sessions that do not meet criteria
        if method=='frac_active':
            bad_light = [s for s,v in light_frac_active.items() if v<=light_val]
            bad_dark = [s for s,v in dark_frac_active.items() if v<=dark_val]
        elif method=='min_active':
            bad_light = [s for s,v in light_min_active.items() if v<=light_val]
            bad_dark = [s for s,v in dark_min_active.items() if v<=dark_val]

        # set columns for stim and session not meeting criteria to NaN
        for s in bad_light:
            self.data[light_cols][self.data['session']==s] = np.nan
        for s in bad_dark:
            self.data[dark_cols][self.data['session']==s] = np.nan

    def set_experiment(self, exptype):
        if exptype=='hffm':
            self.data = self.data[self.data['use_in_dark_analysis']==False]
            self.data = self.data.drop(columns=[col for col in self.data.columns.values if 'FmDk' in col])
        elif exptype=='ltdk':
            self.data = self.data[self.data['use_in_dark_analysis']==True]
        self.exptype = exptype

    def find_SbCs_and_trGratPsth(self):
        for ind, row in self.data.iterrows():
            if row['Wn_contrast_tuning'][0]<1:
                self.data.at[ind, 'is_SbC'] = False
                self.data.at[ind, 'is_grat_trpsth'] = False
                continue
            high_contrast_std = np.std(row['Wn_contrast_tuning'][3:])
            self.data.at[ind, 'high_contrast_std'] = high_contrast_std
            min_contrast = row['Wn_contrast_tuning'][0]
            mid_contrast = row['Wn_contrast_tuning'][3]
            max_contrast = row['Wn_contrast_tuning'][-1]
            min_mid_mod = (min_contrast - mid_contrast) / (min_contrast + mid_contrast)
            min_max_mod = (min_contrast - max_contrast) / (min_contrast + max_contrast)
            self.data.at[ind, 'SbC_min_mid_mod'] = min_mid_mod
            self.data.at[ind, 'SbC_min_max_mod'] = min_max_mod
            # gratings psth
            psth = row['Gt_grating_psth']
            during_stim = psth[(self.grat_psth_x<1) * (self.grat_psth_x>0)]
            stim_onset = psth[(self.grat_psth_x>0) * (self.grat_psth_x<0.10)]
            prestim = psth[(self.grat_psth_x<0)]
            sbgrat = (np.nanmean(during_stim) - np.nanmean(prestim)) / np.nanmean(prestim)
            self.data.at[ind, 'grat_psth_drop'] = sbgrat
            self.data.at[ind, 'change_at_gratstim_onset'] = stim_onset - np.mean(prestim)
            self.data.at[ind, 'change_during_gratstim'] = np.mean(during_stim) - stim_onset
            if (min_mid_mod>0.25 and sbgrat<0):
                isSbC = True
            else:
                isSbC = False
            if (np.mean(stim_onset) > 2*np.mean(prestim) and np.mean(during_stim) < 0.66*stim_onset):
                isgrat_trpsth = True
            else:
                isgrat_trpsth = False
            self.data.at[ind, 'is_SbC'] = isSbC
            self.data.at[ind, 'is_grat_trpsth'] = isgrat_trpsth

    def calc_firing_rates_hffm(self):
        model_dt = 0.025
        for ind, row in self.data.iterrows():
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
            self.data.at[ind, 'FmLt_model_active'] = model_active.astype(object)
            self.data.at[ind, 'FmLt_modelT'] = modelT.astype(object)
            
            modelT = np.arange(0, np.nanmax(row['Wn_eyeT']), model_dt)
            diff = len(modelT) - len(row['Wn_rate'])
            if diff>0: # modelT is longer
                modelT = modelT[:-diff]
            elif diff<0: # modelT is shorted
                for i in range(np.abs(diff)):
                    modelT = np.append(modelT, modelT[-1]+model_dt)
            ballT = np.linspace(row['Wn_eyeT'][0], row['Wn_eyeT'][-1], row['Wn_ballspeed'].values.shape[0])
            model_speed = interp1d(ballT, row['Wn_ballspeed'].values, bounds_error=False)(modelT)
            self.data.at[ind, 'Wn_modelT'] = modelT.astype(object)
            self.data.at[ind, 'Wn_ball_speed'] = model_speed.astype(object)

        for ind, row in self.data.iterrows():
            self.data.at[ind,'FmLt_active_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']>40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']>40])
            self.data.at[ind,'FmLt_inactive_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']<40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']<40])
            self.data.at[ind,'Wn_active_fr'] = (np.sum(row['Wn_rate'][row['Wn_ball_speed']>=1.0])) / np.size(row['Wn_modelT'][row['Wn_ball_speed']>=1.0])
            self.data.at[ind,'Wn_inactive_fr'] = (np.sum(row['Wn_rate'][row['Wn_ball_speed']<1.0])) / np.size(row['Wn_modelT'][row['Wn_ball_speed']<1.0])

    def calc_firing_rates_ltdk(self):
        model_dt = 0.025
        for ind, row in self.data.iterrows():
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
            self.data.at[ind, 'FmLt_model_active'] = model_active.astype(object)
            self.data.at[ind, 'FmLt_modelT'] = modelT.astype(object)
            
            modelT = np.arange(0, np.nanmax(row['FmDk_eyeT']), model_dt)
            diff = len(modelT) - len(row['FmDk_rate'])
            if diff>0: # modelT is longer
                modelT = modelT[:-diff]
            elif diff<0: # modelT is shorted
                for i in range(np.abs(diff)):
                    modelT = np.append(modelT, modelT[-1]+model_dt)
            model_gz = interp1d(row['FmDk_imuT'], row['FmDk_gyro_z'], bounds_error=False)(modelT)
            model_active = np.convolve(np.abs(model_gz), np.ones(int(1/model_dt)), 'same') / len(np.ones(int(1/model_dt)))
            self.data.at[ind, 'FmDk_model_active'] = model_active.astype(object)
            self.data.at[ind, 'FmDk_modelT'] = modelT.astype(object)
        
        self.data['FmLt_fr'] = ((self.data['FmLt_rate'].apply(np.sum)*0.025) / self.data['FmLt_eyeT'].apply(np.nanmax)).to_numpy()
        self.data['FmDk_fr'] = ((self.data['FmDk_rate'].apply(np.sum)*0.025) / self.data['FmDk_eyeT'].apply(np.nanmax)).to_numpy()

        FmLt_fr = np.zeros([len(self.data.index.values)])
        FmDk_fr = np.zeros([len(self.data.index.values)])
        for ind, row in self.data.iterrows():
            self.data.at[ind,'FmLt_active_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']>40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']>40])
            self.data.at[ind,'FmLt_inactive_fr'] = (np.sum(row['FmLt_rate'][row['FmLt_model_active']<40])) / np.size(row['FmLt_modelT'][row['FmLt_model_active']<40])
            self.data.at[ind,'FmDk_active_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']>40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']>40])
            self.data.at[ind,'FmDk_inactive_fr'] = (np.sum(row['FmDk_rate'][row['FmDk_model_active']<40])) / np.size(row['FmDk_modelT'][row['FmDk_model_active']<40])

    def norm_PSTH(self, x, pref=None, bckgnd=None):
        if pref is None:
            pref = x.copy()
        if bckgnd is not None:
            return (x-x[bckgnd]) / np.nanmax(np.abs(pref))
        else:
            return (x-np.mean(bckgnd)) / np.nanmax(np.abs(pref))

    def PSTH_for_SnRc(self, bval=39):
        bval = 39
        for ind, row in self.data.iterrows():
            self.data.at[ind, 'norm_Rc_psth'] = self.norm_PSTH(row['Rc_psth'], bckgnd=bval).astype(object)
            if not np.isnan(row['Sn_on_background_psth']).all():
                Sn_selective_on = row['Sn_on_lightstim_psth'] - row['Sn_on_background_psth']
                self.data.at[ind, 'norm_Sn_selective_on'] = self.norm_PSTH(Sn_selective_on, bckgnd=bval).astype(object)
                self.data.at[ind, 'norm_Sn_background_on'] = self.norm_PSTH(row['Sn_on_background_psth'], bckgnd=bval).astype(object)
            if not np.isnan(row['Sn_off_background_psth']).all():
                Sn_selective_off = row['Sn_off_darkstim_psth'] - row['Sn_off_background_psth']
                self.data.at[ind, 'norm_Sn_selective_off'] = self.norm_PSTH(Sn_selective_off, bckgnd=bval).astype(object)
                self.data.at[ind, 'norm_Sn_background_off'] = self.norm_PSTH(row['Sn_off_background_psth'], bckgnd=bval).astype(object)
            if not np.isnan(row['Sn_on_all_psth']).all():
                self.data.at[ind, 'norm_Sn_on_all_psth'] = self.norm_PSTH(row['Sn_on_all_psth'], bckgnd=bval).astype(object)
        
        # selective response?
        thresh = 1.5
        self.data['has_on_Sn_selective_resp'] = False
        self.data['has_off_Sn_selective_resp'] = False
        ons = np.zeros(len(self.data.index.values))
        offs = np.zeros(len(self.data.index.values))
        i = 0
        for ind, row in self.data.iterrows():
            on = np.abs(row['norm_Sn_selective_on'])
            off = np.abs(row['norm_Sn_selective_off'])
            on_zscore = (np.max(on[35:45]) - np.mean(on)) / (np.std(on))
            off_zscore = (np.max(on[35:45]) - np.mean(off)) / (np.std(on))
            ons[i] = on_zscore
            offs[i] = off_zscore
            if on_zscore >= thresh:
                self.data.at[ind, 'has_on_Sn_selective_resp'] = True
            if off_zscore >= thresh:
                self.data.at[ind, 'has_off_Sn_selective_resp'] = True
            i += 1
    
    def calc_psth(self, spikeT, eventT):
        psth = np.zeros(self.trange.size-1)
        for s in np.array(eventT):
            hist, _ = np.histogram(spikeT-s, self.trange)
            psth = psth + hist / (eventT.size*np.diff(self.trange))
        return psth

    def apply_win_to_comp_sacc(self, comp, gazeshift, win=0.25):
        bad_comp = np.array([c for c in comp for g in gazeshift if ((g>(c-win)) & (g<(c+win)))])
        comp_times = np.delete(comp, np.isin(comp, bad_comp))
        return comp_times

    def recalc_saccades(self, stim='FmLt'):
        """
        need to recalculate for a few changes:
         - slightly different thresholds, not changed much
         - recalculate compensatory movements
            * previously, comp. saccades were contaminated with neural signals
                from gaze-shifting saccades that came right before the comp movement
            * now, if a gaze shift happened 250 msec BEFORE or AFTER any compensatory
                movement, that movement is removed
        """
        for ind, row in tqdm(self.data.iterrows()):
            eyeT = row[stim+'_eyeT']
            dEye = row[stim+'_dEye_dps']
            dHead = row[stim+'_dHead']
            dGaze = row[stim+'_dGaze']
            spikeT = row[stim+'_spikeT']

            # all eye movements
            left = eyeT[(np.append(dEye, 0) > self.low_sacc_thresh)]
            right = eyeT[(np.append(dEye, 0) < -self.low_sacc_thresh)]
            # save saccade times
            self.data.at[ind, stim+'_leftsacc_times'] = left.astype(object); self.data.at[ind, stim+'_rightsacc_times'] = right.astype(object)
            # save neural activity around saccades
            self.data.at[ind, stim+'_leftsacc_avg'] =  self.calc_psth(spikeT, left).astype(object)
            self.data.at[ind, stim+'_rightsacc_avg'] =  self.calc_psth(spikeT, right).astype(object)

            # all head movements
            left = eyeT[(np.append(dHead, 0) > self.low_sacc_thresh)]
            right = eyeT[(np.append(dHead, 0) < -self.low_sacc_thresh)]
            # save saccade times
            self.data.at[ind, stim+'_leftsacc_dHead_times'] = left.astype(object); self.data.at[ind, stim+'_rightsacc_dHead_times'] = right.astype(object)
            # save neural activity around saccades
            self.data.at[ind, stim+'_leftsacc_dHead_avg'] =  self.calc_psth(spikeT, left).astype(object)
            self.data.at[ind, stim+'_rightsacc_dHead_avg'] =  self.calc_psth(spikeT, right).astype(object)

            # gaze-shift dEye
            left_gaze_dEye = eyeT[(np.append(dEye, 0) > self.high_sacc_thresh) & (np.append(dGaze,0) > self.high_sacc_thresh)]
            right_gaze_dEye = eyeT[(np.append(dEye, 0) < -self.high_sacc_thresh) & (np.append(dGaze, 0) < -self.high_sacc_thresh)]
            self.data.at[ind, stim+'_leftsacc_avg_gaze_shift_dEye_times'] = left_gaze_dEye.astype(object); self.data.at[ind, stim+'_rightsacc_avg_gaze_shift_dEye_times'] = right_gaze_dEye.astype(object)
            self.data.at[ind, stim+'_leftsacc_avg_gaze_shift_dEye'] =  self.calc_psth(spikeT, left_gaze_dEye).astype(object)
            self.data.at[ind, stim+'_rightsacc_avg_gaze_shift_dEye'] =  self.calc_psth(spikeT, right_gaze_dEye).astype(object)
            
            # comp dEye
            left_comp_dEye = eyeT[(np.append(dEye, 0) > self.low_sacc_thresh) & (np.append(dGaze, 0) < self.gaze_sacc_thresh)]
            right_comp_dEye = eyeT[(np.append(dEye, 0) < -self.low_sacc_thresh) & (np.append(dGaze, 0) > -self.gaze_sacc_thresh)]
            left_comp_dEye = self.apply_win_to_comp_sacc(left_comp_dEye, left_gaze_dEye)
            right_comp_dEye = self.apply_win_to_comp_sacc(right_comp_dEye, right_gaze_dEye)
            self.data.at[ind, stim+'_leftsacc_avg_comp_dEye_times'] = left_comp_dEye.astype(object); self.data.at[ind, stim+'_rightsacc_avg_comp_dEye_times'] = right_comp_dEye.astype(object)
            self.data.at[ind, stim+'_leftsacc_avg_comp_dEye'] =  self.calc_psth(spikeT, left_comp_dEye).astype(object)
            self.data.at[ind, stim+'_rightsacc_avg_comp_dEye'] =  self.calc_psth(spikeT, right_comp_dEye).astype(object)
            
            # gaze-shift dHead
            left_gaze_dHead = eyeT[(np.append(dHead, 0) > self.low_sacc_thresh) & (np.append(dGaze, 0) > self.low_sacc_thresh)]
            right_gaze_dHead = eyeT[(np.append(dHead, 0) < -self.low_sacc_thresh) & (np.append(dGaze, 0) < -self.low_sacc_thresh)]
            self.data.at[ind, stim+'_leftsacc_avg_gaze_shift_dHead_times'] = left_gaze_dHead.astype(object); self.data.at[ind, stim+'_rightsacc_avg_gaze_shift_dHead_times'] = right_gaze_dHead.astype(object)
            self.data.at[ind, stim+'_leftsacc_avg_gaze_shift_dHead'] =  self.calc_psth(spikeT, left_gaze_dHead).astype(object)
            self.data.at[ind, stim+'_rightsacc_avg_gaze_shift_dHead'] =  self.calc_psth(spikeT, right_gaze_dHead).astype(object)
            
            # comp dHead
            left_comp_dHead = eyeT[(np.append(dHead,0) > self.low_sacc_thresh) & (np.append(dGaze, 0) < self.gaze_sacc_thresh)]
            right_comp_dHead = eyeT[(np.append(dHead,0) < -self.low_sacc_thresh) & (np.append(dGaze,0) > -self.gaze_sacc_thresh)]
            left_comp_dHead = self.apply_win_to_comp_sacc(left_comp_dHead, left_gaze_dHead)
            right_comp_dHead = self.apply_win_to_comp_sacc(right_comp_dHead, right_gaze_dHead)
            self.data.at[ind, stim+'_leftsacc_avg_comp_dHead_times'] = left_comp_dHead.astype(object); self.data.at[ind, stim+'_rightsacc_avg_comp_dHead_times'] = right_comp_dHead.astype(object)
            self.data.at[ind, stim+'_leftsacc_avg_comp_dHead'] =  self.calc_psth(spikeT, left_comp_dHead).astype(object)
            self.data.at[ind, stim+'_rightsacc_avg_comp_dHead'] =  self.calc_psth(spikeT, right_comp_dHead).astype(object)

    def summarize_population(self, extras=False):
        # print('applying activity thresholds')
        # self.set_activity_thresh()

        # self.poppdf = PdfPages(os.path.join(self.savepath, 'population_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))

        # print('recalculating FmLt saccades')
        # self.recalc_saccades(stim='FmLt')
        # if self.exptype == 'ltdk':
        #     print('recalculating FmDk saccades')
        #     self.recalc_saccades(stim='FmDk')

        print('clustering by waveform')
        self.putative_celltype()

        if np.sum(self.data['has_hf'])>1:
            print('contrast response')
            self.neural_response_to_contrast()

        if np.sum(self.data['has_hf'])>1:
            print('gratings response')
            self.neural_response_to_gratings()

        if np.sum(self.data['has_hf'])>1:
            print('median firing rate by stim and animal activity')
            self.spike_rate_by_stim()

        print('movement tuning')
        self.neural_response_to_movement()

        print('dEye clustering')
        self.deye_clustering()

        if self.exptype == 'hffm':
            print('SbCs')
            self.find_SbCs_and_trGratPsth()
            print('firing rates')
            self.calc_firing_rates_hffm()

        if self.exptype == 'ltdk':
            print('firing rates')
            self.calc_firing_rates_ltdk()

        if self.exptype == 'hffm':
            print('sparese noise and revchecker')
            self.PSTH_for_SnRc()

        if extras:
            print('dhead and deye around time of gaze shifting eye movements')
            self.position_around_saccade('eye_gaze_shifting')
            print('dhead and deye around time of compesatory eye movements')
            self.position_around_saccade('eye_comp')
            print('dhead and deye around time of gaze shifting head movements')
            self.position_around_saccade('head_gaze_shifting')
            print('dhead and deye around time of compensatory head movements')
            self.position_around_saccade('head_comp')

        # self.poppdf.close()

    def process(self):
        self.gather_data()
        self.data = self.data.reset_index()
        self.save_as_pickle(stage='gathered')
        
        self.add_available_optic_flow_data()

        self.summarize_sessions()

        self.summarize_units()
        self.save_as_pickle(stage='unit')

        self.summarize_population()
        self.save_as_pickle(stage='population')