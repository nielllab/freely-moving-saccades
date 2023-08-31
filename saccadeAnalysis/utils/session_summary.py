

import os
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from matplotlib.backends.backend_pdf import PdfPages

import fmEphys as fme
import saccadeAnalysis as sacc


def get_animal_activity(data):
        
        model_dt = 0.025

        active_time_by_session = dict()
        dark_len = []; light_len = []
        sessions = [x for x in data['session'].unique() if str(x) != 'nan']
        for session in sessions:
            session_data = data[data['session']==session]

            # find active times
            if 'FmLt_eyeT' in session_data.columns.values and type(session_data['FmLt_eyeT'].iloc[0]) != float:
                # light setup
                fm_light_eyeT = np.array(session_data['FmLt_eyeT'].iloc[0])
                fm_light_gz = session_data['FmLt_gyro_z'].iloc[0]
                fm_light_accT = session_data['FmLt_imuT'].iloc[0]
                light_model_t = np.arange(0,np.nanmax(fm_light_eyeT), model_dt)
                light_model_gz = interp1d(fm_light_accT,(fm_light_gz-np.mean(fm_light_gz))*7.5,bounds_error=False)(light_model_t)
                light_model_active = np.convolve(np.abs(light_model_gz),np.ones(np.int(1/model_dt)),'same')
                light_active = light_model_active>40

                n_units = len(session_data)
                light_model_nsp = np.zeros((n_units, len(light_model_t)))
                bins = np.append(light_model_t, light_model_t[-1] + model_dt)
                
                i = 0
                for ind, row in session_data.iterrows():
                    light_model_nsp[i,:], bins = np.histogram(row['FmLt_spikeT'], bins)
                    unit_active_spikes = light_model_nsp[i, light_active]
                    unit_stationary_spikes = light_model_nsp[i, ~light_active]
                    data.at[ind,'FmLt_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*model_dt)
                    data.at[ind,'FmLt_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*model_dt)
                    i += 1

                active_time_by_session.setdefault('light', {})[session] = np.sum(light_active) / len(light_active)
                light_len.append(len(light_active))

            if 'FmDk_eyeT' in session_data.columns.values and type(session_data['FmDk_eyeT'].iloc[0]) != float:
                del unit_active_spikes, unit_stationary_spikes

                # dark setup
                FmDk_eyeT = np.array(session_data['FmDk_eyeT'].iloc[0])
                FmDk_gz = session_data['FmDk_gyro_z'].iloc[0]
                FmDk_accT = session_data['FmDk_imuT'].iloc[0]
                dark_model_t = np.arange(0,np.nanmax(FmDk_eyeT), model_dt)
                dark_model_gz = interp1d(FmDk_accT,(FmDk_gz-np.mean(FmDk_gz))*7.5,bounds_error=False)(dark_model_t)
                dark_model_active = np.convolve(np.abs(dark_model_gz),np.ones(np.int(1/model_dt)),'same')
                dark_active = dark_model_active > 40

                n_units = len(session_data)
                dark_model_nsp = np.zeros((n_units, len(dark_model_t)))
                bins = np.append(dark_model_t, dark_model_t[-1] + model_dt)
                i = 0
                for ind, row in session_data.iterrows():
                    dark_model_nsp[i,:], bins = np.histogram(row['FmDk_spikeT'], bins)
                    unit_active_spikes = dark_model_nsp[i, dark_active]
                    unit_stationary_spikes = dark_model_nsp[i, ~dark_active]
                    data.at[ind,'FmDk_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*model_dt)
                    data.at[ind,'FmDk_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*model_dt)
                    i += 1

                active_time_by_session.setdefault('dark', {})[session] = np.sum(dark_active) / len(dark_active)
                dark_len.append(len(dark_active))

        return active_time_by_session, light_len, dark_len

def summarize_sessions(data, savedir):

    model_dt = 0.025
    
    pdf = PdfPages(
        os.path.join(
            savedir,
            'session_summary_{}.pdf'.format(fme.fmt_now(c=True))
        )
    )

    if 'FmDk_theta' in data.columns:
        data['has_dark'] = ~data['FmDk_theta'].isna()
    else:
        data['has_dark'] = False
    
    if 'Wn_contrast_tuning' in data.columns:
        data['has_hf'] = ~data['Wn_contrast_tuning'].isna()
    else:
        data['has_hf'] = False
        
    if data['has_dark'].sum() > 0 and data['has_hf'].sum() > 0:
        # For a group of sessions where there is at least one dark session
        # and one head-fixed session.

        active_time_by_session, light_len, dark_len = get_animal_activity()

        # fraction active time: light vs dark
        light = np.array([val for key,val in active_time_by_session['light'].items()])
        light_err = np.std(light) / np.sqrt(len(light))
        dark = np.array([val for key,val in active_time_by_session['dark'].items()])
        dark_err = np.std(dark) / np.sqrt(len(dark))

        fig, ax = plt.subplots(1,1,figsize=(3,5))
        ax.bar(0, np.mean(light), yerr=light_err, width=0.5, color='yellow')
        ax.plot(np.zeros(len(light)), light, 'o', color='tab:gray')
        ax.bar(1, np.mean(dark), yerr=dark_err, width=0.5, color='cadetblue')
        ax.plot(np.ones(len(dark)), dark, 'o', color='tab:gray')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['light','dark'])
        ax.set_ylim([0,1])
        ax.set_ylabel('fraction of time spent active')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

        # fraction active time: light vs dark (broken up by session)
        dark_active_times = [active_frac for session, active_frac in active_time_by_session['dark'].items()]
        dark_session_names = [session for session, active_frac in active_time_by_session['dark'].items()]
        
        fig, ax = plt.subplots(1,1, figsize=(5,10))
        ax.bar(np.arange(0, len(dark_session_names)), dark_active_times, color='cadetblue')
        ax.set_xticks(np.arange(0, len(dark_session_names)))
        ax.set_xticklabels(dark_session_names, rotation=90)
        ax.set_ylabel('frac active time')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

        light_active_times = [active_frac for session, active_frac in active_time_by_session['light'].items()]
        light_session_names = [session for session, active_frac in active_time_by_session['light'].items()]
        
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        ax.bar(np.arange(0, len(light_session_names)), light_active_times, color='khaki')
        ax.set_xticks(np.arange(len(light_session_names)))
        ax.set_xticklabels(light_session_names, rotation=90)
        ax.set_ylabel('frac active time')
        ax.set_ylim([0,1])
        fig.tight_layout()
        pdf.savefig()
        fig.close()

        # minutes active or stationary: light vs dark
        total_min = [(i*model_dt)/60 for i in light_len]
        frac_active = [active_frac for session, active_frac in active_time_by_session['light'].items()]
        light_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
        light_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]
        light_session_names = [session for session, active_frac in active_time_by_session['light'].items()]
        
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        ax.bar(np.arange(0, len(light_session_names)), light_active_min,
               color='salmon', label='active')
        ax.bar(np.arange(0, len(light_session_names)), light_stationary_min,
               bottom=light_active_min,
               color='gray', label='stationary')
        ax.set_xticks(np.arange(len(light_session_names)))
        ax.set_xticklabels(light_session_names, rotation=90)
        ax.legend()
        ax.set_ylabel('recording time (min)')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

        total_min = [(i*model_dt)/60 for i in dark_len]
        frac_active = [active_frac for session, active_frac in active_time_by_session['dark'].items()]
        dark_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
        dark_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]
        dark_session_names = [session for session, active_frac in active_time_by_session['dark'].items()]
        
        fig, ax = plt.subplots(1,1, figsize=(12,10))
        ax.bar(np.arange(0, len(dark_session_names)), dark_active_min,
               color='salmon', label='active')
        ax.bar(np.arange(0, len(dark_session_names)), dark_stationary_min,
                bottom=dark_active_min,
                color='gray', label='stationary')
        ax.set_xticks(np.arange(len(dark_session_names)))
        ax.set_xticklabels(dark_session_names, rotation=90)
        fig.legend()
        ax.est_ylabel('recording time (min)')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

    movement_count_dict = dict()
    session_stim_list = []

    if data['has_dark'].sum() > 0:
        session_stim_list.append('FmDk')

    if data['has_hf'].sum() > 0:
        session_stim_list.append('FmLt')

    for base in session_stim_list:
        
        for movement in ['eye_gaze_shifting', 'eye_comp']:
            
            sessions = [i for i in data['session'].unique() if type(i) != float]
            n_sessions = len(data['session'].unique())
            trange = np.arange(-1,1.1,0.025)
            
            for session_num, session_name in enumerate(sessions):
                
                row = data[data['session']==session_name].iloc[0]

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

                deye_mov_right = np.zeros([len(rightsacc), len(trange)])
                deye_mov_left = np.zeros([len(leftsacc), len(trange)])

                dgz_mov_right = np.zeros([len(rightsacc), len(trange)])
                dgz_mov_left = np.zeros([len(leftsacc), len(trange)])
                
                dhead_mov_right = np.zeros([len(rightsacc), len(trange)])
                dhead_mov_left = np.zeros([len(leftsacc), len(trange)])

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

    if np.sum(data['has_dark']) > 0:

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

        ax.bar(x - width/2, np.mean(right_gaze), width,
               color='lightcoral')
        ax.bar(x - width/2, np.mean(left_gaze), width,
               bottom=np.mean(right_gaze), color='lightsteelblue')
        ax.plot(np.ones(len(right_gaze))*(0 - width/2),
                np.add(right_gaze, left_gaze), '.', color='gray')

        ax.bar(x + width/2, np.mean(right_gaze_dark), width,
               color='lightcoral')
        ax.bar(x + width/2, np.mean(left_gaze_dark), width,
               bottom=np.mean(right_gaze_dark), color='lightsteelblue')
        ax.plot(np.ones(len(right_gaze_dark))*(0 + width/2),
                np.add(right_gaze_dark, left_gaze_dark), '.', color='gray')

        ax.bar(x - width/2, np.mean(right_comp), width,
               color='lightcoral')
        ax.bar(x - width/2, np.mean(left_comp), width,
               bottom=np.mean(right_comp), color='lightsteelblue')
        ax.plot(np.ones(len(right_comp))*(1 - width/2),
                np.add(right_comp, left_comp), '.', color='gray')

        ax.bar(x + width/2, np.mean(right_comp_dark), width,
               color='lightcoral')
        ax.bar(x + width/2, np.mean(left_comp_dark), width,
               bottom=np.mean(right_comp_dark), color='lightsteelblue')
        ax.plot(np.ones(len(right_comp_dark))*(1 + width/2),
                np.add(right_comp_dark, left_comp_dark), '.', color='gray')

        ax.set_xticks(x)
        ax.set_xticklabels(['gaze-shifting', 'compensatory'])
        ax.set_ylim([0,3700])
        ax.set_ylabel('number of eye movements')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

        total_min = [(i*model_dt)/60 for i in light_len]
        frac_active = [active_frac for session, active_frac in active_time_by_session['light'].items()]
        light_active_min = [total_min[i] * frac_active[i] for i in range(len(total_min))]
        light_stationary_min = [total_min[i] * (1-frac_active[i]) for i in range(len(total_min))]

        # number of eye movements per minute of active time: light vs dark (broken up by session)
        fig, [ax1,ax2] = plt.subplots(2,1,figsize=(10,15))
        ax1.bar(light_session_names, np.add(right_gaze, left_gaze) / light_active_min)
        ax1.set_xticklabels(light_session_names, rotation=90)
        ax1.set_ylim([0,220])
        ax1.set_ylabel('eye movements per min during active periods')
        ax1.set_title('light stim')
        
        ax2.bar(dark_session_names, np.add(right_gaze_dark, left_gaze_dark) / dark_active_min, width=0.3)
        ax2.set_xticklabels(dark_session_names, rotation=90)
        ax2.ylim([0,220])
        ax2.ylabel('eye movements per min during active periods')
        ax2.set_title('dark stim')
        fig.tight_layout()
        pdf.savefig()
        fig.close()

    session_data = data.set_index('session')
    unique_inds = sorted(list(set(session_data.index.values)))
    
    for unique_ind in tqdm(unique_inds):
        
        uniquedf = session_data.loc[unique_ind]

        fmt_m = str(np.round(uniquedf['best_ellipse_fit_m'].iloc[0],4))
        fmt_r = str(np.round(uniquedf['best_ellipse_fit_r'].iloc[0],4))

        fig, axs = plt.subplots(5,5,figsize=(40,30))
        axs = axs.flatten()

        axs[0].set_title(unique_ind+' eye fit: m='+fmt_m+' r='+fmt_r, fontsize=20)
        dEye = uniquedf['FmLt_dEye_dps'].iloc[0]
        dHead = uniquedf['FmLt_dHead'].iloc[0]
        eyeT = uniquedf['FmLt_eyeT'].iloc[0]
        axs[0].plot(dEye[::10], dHead[::10], 'k.')
        axs[0].set_xlabel('dEye (deg/sec)', fontsize=20)
        axs[0].set_ylabel('dHead (deg/sec)', fontsize=20)
        axs[0].set_xlim((-700,700))
        axs[0].set_ylim((-700,700))
        axs[0].plot([-700,700],[700,-700], 'r:')

        imuT = uniquedf['FmLt_imuT'].iloc[0]
        roll = uniquedf['FmLt_roll'].iloc[0]
        pitch = uniquedf['FmLt_pitch'].iloc[0]

        centered_roll = roll - np.mean(roll)
        roll_interp = interp1d(imuT, centered_roll, bounds_error=False)(eyeT)

        centered_pitch = pitch - np.mean(pitch)
        pitch_interp = interp1d(imuT, centered_pitch, bounds_error=False)(eyeT)

        th = uniquedf['FmLt_theta'].iloc[0]
        phi = uniquedf['FmLt_phi'].iloc[0]

        axs[1].plot(pitch_interp[::100], th[::100], '.')
        axs[1].set_xlabel('pitch (deg)', fontsize=20)
        axs[1].set_ylabel('theta (deg)', fontsize=20)
        axs[1].set_ylim([-60,60])
        axs[1].set_xlim([-60,60])
        axs[1].plot([-60,60],[-60,60], 'r:')
        
        axs[2].subplot(5,5,3)
        axs[2].plot(roll_interp[::100], phi[::100], '.')
        axs[2].set_xlabel('roll (deg)', fontsize=20)
        axs[2].set_ylabel('phi (deg)', fontsize=20)
        axs[2].set_ylim([-60,60])
        axs[2].set_xlim([-60,60])
        axs[2].plot([-60,60],[60,-60], 'r:')
        
        # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
        axs[3].hist(uniquedf['FmLt_theta'].iloc[0], range=[-45,45], alpha=0.5)
        axs[3].xlabel('FmLt theta (deg)', fontsize=20)
        
        # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
        axs[4].hist(uniquedf['FmLt_phi'].iloc[0], range=[-45,45], alpha=0.5)
        axs[4].xlabel('FmLt phi (deg)', fontsize=20)
        
        # histogram of gyro z (resonable range?)
        axs[5].hist(uniquedf['FmLt_gyro_z'].iloc[0], range=[-400,400], alpha=0.5)
        axs[5].xlabel('FmLt gyro z (deg)', fontsize=20)
        
        # plot of contrast response functions on same panel scaled to max 30sp/sec
        # plot of average contrast reponse function across units
        if uniquedf['has_hf'].iloc[0]:
            for ind, row in uniquedf.iterrows():
                axs[6].errorbar(row['Wn_contrast_tuning_bins'],
                                row['Wn_contrast_tuning'],
                                yerr=row['Wn_contrast_tuning_err'],
                                alpha=0.5, linewidth=4)
            axs[6].set_ylim(0,30)
            axs[6].set_xlabel('contrast a.u.', fontsize=20)
            axs[6].set_ylabel('sp/sec', fontsize=20)
            axs[6].set_title('hf contrast tuning', fontsize=20)
            axs[6].errorbar(uniquedf['Wn_contrast_tuning_bins'].iloc[0],
                         np.mean(uniquedf['Wn_contrast_tuning'], axis=0),
                         yerr=np.mean(uniquedf['Wn_contrast_tuning_err'],axis=0),
                         color='k', linewidth=6)
            
            # lfp traces as separate shanks
            colors = plt.cm.jet(np.linspace(0,1,32))
            num_channels = np.size(uniquedf['Rc_response_by_channel'].iloc[0],0)
            
            if num_channels == 64:
                
                for ch_num in np.arange(0,64):
                    
                    if ch_num<=31:
                        axs[7].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                    color=colors[ch_num], linewidth=1)
                        axs[7].set_title('shank0', fontsize=20)
                        axs[7].axvline(x=(0.1*30000))
                        axs[7].set_xticks(np.arange(0,18000,18000/5),
                                          np.arange(0,600,600/5))
                        axs[7].set_ylim([-1200,400])
                        axs[7].set_xlabel('msec', fontsize=20)
                        axs[7].set_ylabel('uvolts', fontsize=20)
                    
                    if ch_num>31:
                        axs[8].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                 color=colors[ch_num-32], linewidth=1)
                        axs[8].set_title('shank1', fontsize=20)
                        axs[8].axvline(x=(0.1*30000))
                        axs[8].set_xticks(np.arange(0,18000,18000/5),
                                          np.arange(0,600,600/5))
                        axs[8].set_ylim([-1200,400])
                        axs[8].set_xlabel('msec', fontsize=20)
                        axs[8].set_ylabel('uvolts', fontsize=20)

                axs[9].axis('off')
                axs[10].axis('off')
            
            elif num_channels == 128:

                for ch_num in np.arange(0,128):

                    if ch_num < 32:
                        axs[7].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                   color=colors[ch_num], linewidth=1)
                        axs[7].set_title('shank0')
                        axs[7].axvline(x=(0.1*30000))
                        axs[7].set_xlabel('msec', fontsize=20)
                        axs[7].set_ylabel('uvolts', fontsize=20)
                        axs[7].set_xticks(np.arange(0,18000,18000/5),
                                   np.arange(0,600,600/5))

                    elif 32 <= ch_num < 64:
                        axs[8].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                 color=colors[ch_num-32], linewidth=1)
                        axs[8].set_title('shank1')
                        axs[8].set_axvline(x=(0.1*30000))
                        axs[8].set_xlabel('msec', fontsize=20)
                        axs[8].set_ylabel('uvolts', fontsize=20)
                        axs[8].set_xticks(np.arange(0,18000,18000/5),
                                   np.arange(0,600,600/5))
                    
                    elif 64 <= ch_num < 10:
                        axs[9].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                    color=colors[ch_num-64], linewidth=1)
                        axs[9].set_title('shank2')
                        axs[9].axvline(x=(0.1*30000))
                        axs[9].set_xlabel('msec', fontsize=20)
                        axs[9].set_ylabel('uvolts', fontsize=20)
                        axs[9].set_xticks(np.arange(0,18000,18000/5),
                                   np.arange(0,600,600/5))
                    
                    elif 96 <= ch_num < 128:
                        axs[10].plot(uniquedf['Rc_response_by_channel'].iloc[0][ch_num],
                                     color=colors[ch_num-96], linewidth=1)
                        axs[10].set_title('shank3')
                        axs[10].axvline(x=(0.1*30000))
                        axs[10].set_xlabel('msec', fontsize=20)
                        axs[10].set_ylabel('uvolts', fontsize=20)
                        axs[10].set_xticks(np.arange(0,18000,18000/5),
                                   np.arange(0,600,600/5))
        
        # fm spike raster
        i = 0
        for ind, row in uniquedf.iterrows():
            axs[11].vlines(row['FmLt_spikeT'], i-0.25, i+0.25)
            axs[11].set_xlim(0, 10)
            axs[11].set_xlabel('sec', fontsize=20)
            axs[11].set_ylabel('unit #', fontsize=20)
            i = i+1

        if uniquedf['has_hf'].iloc[0]:
            try:
                lower = -0.5
                upper = 1.5
                dt = 0.1
                bins = np.arange(lower,upper+dt,dt)

                psth_list = []
                for ind, row in uniquedf.iterrows():
                    axs[12].plot(bins[0:-1]+dt/2,row['Gt_grating_psth'])
                    psth_list.append(row['Gt_grating_psth'])
                avg_psth = np.mean(np.array(psth_list), axis=0)
                axs[12].plot(bins[0:-1]+dt/2,avg_psth,color='k',linewidth=6)
                axs[12].set_title('gratings psth', fontsize=20)
                axs[12].set_xlabel('sec', fontsize=20)
                axs[12].set_ylabel('sp/sec', fontsize=20)
                axs[12].set_ylim([0,np.nanmax(avg_psth)*1.5])
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
   
                    axs[13].plot(norm_profile_sh0,range(0,32))
                    axs[13].plot(norm_profile_sh0[layer5_cent_sh0]+0.01,
                                 layer5_cent_sh0,'r*',markersize=12)
                    axs[13].set_ylim([33,-1])
                    axs[13].set_yticks(ticks=list(range(-1,33)),
                                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                    axs[13].set_title('shank0', fontsize=20)
                    axs[13].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[13].set_xlabel('norm mua power', fontsize=20)

                    axs[14].plot(norm_profile_sh1,range(0,32))
                    axs[14].plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                    axs[14].set_ylim([33,-1])
                    axs[14].set_yticks(ticks=list(range(-1,33)),
                                        labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                    axs[14].set_title('shank1', fontsize=20)
                    axs[14].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[14].set_xlabel('norm mua power', fontsize=20)
                    
                    axs[15].axis('off')
                    
                    axs[16].axis('off')
                
                if '16' in uniquedf['probe_name'].iloc[0]:

                    norm_profile_sh0 = lfp_power_profile[0]
                    layer5_cent_sh0 = layer5_cent[0]
                    
                    axs[13].plot(norm_profile_sh0,range(0,16))
                    axs[13].plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                    axs[13].set_ylim([17,-1])
                    axs[13].set_yticks(ticks=list(range(-1,17)),
                                   labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
                    axs[13].set_title('shank0', fontsize=20)
                    axs[13].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[13].set_xlabel('norm mua power', fontsize=20)
                    
                    axs[14].axis('off')
                    
                    axs[15].axis('off')
                    
                    axs[16].axis('off')

                if '128' in uniquedf['probe_name'].iloc[0]:

                    norm_profile_sh0 = lfp_power_profile[0]
                    layer5_cent_sh0 = layer5_cent[0]
                    norm_profile_sh1 = lfp_power_profile[1]
                    layer5_cent_sh1 = layer5_cent[1]
                    norm_profile_sh2 = lfp_power_profile[2]
                    layer5_cent_sh2 = layer5_cent[2]
                    norm_profile_sh3 = lfp_power_profile[3]
                    layer5_cent_sh3 = layer5_cent[3]

                    axs[13].plot(norm_profile_sh0,range(0,32))
                    axs[13].plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                    axs[13].set_ylim([33,-1])
                    axs[13].set_yticks(ticks=list(range(-1,33)),
                                   labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                    axs[13].set_title('shank0', fontsize=20)
                    axs[13].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[13].set_xlabel('norm mua power', fontsize=20)

                    axs[14].plot(norm_profile_sh1,range(0,32))
                    axs[14].plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                    axs[14].set_ylim([33,-1])
                    axs[14].set_yticks(ticks=list(range(-1,33)),
                               labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                    axs[14].set_title('shank1', fontsize=20)
                    axs[14].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[14].set_xlabel('norm mua power', fontsize=20)

                    axs[15].plot(norm_profile_sh2,range(0,32))
                    axs[15].plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
                    axs[15].set_ylim([33,-1])
                    axs[15].set_yticks(ticks=list(range(-1,33)),
                                labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
                    axs[15].set_title('shank2', fontsize=20)
                    axs[15].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[15].set_xlabel('norm mua power', fontsize=20)

                    axs[16].plot(norm_profile_sh3,range(0,32))
                    axs[16].plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
                    axs[16].set_ylim([33,-1])
                    axs[16].set_yticks(ticks=list(range(-1,33)),
                                labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
                    axs[16].set_title('shank3', fontsize=20)
                    axs[16].set_ylabel('depth relative to layer 5', fontsize=20)
                    axs[16].set_xlabel('norm mua power', fontsize=20)

        if not uniquedf['has_dark'].iloc[0]:
            axs[17].axis('off')
            axs[18].axis('off')
            axs[19].axis('off')

        elif uniquedf['has_dark'].iloc[0]:

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

            axs[17].plot(dEye_dk[::10], dHead_dk[::10], 'k.')
            axs[17].set_xlabel('dark dEye (deg)', fontsize=20)
            axs[17].set_ylabel('dark dHead (deg)', fontsize=20)
            axs[17].set_xlim((-700,700))
            axs[17].set_ylim((-700,700))
            axs[17].plot([-700,700],[700,-700], 'r:')

            axs[18].plot(pitch_dk_interp[::100], theta_dk[::100], '.')
            axs[18].set_xlabel('dark pitch (deg)', fontsize=20)
            axs[18].set_ylabel('dark theta (deg)', fontsize=20)
            axs[18].set_ylim([-60,60])
            axs[18].set_xlim([-60,60])
            axs[18].plot([-60,60],[-60,60], 'r:')
            
            axs[19].plot(roll_dk_interp[::100], phi_dk[::100], '.')
            axs[19].set_xlabel('dark roll (deg)', fontsize=20)
            axs[19].set_ylabel('dark phi (deg)', fontsize=20)
            axs[19].set_ylim([-60,60])
            axs[19].set_xlim([-60,60])
            axs[19].plot([-60,60],[60,-60], 'r:')

        fig.tight_layout(); pdf.savefig(); fig.close()

    pdf.close()
