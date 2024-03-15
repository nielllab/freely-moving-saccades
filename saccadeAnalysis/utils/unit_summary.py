

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys as fme
import saccadeAnalysis as sacc


def summarize_units(data_filepath, use_pop_outputs=False):

    data = fme.read_group_h5(data_filepath)

    savepath = os.path.split(data_filepath)[0]

    pdf = PdfPages(os.path.join(savepath, 'unit_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))

    ### Set up columns indicating which stimuli exist

    if 'FmDk_theta' in data.columns:
        sacc.is_empty_index(data, 'FmDk_theta', 'has_dark')
    else:
        data['has_dark'] = False
    
    if 'Wn_contrast_tuning' in data.columns:
        sacc.is_empty_index(data, 'Wn_contrast_tuning', 'has_hf')
    else:
        data['has_hf'] = False

    if 'has_optic_flow' not in data.columns:
        data['has_optic_flow'] = False

    print('num units=' + str(len(data)))


    for ind, row in tqdm(data.iterrows()):

        # set up page
        fig = plt.figure(constrained_layout=True, figsize=(30,22))
        spec = gridspec.GridSpec(ncols=7, nrows=10, figure=fig)

        # page title
        title = fig.add_subplot(spec[0,0])
        title.axis('off')
        title.annotate(str(row['session'])+'_unit'+str(ind),
                       xy=(0.05, 0.95),
                       xycoords='axes fraction',
                       fontsize=20)

        # unit waveform
        unitfig_waveform = fig.add_subplot(spec[0,1])
        sacc.waveform(
            ax=unitfig_waveform,
            row=row
        )

        # whitenoise contrast tuning curve
        fig_contrast_tuning = fig.add_subplot(spec[0,2])
        if row['has_hf']:
            Wn_contrast_modind = sacc.tuning_curve(
                ax=fig_contrast_tuning,
                row=row,
                varcent_name='Wn_contrast_tuning_bins',
                tuning_name='Wn_contrast_tuning',
                err_name='Wn_contrast_tuning_err',
                title='Wn contrast',
                xlabel='contrast a.u.'
            )
            data.at[ind, 'Wn_contrast_modind'] = Wn_contrast_modind
        else:
            fig_contrast_tuning.axis('off')

        # gratings psth
        fig_grat_psth = fig.add_subplot(spec[0,3])
        if row['has_hf']:
            sacc.grat_psth(ax=fig_grat_psth,
                           row=row)
        else:
            fig_grat_psth.axis('off')

        # laminar depth relative to cortex layer 4
        # based on revchecker stim
        #fig_revchecker_depth = fig.add_subplot(spec[0,4])
        #if row['has_hf']:
            #sacc.revchecker_laminar_depth(
                #ax=fig_revchecker_depth,
                #row=row,
                #ind=ind,
                #data=data
                #)
        #else:
            #fig_revchecker_depth.axis('off')

        # laminar depth relative to cortex layer 5
        # based on whitenoise stim, but the data exist for all stim except for fm
        #fig_lfp_depth = fig.add_subplot(spec[6:8,4])
        #if row['has_hf']:
            #sacc.lfp_laminar_depth(
                #ax=fig_lfp_depth,
                #row=row,
                #data=data,
                #ind=ind)
        #else:
            #fig_lfp_depth.axis('off')

        # whitenoise sta
        fig_wn_sta = fig.add_subplot(spec[1,0])
        if row['has_hf']:
            sacc.sta(ax=fig_wn_sta,
                        sta_name='Wn_spike_triggered_average',
                        title='Wn STA',
                        row=row)
        else:
            fig_wn_sta.axis('off')

        # whitenoise stv
        fig_wn_stv = fig.add_subplot(spec[1,1])
        if row['has_hf']:
            sacc.stv(ax=fig_wn_stv,
                        stv_name='Wn_spike_triggered_variance',
                        title='Wn STV',
                        row=row)
        else:
            fig_wn_stv.axis('off')

        # whitenoise eye movement psth
        fig_wn_eye_psth = fig.add_subplot(spec[1,2])
        if row['has_hf']:
            sacc.movement_psth(ax=fig_wn_eye_psth,
                                    rightsacc='Wn_saccade_rightPSTH',
                                    leftsacc='Wn_saccade_leftPSTH',
                                    title='Wn left/right saccades',
                                    row=row)
        else:
            fig_wn_eye_psth.axis('off')

        # whitenoise pupil radius tuning curve
        fig_wn_pupilradius_tuning = fig.add_subplot(spec[1,3])
        if row['has_hf']:
            _ = sacc.tuning_curve(
                ax=fig_wn_pupilradius_tuning,
                row=row,
                varcent_name='Wn_pupilradius_tuning_bins',
                tuning_name='Wn_pupilradius_tuning',
                err_name='Wn_pupilradius_tuning_err',
                title='Wn pupil radius',
                xlabel='pxls'
        )
        else:
            fig_wn_pupilradius_tuning.axis('off')

        # whitenoise running speed tuning curve
        fig_speed_tuning = fig.add_subplot(spec[1,4])
        if row['has_hf']:
            _ = sacc.tuning_curve(
                ax=fig_speed_tuning,
                row=row,
                varcent_name='Wn_ballspeed_tuning_bins',
                tuning_name='Wn_ballspeed_tuning',
                err_name='Wn_ballspeed_tuning_err',
                title='Wn ball speed',
                xlabel='cm/sec'
            )
        else:
            fig_speed_tuning.axis('off')

        # FmLt sta
        fig_FmLt_sta = fig.add_subplot(spec[2,0])
        sacc.sta(ax=fig_FmLt_sta,
                    sta_name='FmLt_spike_triggered_average',
                    title='FmLt STA',
                    row=row)

        # FmLt stv
        fig_FmLt_stv = fig.add_subplot(spec[2,1])
        sacc.stv(ax=fig_FmLt_stv,
                    stv_name='FmLt_spike_triggered_variance',
                    title='FmLt STV',
                    row=row)

        # FmLt gyro z tuning curve
        fig_FmLt_gyro_z_tuning = fig.add_subplot(spec[2,2])
        _ = sacc.tuning_curve(
            ax=fig_FmLt_gyro_z_tuning,
            row=row,
            varcent_name='FmLt_gyroz_tuning_bins',
            tuning_name='FmLt_gyroz_tuning',
            err_name='FmLt_gyroz_tuning_err',
            title='FmLt gyro z',
            xlabel='deg/sec'
        )

        # FmLt gyro x tuning curve
        fig_FmLt_gyro_x_tuning = fig.add_subplot(spec[2,3])
        _ = sacc.tuning_curve(
            ax=fig_FmLt_gyro_x_tuning,
            varcent_name='FmLt_gyrox_tuning_bins',
            tuning_name='FmLt_gyrox_tuning',
            err_name='FmLt_gyrox_tuning_err',
            title='FmLt gyro x',
            xlabel='deg/sec',
            row=row
        )

        # FmLt gyro y tuning curve
        fig_FmLt_gyro_y_tuning = fig.add_subplot(spec[2,4])
        _ = sacc.tuning_curve(
            ax=fig_FmLt_gyro_y_tuning,
            varcent_name='FmLt_gyroy_tuning_bins',
            tuning_name='FmLt_gyroy_tuning',
            err_name='FmLt_gyroy_tuning_err',
            title='FmLt gyro y',
            xlabel='deg/sec',
            row=row
        )

        # FmLt gaze shift dEye psth
        fig_FmLt_gaze_dEye = fig.add_subplot(spec[4,1])
        sacc.movement_psth(
            ax=fig_FmLt_gaze_dEye,
            rightsacc='FmLt_gazeshift_rightPSTH',
            leftsacc='FmLt_gazeshift_leftPSTH',
            title='FmLt gaze shift',
            row=row
        )
        
        # FmLt comp dEye psth
        fig_FmLt_comp_dEye = fig.add_subplot(spec[4,2])
        sacc.movement_psth(
            ax=fig_FmLt_comp_dEye,
            rightsacc='FmLt_compensatory_rightPSTH',
            leftsacc='FmLt_compensatory_leftPSTH',
            title='FmLt comp',
            row=row
        )

        # FmLt gaze shift dHead psth
        fig_FmLt_gaze_dHead = fig.add_subplot(spec[4,3])
        sacc.movement_psth(
            ax=fig_FmLt_gaze_dHead,
            rightsacc='FmLt_saccade_rightPSTH',
            leftsacc='FmLt_saccade_leftPSTH',
            title='FmLt all sacc',
            row=row
        )
        
        # FmLt comp dHead psth
        fig_FmLt_comp_dHead = fig.add_subplot(spec[4,4])
        fig_FmLt_comp_dHead.axis('off')

        fig_mean_grat_ori_tuning = fig.add_subplot(spec[6,0])
        if row['has_hf']:
            sacc.grat_stim_tuning(ax=fig_mean_grat_ori_tuning,
                                    tf_sel='mean',
                                    row=row,
                                    data=data,
                                    ind=ind)
        else:
            fig_mean_grat_ori_tuning.axis('off')
        
        fig_low_grat_ori_tuning = fig.add_subplot(spec[6,1])
        if row['has_hf']:
            sacc.grat_stim_tuning(ax=fig_low_grat_ori_tuning,
                                    tf_sel='low',
                                    row=row,
                                    data=data,
                                    ind=ind)
        else:
            fig_low_grat_ori_tuning.axis('off')

        fig_high_grat_ori_tuning = fig.add_subplot(spec[6,2])
        if row['has_hf']:
            sacc.grat_stim_tuning(ax=fig_high_grat_ori_tuning,
                                    tf_sel='high',
                                    row=row,
                                    data=data,
                                    ind=ind)
        else:
            fig_high_grat_ori_tuning.axis('off')

        # FmLt all dEye psth
        fig_FmLt_all_dEye = fig.add_subplot(spec[4,0])
        fig_FmLt_all_dEye.axis('off')

        # FmLt pupil radius tuning
        fig_FmLt_pupilradius_tuning = fig.add_subplot(spec[5,0])
        _ = sacc.tuning_curve(ax=fig_FmLt_pupilradius_tuning,
                              row=row,
                                varcent_name='FmLt_pupilradius_tuning_bins',
                                tuning_name='FmLt_pupilradius_tuning',
                                err_name='FmLt_pupilradius_tuning_err',
                                title='FmLt pupil radius',
                                xlabel='pupil radius')

        # FmLt theta tuning
        fig_FmLt_theta_tuning = fig.add_subplot(spec[5,1])
        _ = sacc.tuning_curve(ax=fig_FmLt_theta_tuning,
                              row=row,
                                varcent_name='FmLt_theta_tuning_bins',
                                tuning_name='FmLt_theta_tuning',
                                err_name='FmLt_theta_tuning_err',
                                title='FmLt theta',
                                xlabel='deg')

        # FmLt phi tuning
        fig_FmLt_phi_tuning = fig.add_subplot(spec[5,2])
        FmLt_phi_modind = sacc.tuning_curve(ax=fig_FmLt_phi_tuning,
                                            row=row,
                                varcent_name='FmLt_phi_tuning_bins',
                                tuning_name='FmLt_phi_tuning',
                                err_name='FmLt_phi_tuning_err',
                                title='FmLt phi',
                                xlabel='deg')
        data.at[ind, 'FmLt_phi_modind'] = FmLt_phi_modind

        # FmLt roll tuning
        fig_FmLt_roll_tuning = fig.add_subplot(spec[5,3])
        FmLt_roll_modind = sacc.tuning_curve(ax=fig_FmLt_roll_tuning,
                                             row=row,
                                varcent_name='FmLt_roll_tuning_bins',
                                tuning_name='FmLt_roll_tuning',
                                err_name='FmLt_roll_tuning_err',
                                title='FmLt roll',
                                xlabel='deg')
        data.at[ind, 'FmLt_roll_modind'] = FmLt_roll_modind

        # FmLt pitch tuning
        fig_FmLt_pitch_tuning = fig.add_subplot(spec[5,4])
        FmLt_pitch_modind = sacc.tuning_curve(ax=fig_FmLt_pitch_tuning,
                                              row=row,
                                varcent_name='FmLt_pitch_tuning_bins',
                                tuning_name='FmLt_pitch_tuning',
                                err_name='FmLt_pitch_tuning_err',
                                title='FmLt pitch',
                                xlabel='deg')
        data.at[ind, 'FmLt_pitch_modind'] = FmLt_pitch_modind

        if row['has_optic_flow']:
            fig_flow_full_vec = fig.add_subplot(spec[0,6])
            fig_flow_full_amp = fig.add_subplot(spec[0,5])

            fig_flow_ag_vec = fig.add_subplot(spec[1,6])
            fig_flow_ag_amp = fig.add_subplot(spec[1,5])

            fig_flow_ig_vec = fig.add_subplot(spec[2,6])
            fig_flow_ig_amp = fig.add_subplot(spec[2,5])

            if row['has_topdown_optic_flow']:
                fig_flow_rf_vec = fig.add_subplot(spec[3,6])
                fig_flow_rf_amp = fig.add_subplot(spec[3,5])

                fig_flow_rb_vec = fig.add_subplot(spec[4,6])
                fig_flow_rb_amp = fig.add_subplot(spec[4,5])

                fig_flow_fm_vec = fig.add_subplot(spec[5,6])
                fig_flow_fm_amp = fig.add_subplot(spec[5,5])

                fig_flow_im_vec = fig.add_subplot(spec[6,6])
                fig_flow_im_amp = fig.add_subplot(spec[6,5])

                movstates = ['full','active_gyro','inactive_gyro','running_forward',
                             'running_backward','fine_motion','immobile']
                statevecs = [fig_flow_full_vec, fig_flow_ag_vec, fig_flow_ig_vec,
                             fig_flow_rf_vec, fig_flow_rb_vec, fig_flow_fm_vec, fig_flow_im_vec]
                stateamps = [fig_flow_full_amp, fig_flow_ag_amp, fig_flow_ig_amp,
                             fig_flow_rf_amp, fig_flow_rb_amp, fig_flow_fm_amp, fig_flow_im_amp]

            elif not row['has_topdown_optic_flow']:
                movstates = ['full','active_gyro','inactive_gyro']
                statevecs = [fig_flow_full_vec, fig_flow_ag_vec, fig_flow_ig_vec]
                stateamps = [fig_flow_full_amp, fig_flow_ag_amp, fig_flow_ig_amp]

            for i in range(len(movstates)):
                sacc.optic_flow_vec(ax=statevecs[i],
                                    movstate=movstates[i])
                sacc.optic_flow_amp(ax=stateamps[i],
                                movstate=movstates[i])

        # set up panels for dark figures
        fig_fmdark_gyro_z_tuning = fig.add_subplot(spec[7,0])
        fig_fmdark_gyro_x_tuning = fig.add_subplot(spec[7,1])
        fig_fmdark_gyro_y_tuning = fig.add_subplot(spec[7,2])
        fig_fmdark_gaze_dEye = fig.add_subplot(spec[8,1])
        fig_fmdark_comp_dEye = fig.add_subplot(spec[8,2])
        fig_fmdark_gaze_dHead = fig.add_subplot(spec[8,3])
        fig_fmdark_comp_dHead = fig.add_subplot(spec[8,4])
        fig_fmdark_all_dEye = fig.add_subplot(spec[8,0])
        fig_fmdark_pupilradius_tuning = fig.add_subplot(spec[9,0])
        fig_fmdark_theta_tuning = fig.add_subplot(spec[9,1])
        fig_fmdark_phi_tuning = fig.add_subplot(spec[9,2])
        fig_fmdark_roll_tuning = fig.add_subplot(spec[9,3])
        fig_fmdark_pitch_tuning = fig.add_subplot(spec[9,4])

        if not row['has_dark']:
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

        elif row['has_dark']:
            # fm dark gyro z tuning curve
            fmdark_gyro_z_tuning_modind = sacc.tuning_curve(
                ax=fig_fmdark_gyro_z_tuning,
                row=row,
                varcent_name='FmDk_gyroz_tuning_bins',
                tuning_name='FmDk_gyroz_tuning',
                err_name='FmDk_gyroz_tuning_err',
                title='FmDk gyro z',
                xlabel='deg/sec'
            )
            data.at[ind, 'FmDk_gyroz_modind'] = fmdark_gyro_z_tuning_modind

            # fm dark gyro x tuning curve
            fmdark_gyro_x_tuning_modind = sacc.tuning_curve(
                ax=fig_fmdark_gyro_x_tuning,
                row=row,
                varcent_name='FmDk_gyrox_tuning_bins',
                tuning_name='FmDk_gyrox_tuning',
                err_name='FmDk_gyrox_tuning_err',
                title='FmDk gyro x',
                xlabel='deg/sec'
            )
            data.at[ind, 'FmDk_gyrox_modind'] = fmdark_gyro_x_tuning_modind

            # fm dark gyro y tuning curve
            fmdark_gyro_y_tuning_modind = sacc.tuning_curve(
                ax=fig_fmdark_gyro_y_tuning,
                row=row,
                varcent_name='FmDk_gyroy_tuning_bins',
                tuning_name='FmDk_gyroy_tuning',
                err_name='FmDk_gyroy_tuning_err',
                title='FmDk gyro y',
                xlabel='deg/sec'
            )
            data.at[ind, 'FmDk_gyroy_modind'] = fmdark_gyro_y_tuning_modind

            # fm dark gaze shift dEye psth
            sacc.movement_psth(
                ax=fig_fmdark_gaze_dEye,
                row=row,
                rightsacc='FmDk_rightsacc_avg_gaze_shift_dEye',
                leftsacc='FmDk_leftsacc_avg_gaze_shift_dEye',
                title='FmDk gaze shift dEye'
            )

            # fm dark comp dEye psth
            sacc.movement_psth(
                ax=fig_fmdark_comp_dEye,
                row=row,
                rightsacc='FmDk_rightsacc_avg_comp_dEye',
                leftsacc='FmDk_leftsacc_avg_comp_dEye',
                title='FmDk comp dEye'
            )


            # fm dark gaze shift dHead psth
            sacc.movement_psth(
                ax=fig_fmdark_gaze_dHead,
                row=row,
                rightsacc='FmDk_rightsacc_avg_gaze_shift_dHead',
                leftsacc='FmDk_leftsacc_avg_gaze_shift_dHead',
                title='FmDk gaze shift dHead',
            )

            # fm dark comp dHead psth
            sacc.movement_psth(
                ax=fig_fmdark_comp_dHead,
                row=row,
                rightsacc='FmDk_rightsacc_avg_comp_dHead',
                leftsacc='FmDk_leftsacc_avg_comp_dHead',
                title='FmDk comp dHead',
            )

            # fm dark all dEye psth
            sacc.movement_psth(
                ax=fig_fmdark_all_dEye,
                row=row,
                rightsacc='FmDk_rightsacc_avg',
                leftsacc='FmDk_leftsacc_avg',
                title='FmDk all dEye',
            )

            # fm dark pupil radius tuning
            fmdark_pupilradius_modind = sacc.tuning_curve(
                ax=fig_fmdark_pupilradius_tuning,
                row=row,
                varcent_name='FmDk_pupilradius_tuning_bins',
                tuning_name='FmDk_pupilradius_tuning',
                err_name='FmDk_pupilradius_tuning_err',
                title='FmDk pupil radius',
                xlabel='pxls')
            data.at[ind, 'FmDk_pupilradius_modind'] = fmdark_pupilradius_modind

            # fm dark theta tuning
            fmdark_theta_modind = sacc.tuning_curve(
                ax=fig_fmdark_theta_tuning,
                row=row,
                varcent_name='FmDk_theta_tuning_bins',
                tuning_name='FmDk_theta_tuning',
                err_name='FmDk_theta_tuning_err',
                title='FmDk theta',
                xlabel='deg'
            )
            data.at[ind, 'FmDk_theta_modind'] = fmdark_theta_modind

            # fm dark phi tuning
            fmdark_phi_modind = sacc.tuning_curve(
                ax=fig_fmdark_phi_tuning,
                row=row,
                varcent_name='FmDk_phi_tuning_bins',
                tuning_name='FmDk_phi_tuning',
                err_name='FmDk_phi_tuning_err',
                title='FmDk phi',
                xlabel='deg'
            )
            data.at[ind, 'FmDk_phi_modind'] = fmdark_phi_modind

            # fm dark roll tuning
            fmdark_roll_modind = sacc.tuning_curve(
                ax=fig_fmdark_roll_tuning,
                row=row,
                varcent_name='FmDk_roll_tuning_bins',
                tuning_name='FmDk_roll_tuning',
                err_name='FmDk_roll_tuning_err',
                title='FmDk roll',
                xlabel='deg'
            )
            data.at[ind, 'FmDk_roll_modind'] = fmdark_roll_modind
            
            # fm dark pitch tuning
            fmdark_pitch_modind = sacc.tuning_curve(
                ax=fig_fmdark_pitch_tuning,
                row=row,
                varcent_name='FmDk_pitch_tuning_bins',
                tuning_name='FmDk_pitch_tuning',
                err_name='FmDk_pitch_tuning_err',
                title='FmDk pitch',
                xlabel='deg'
            )
            data.at[ind, 'FmDk_pitch_modind'] = fmdark_pitch_modind

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print('saving unit summary pdf')
    pdf.close()


