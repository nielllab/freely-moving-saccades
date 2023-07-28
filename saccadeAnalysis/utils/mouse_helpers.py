


import numpy as np

import saccadeAnalysis as sacc


def get_norm_FmLt_PSTHs(data):

    for ind, row in data.iterrows():

        pref, nonpref, prefname, nonprefname = sacc.get_direction_pref(
            left = row['FmLt_gazeshift_left_saccPSTH_dHead1'],
            right = row['FmLt_gazeshift_right_saccPSTH_dHead1']
        )
        
        data.at[ind,'pref_gazeshift_direction'] = prefname
        data.at[ind,'nonpref_gazeshift_direction'] = nonprefname

        data.at[ind,'gazeshift_DSI'] = sacc.calc_PSTH_DSI(
            pref = pref,
            nonpref = nonpref
        )

        # Normalize gazeshift PSTHs.
        data.at[ind, 'pref_gazeshift_psth'] = sacc.norm_PSTH(
            psth = pref,
            trange = 'fm'
        ).astype(object)
        
        data.at[ind, 'nonpref_gazeshift_psth'] = sacc.norm_PSTH(
            psth = nonpref,
            rawpref = pref,
            trange = 'fm'
        ).astype(object)
        
        # Keep a copy of the raw PSTHs in the preferred and non-preferred directions.
        data.at[ind, 'pref_gazeshift_psth_raw'] = pref.copy().astype(object)

        data.at[ind, 'nonpref_gazeshift_psth_raw'] = nonpref.copy().astype(object)
        
        # Also keep copies of the raw PSTHs for compensatory eye/head movements.
        data.at[ind, 'pref_comp_psth_raw'] = row['FmLt_comp_{}_saccPSTH_dHead1'.format(
                                                prefname)].copy().astype(object)
        
        data.at[ind, 'nonpref_comp_psth_raw'] = row['FmLt_comp_{}_saccPSTH_dHead1'.format(
                                                 nonprefname)].copy().astype(object)
        
        # Normalize compensatory eye/head movement PSTHs.
        data.at[ind, 'pref_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmLt_comp_{}_saccPSTH_dHead1'.format(prefname)],
            rawpref = pref,
            trange = 'fm'
        ).astype(object)
        
        data.at[ind, 'nonpref_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmLt_comp_{}_saccPSTH_dHead1'.format(nonprefname)],
            rawpref = pref,
            trange = 'fm'
        ).astype(object)

        

    return data

def get_norm_FmDk_PSTHs(data):
    for ind, row in data.iterrows():

        pref = row['FmLt_gazeshift_{}_saccPSTH_dHead1'.format(
                                row['pref_gazeshift_direction'])]
        
        # gaze shifts
        data.at[ind, 'pref_dark_gazeshift_psth'] = sacc.norm_PSTH(
            psth = pref,
            trange = 'fm'
        ).astype(object)

        data.at[ind, 'nonpref_dark_gazeshift_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(
                                row['nonpref_gazeshift_direction'])],
            rawpref = pref).astype(object)
        
        # compensatory
        data.at[ind, 'pref_dark_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_comp_{}_saccPSTH_dHead1'.format(
                        row['pref_gazeshift_direction'])],
            rawpref = pref).astype(object)
        
        data.at[ind, 'nonpref_dark_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_comp_{}_saccPSTH_dHead1'.format(
            row['nonpref_gazeshift_direction'])],
            rawpref = pref).astype(object)
        
        # raw gaze shifts
        data.at[ind, 'pref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(
                                                    row['pref_gazeshift_direction'])].astype(object)
        data.at[ind, 'nonpref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead1'.format(
                                                    row['nonpref_gazeshift_direction'])].astype(object)
        
        # compensatory
        data.at[ind, 'pref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead1'.format(
                                                    row['pref_gazeshift_direction'])].astype(object)
        data.at[ind, 'nonpref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead1'.format(
                                                    row['nonpref_gazeshift_direction'])].astype(object)


def get_norm_Hf_PSTHs(data):

    for ind, row in data.iterrows():

        # reversing checkerboard
        data.at[ind, 'norm_Rc_psth'] = sacc.norm_PSTH(
            psth = row['Rc_psth'],
            trange = 'fm'
        ).astype(object)
        
        # gratings
        data.at[ind, 'norm_gratings_psth'] = sacc.norm_PSTH(
            psth = row['Gt_grating_psth'],
            trange = 'gt'
        ).astype(object)
        
        # sparse noise
        data.at[ind, 'norm_Sn_psth'] = sacc.norm_PSTH(
            row['Sn_on_background_psth'],
            trange = 'sn'
        ).astype(object)

def FmLtDk_peak_time(data):

    ## Peak time
    psth_bins = np.arange(-1,1.001,1/1000)

    ### FmLt
    for ind, row in data.iterrows():
        raw_psth = row['pref_gazeshift_psth_raw']
        norm_psth = row['pref_gazeshift_psth']
        
        peakT, peak_val = sacc.calc_PSTH_latency(norm_psth)
        
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
        
        peakT, peak_val = sacc.calc_PSTH_latency(norm_psth)
        
        data.at[ind, 'FmDk_gazeshift_baseline'] = raw_psth[0:800].astype(object)
        data.at[ind, 'FmDk_gazeshift_med_baseline'] = np.median(raw_psth[0:800])
        data.at[ind, 'FmDk_gazeshift_peak_val'] = peak_val
        data.at[ind, 'FmDk_gazeshift_peakT'] = peakT

