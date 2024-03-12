


import numpy as np

import saccadeAnalysis as sacc


def get_norm_FmLt_PSTHs(data):

    # append str
    # ap = '1'
    ap = ''

    for ind, row in data.iterrows():

        pref, nonpref, prefname, nonprefname = sacc.get_direction_pref(
            left = row['FmLt_gazeshift_leftPSTH'],
            right = row['FmLt_gazeshift_rightPSTH']
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
        data.at[ind, 'pref_comp_psth_raw'] = row['FmLt_compensatory_{}PSTH'.format(
                                                prefname)].copy().astype(object)
        
        data.at[ind, 'nonpref_comp_psth_raw'] = row['FmLt_compensatory_{}PSTH'.format(
                                                 nonprefname)].copy().astype(object)
        
        # Normalize compensatory eye/head movement PSTHs.
        data.at[ind, 'pref_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmLt_compensatory_{}PSTH'.format(prefname)],
            rawpref = pref,
            trange = 'fm'
        ).astype(object)
        
        data.at[ind, 'nonpref_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmLt_compensatory_{}PSTH'.format(nonprefname)],
            rawpref = pref,
            trange = 'fm'
        ).astype(object)

    return data

def get_norm_FmDk_PSTHs(data):
    for ind, row in data.iterrows():

        pref = row['FmLt_gazeshift_{}_saccPSTH_dHead{}'.format(
                                row['pref_gazeshift_direction'])]
        
        # gaze shifts
        data.at[ind, 'pref_dark_gazeshift_psth'] = sacc.norm_PSTH(
            psth = pref,
            trange = 'fm'
        ).astype(object)

        data.at[ind, 'nonpref_dark_gazeshift_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_gazeshift_{}_saccPSTH_dHead{}'.format(
                                row['nonpref_gazeshift_direction'])],
            rawpref = pref).astype(object)
        
        # compensatory
        data.at[ind, 'pref_dark_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_comp_{}_saccPSTH_dHead{}'.format(
                        row['pref_gazeshift_direction'], ap)],
            rawpref = pref).astype(object)
        
        data.at[ind, 'nonpref_dark_comp_psth'] = sacc.norm_PSTH(
            psth = row['FmDk_comp_{}_saccPSTH_dHead{}'.format(
            row['nonpref_gazeshift_direction'], ap)],
            rawpref = pref).astype(object)
        
        # raw gaze shifts
        data.at[ind, 'pref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead{}'.format(
                                                    row['pref_gazeshift_direction'], ap)].astype(object)
        data.at[ind, 'nonpref_dark_gazeshift_psth_raw'] = row['FmDk_gazeshift_{}_saccPSTH_dHead{}'.format(
                                                    row['nonpref_gazeshift_direction'], ap)].astype(object)
        
        # compensatory
        data.at[ind, 'pref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead{}'.format(
                                                    row['pref_gazeshift_direction'], ap)].astype(object)
        data.at[ind, 'nonpref_dark_comp_psth_raw'] = row['FmDk_comp_{}_saccPSTH_dHead{}'.format(
                                                    row['nonpref_gazeshift_direction'], ap)].astype(object)

    return data

def get_norm_Hf_PSTHs(data):

    data = drop_if_missing(data, 'Rc_stim_PSTH')
    data = drop_if_missing(data, 'Gt_stim_PSTH')
    data = drop_if_missing(data, 'Sn_stim_PSTH_onSub_bckgndRF')

    
    for ind, row in data.iterrows():

        # reversing checkerboard
        #data.at[ind, 'norm_Rc_psth'] = sacc.norm_PSTH(
            #psth = row['Rc_stim_PSTH'],
            #trange = 'fm'
        #).astype(object)
        
        # gratings
       #data.at[ind, 'norm_gratings_psth'] = sacc.norm_PSTH(
         #   psth = row['Gt_stim_PSTH'],
          #  trange = 'gt'
        #).astype(object)
        
        # sparse noise
        #data.at[ind, 'norm_Sn_psth'] = sacc.norm_PSTH(
            #row['Sn_stim_PSTH_onSub_bckgndRF'],
            #trange = 'sn'
        #).astype(object)

    return data

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

    return data


def drop_if_missing(data, key):

    has = []
    for i in data[key]:
        if type(i) == float:
            has.append(False)
        else:
            has.append(True)

    data['has'] = has

    data = data[data['has']==True]

    data.drop(columns=['has'], inplace=True)

    return data