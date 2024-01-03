

import numpy as np
import matplotlib.pyplot as plt


def tuning_modulation_index(tuning):
    """ Calculate the modulation index of a tuning curve.
    """

    tuning = tuning[~np.isnan(tuning)]
    modind = (np.max(tuning) - np.min(tuning)) / (np.max(tuning) + np.min(tuning))
    
    return modind


def saccade_modulation_index(saccavg):
    """ Calculate the modulation index of a saccade-triggered PSTH.
    """

    model_dt = 0.025

    trange = np.arange(-1, 1.1, model_dt)

    t0ind = (np.abs(trange-0)).argmin()

    t100ind = t0ind+4
    baseline = np.nanmean(saccavg[0:int(t100ind-((1/4)*t100ind))])

    r0 = np.round((saccavg[t0ind] - baseline) / (saccavg[t0ind] + baseline), 3)

    r100 = np.round((saccavg[t100ind] - baseline) / (saccavg[t100ind] + baseline), 3)
    
    return r0, r100


def waveform(ax, row):

    samprate = 30000

    wv = row['FmLt_waveform']
    ax.plot(np.arange(len(wv)) * 1000 / samprate, wv, color='k')
    ax.set_ylabel('millivolts')
    ax.set_xlabel('msec')
    ax.set_title(row['FmLt_KSLabel']+' cont='+str(np.round(row['FmLt_ContamPct'],3)), fontsize=20)


def tuning_curve(ax, row, varcent_name, tuning_name, err_name, title, xlabel):

    var_cent = row[varcent_name]
    tuning = row[tuning_name]
    tuning_err = row[err_name]
    ax.errorbar(var_cent,tuning[:],yerr=tuning_err[:], color='k')

    modind = tuning_modulation_index(tuning)

    ax.set_title(title+'\nmod.ind.='+str(modind), fontsize=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('sp/sec')
    maxval = np.nanmax(tuning[:]*1.2)
    if not np.isfinite(maxval):
        maxval = 1
    ax.set_ylim(0, maxval*1.2)

    return modind


def grat_stim_tuning(ax, row, ind, data, tf_sel='mean'):

    # [low, mid, high, spont]
    cmap_orientation = ['#fec44f','#ec7014','#993404','#000000']

    if tf_sel=='mean':
        raw_tuning = np.mean(row['Gt_ori_tuning_tf'],2)
    elif tf_sel=='low':
        raw_tuning = row['Gt_ori_tuning_tf'][:,:,0]
    elif tf_sel=='high':
        raw_tuning = row['Gt_ori_tuning_tf'][:,:,1]
    drift_spont = row['Gt_drift_spont']
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

    ax.set_title(tf_sel+' tf\n OSI l='+str(np.round(osi[0],3))+'m='+str(np.round(osi[1],3))+'h='+str(np.round(osi[2],3))+'\n DSI l='+str(np.round(dsi[0],3))+'m='+str(np.round(dsi[1],3))+'h='+str(np.round(dsi[2],3)), fontsize=20)
    ax.plot(np.arange(8)*45, raw_tuning[:,0], label='low sf', color=cmap_orientation[0])
    ax.plot(np.arange(8)*45, raw_tuning[:,1], label='mid sf', color=cmap_orientation[1])
    ax.plot(np.arange(8)*45, raw_tuning[:,2], label='high sf', color=cmap_orientation[2])
    ax.plot([0,315],[drift_spont,drift_spont],':',label='spont', color=cmap_orientation[3])
    
    if tf_sel=='mean':
        ax.legend()
    ax.set_ylim([0,np.nanmax(row['Gt_ori_tuning_tf'][:,:,:])*1.2])
    
    if tf_sel=='mean':
        data.at[ind, 'Gt_osi_low'] = osi[0]
        data.at[ind, 'Gt_osi_mid'] = osi[1]
        data.at[ind, 'Gt_osi_high'] = osi[2]
        data.at[ind, 'Gt_dsi_low'] = dsi[0]
        data.at[ind, 'Gt_dsi_mid'] = dsi[1]
        data.at[ind, 'Gt_dsi_high'] = dsi[2]


#def revchecker_laminar_depth(ax, row, ind, data):

#    if np.size(row['Rc_response_by_channel'],0) == 64:
#        shank_channels = [c for c in range(np.size(row['Rc_response_by_channel'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['FmLt_ch'])/32))]
#        whole_shank = row['Rc_response_by_channel'][shank_channels]
#        shank_num = [0 if np.max(shank_channels) < 40 else 1][0]
#        colors = plt.cm.jet(np.linspace(0,1,32))
#        for ch_num in range(len(shank_channels)):
#            ax.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
#        ax.plot(whole_shank[row['Rc_layer4cent'][shank_num]], color=colors[row['Rc_layer4cent'][shank_num]], label='layer4', linewidth=4) # layer 4
    
#    elif np.size(row['Rc_response_by_channel'],0) == 16:
#        whole_shank = row['Rc_response_by_channel']
#        colors = plt.cm.jet(np.linspace(0,1,16))
#        shank_num = 0
#        for ch_num in range(16):
#            ax.plot(row['Rc_response_by_channel'][ch_num], color=colors[ch_num], alpha=0.3, linewidth=1) # all other channels
#        ax.plot(whole_shank[row['Rc_layer4cent']], color=colors[row['Rc_layer4cent']], label='layer4', linewidth=1) # layer 4
#    
#    elif np.size(row['Rc_response_by_channel'],0) == 128:
#        shank_channels = [c for c in range(np.size(row['Rc_response_by_channel'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['FmLt_ch'])/32))]
#        whole_shank = row['Rc_response_by_channel'][shank_channels]
#        shank_num = int(np.floor(int(row['FmLt_ch'])/32))
#        colors = plt.cm.jet(np.linspace(0,1,32))
#        for ch_num in range(len(shank_channels)):
#            ax.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
#        ax.plot(whole_shank[row['Rc_layer4cent'][shank_num]], color=colors[row['Rc_layer4cent'][shank_num]], label='layer4', linewidth=4) # layer 4
#    
#    else:
#       print('unrecognized probe count in LFP plots during unit summary! index='+str(ind))
#    ax.plot(row['Rc_response_by_channel'][row['FmLt_ch']%32], color=colors[row['FmLt_ch']%32], label='this channel', linewidth=4) # current channel
#    depth_to_layer4 = 0 # could be 350um, but currently, everything will stay relative to layer4 since we don't know angle of probe & other factors
#    # if row['probe_name'] == 'DB_P64-8':
#    #     ch_spacing = 25/2
#    # else:
#    ch_spacing = 25
#    if shank_num == 0:
#        position_of_ch = int(row['Rc_relative_depth'][0][row['FmLt_ch']])
#        data.at[ind, 'Rc_ch_lfp_relative_depth'] = position_of_ch
#        depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
#        data.at[ind, 'Rc_depth_from_layer4'] = depth_from_surface
#        ax.set_title('ch='+str(row['FmLt_ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
#    elif shank_num == 1:
#        position_of_ch = int(row['Rc_relative_depth'][1][row['FmLt_ch']-32])
#        data.at[ind, 'Rc_ch_lfp_relative_depth'] = position_of_ch
#        depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
#        data.at[ind, 'Rc_depth_from_layer4'] = depth_from_surface
#        ax.set_title('ch='+str(row['FmLt_ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
#    elif shank_num == 2:
#        position_of_ch = int(row['Rc_relative_depth'][1][row['FmLt_ch']-64])
#        data.at[ind, 'Rc_ch_lfp_relative_depth'] = position_of_ch
#        depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
#        data.at[ind, 'Rc_depth_from_layer4'] = depth_from_surface
#        ax.set_title('ch='+str(row['FmLt_ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
#    elif shank_num == 3:
#        position_of_ch = int(row['Rc_relative_depth'][1][row['FmLt_ch']-96])
#        data.at[ind, 'Rc_ch_lfp_relative_depth'] = position_of_ch
#        depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
#       data.at[ind, 'Rc_depth_from_layer4'] = depth_from_surface
#        ax.set_title('ch='+str(row['FmLt_ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
#    ax.legend(); ax.axvline(x=(0.1*30000), color='k', linewidth=1)
#    ax.set_xticks(np.arange(0,18000,18000/8))
#    ax.set_xticklabels(np.arange(-100,500,75))
#    ax.set_xlabel('msec')
#  ax.set_ylabel('uvolts')


def grat_psth(ax, row):

    bins = np.linspace(-1.5, 1.5, 3001)

    ax.plot(bins, row['Gt_stim_PSTH'], color='k')
    ax.set_title('gratings psth', fontsize=20)
    ax.set_xlabel('time')
    ax.set_ylabel('sp/sec')
    maxval = np.nanmax(row['Gt_stim_PSTH'])
    if not np.isfinite(maxval):
        maxval = 1
    ax.set_ylim([0, maxval*1.2])


def lfp_laminar_depth(ax, row, data, ind):
    power_profiles = row['Wn_lfp_power']
    ch_shank = int(np.floor(row['FmLt_ch']/32))
    ch_shank_profile = power_profiles[ch_shank]
    ch_power = ch_shank_profile[int(row['FmLt_ch']%32)]
    layer5cent = row['Wn_layer5cent_from_lfp'][ch_shank]
    
    # if row['probe_name'] == 'DB_P64-8':
    #     ch_spacing = 25/2
    # else:
    ch_spacing = 25

    ch_depth = ch_spacing*(row['FmLt_ch']%32)-(layer5cent*ch_spacing)
    num_sites = 32
    ax.plot(ch_shank_profile,range(0,num_sites),color='k')
    ax.plot(ch_shank_profile[layer5cent]+0.01,layer5cent,'r*',markersize=12)
    ax.hlines(y=row['FmLt_ch']%32, xmin=ch_power, xmax=1, colors='g', linewidth=5)
    ax.set_ylim([33,-1])
    ax.set_yticks(list(range(-1,num_sites+1)))
    ax.set_yticklabels(ch_spacing*np.arange(num_sites+2)-(layer5cent*ch_spacing))
    ax.set_title('shank='+str(ch_shank)+' site='+str(row['FmLt_ch']%32)+'\n depth='+str(ch_depth), fontsize=20)
    data.at[ind, 'Wn_depth_from_layer5'] = ch_depth


def sta(ax, row, sta_name, title):
    # wnsta = np.reshape(row[sta_name],tuple(row[shape_name]))
    wnsta = row[sta_name]
    sta_range = np.max(np.abs(wnsta))*1.2
    sta_range = (0.25 if sta_range<0.25 else sta_range)
    ax.set_title(title, fontsize=20)
    ax.imshow(wnsta, vmin=-sta_range, vmax=sta_range, cmap='seismic')
    ax.axis('off')


def stv(ax, row, stv_name, title):
    # wnstv = np.reshape(row[stv_name],tuple(row[shape_name]))
    wnstv = row[stv_name]
    ax.imshow(wnstv, vmin=-1, vmax=1, cmap='cividis')
    ax.set_title(title, fontsize=20)
    ax.axis('off')


def movement_psth(ax, row, rightsacc, leftsacc, title, show_legend=False):

    cmap_sacc = ['steelblue','coral'] # [right, left]
    bins = np.linspace(-1, 1, 2001)

    rightavg = row[rightsacc]
    leftavg = row[leftsacc]
    ax.set_title(title, fontsize=20)

    ax.plot(bins, rightavg, color=cmap_sacc[0])

    ax.plot(bins, leftavg, color=cmap_sacc[1])

    if show_legend:
        ax.legend(['right','left'], loc=1)
    maxval = np.max(np.maximum(rightavg, leftavg))*1.2
    if not np.isfinite(maxval):
        maxval = 1
    ax.set_ylim([0, maxval])


def is_empty_index(data, attr, savekey):
    for ind, val in data[attr].iteritems():
        data.at[ind, savekey] = (True if ~np.isnan(val).all() else False)


def is_empty_cell(row, name):
    if name in row.index.values and type(row[name]) != float and row[name] == []:
        return True
    else:
        return False
