import os, json
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from multiprocessing import Pool

def calc_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000, num_events=None):
    """
    calcualtes for a single cell at a time

    bandwidth (in msec)
    resample_size (msec)
    edgedrop (msec to drop at the start and end of the window so eliminate artifacts of filtering)
    win = 1000msec before and after
    """

    # Some conversions
    bandwidth = bandwidth / 1000
    resample_size = resample_size / 1000
    win = win / 1000
    edgedrop = edgedrop / 1000
    edgedrop_ind = int(edgedrop / resample_size)

    # Setup time bins. The bins of the returned PSTH will be -`win` msec to +`win` msec
    # with `resame_size` msec bins, where 0 is the time of the event. Because of edge effects
    # from the gaussian filter, we'll calculate the PSTH with an extra `edgedrop` msec at the
    # start and end so that the edges of the PSTH can be dropped without losing timepoints
    # we care about.
    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # if there is only one event time (i.e. if it's an int for all spikes like eventT=np.array(0)
    if np.size(eventT)>1:

        # Get the timestamps of spikes relative to events in `eventT`
        sps = []
        for i, t in enumerate(eventT):
            sp = spikeT-t
            # Only keep spikes in this window
            sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))]
            sps.extend(sp)
        num_events = np.size(eventT)
    else:
        sps = eventT.copy().flatten()
    
    # If `win` is 1000 msec, values in `sps` will fall between -1 and 1, since only spikes
    # that fall before or after the event by 1 second are included.
    # (Because of `edge_drop`, values can be a bit beyond `win` but those will be eliminated before
    # the PSTH is returned.
    sps = np.array(sps)

    # Calculate the PSTH using Kernel Density Estimation
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:, np.newaxis])
    density = kernel.score_samples(bins[:, np.newaxis])

    # Here, `density` is a probability function that sums to 1. We want it in sp/sec.
    # We need to multiply by the # spikes to get the spike count per point. Then,
    # we divide by the number of events so that we get teh rate per event.
    psth = np.exp(density) * (np.size(sps) / num_events)

    # Drop the extra time at the start and end of the window, since it will have
    # edge effects from the gaussian filter. After dropping this, it will be the
    # size set by `win`, so no important timepoints are actually lost here.
    psth = psth[edgedrop_ind:-edgedrop_ind]

    return psth

def normalize_psth(psth):
    pref = psth.copy()
    bsln = np.mean(psth[0:150]) # was -100 to -50 ms ; now, -200 to -50 ms
    norm_psth = (psth - bsln) / np.max(pref[200:]) # 0 to 200
    return norm_psth

def mp_PSTH_from_sp():

def main():

    savepath = '/home/niell_lab/Desktop/crossval.json'
    # data = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_Final.mat')
    totdata = loadmat('/home/niell_lab/Data/marmoset/gazeshift/Pooled_V1Hart_Preload_TotalInfo.mat')['TotalInfo']

    spikeT = {}
    for u in range(len(totdata)):
        spikeT[u] = totdata[u][0][0][0]['SacImage'][0][0]['StimRast2']

    crossval = {}

    for u, rast in tqdm(spikeT.items()):

        if u>2:
            break

        print('Getting train/test set inds')
        trials = np.unique(rast[:,1]).astype(int)-1

        train_inds = np.array(sorted(np.random.choice(trials, size=int(np.floor(trials.size/2)), replace=False)))
        test_inds = trials.copy()
        test_inds = np.delete(test_inds, train_inds)
        
        prop_dict = {}

        for tt, tname in enumerate(['train', 'test']):
            
            inds = [train_inds, test_inds][tt]
        
            print(' -> Getting spikes for each {} event'.format(tname))

            all_sps = []
            for tnum in inds:
                sps = rast.copy()
                sps = sps[sps[:,1].astype(int)==int(tnum)]
                sps = sps[:,0]
                all_sps.extend(sps)

            prop_dict['{}_spikeT'.format(tname)] = all_sps



            # PSTH
            psth = calc_PSTH(np.array(all_sps), np.zeros(inds.size), bandwidth=5, win=400)
            psth = psth[200:800]

            print(' -> Normalize')
            norm_psth = normalize_psth(psth)


            print(' -> Response latency')
            # latency
            peakT = np.argmax(norm_psth)-200 # in ms since onset of eye movement
            
            print(' -> Adding to unit dict')
            prop_dict['{}_rawPsth'.format(tname)] = psth
            prop_dict['{}_normPsth'.format(tname)] = norm_psth
            prop_dict['{}_peakT'.format(tname)] = peakT
        
        print('Done with unit')
        crossval[u] = prop_dict

    print('Saving json')
    with open(savepath, 'w') as f:
        json.dump(crossval, f)

if __name__ == '__main__':
    main()