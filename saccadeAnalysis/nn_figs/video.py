""" Generate supplemental video for the manuscript.
saccadeAnalysis/nn_figs/video.py

Written by DMM, 2022
"""


import ray
from ray.actor import ActorHandle
from asyncio import Event
from typing import Tuple
from time import sleep
from tqdm.auto import tqdm

import os
import cv2
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size':10})

import fmEphys as fme



@ray.remote
def plot_frame_img(currentT, plot_t0, plot_tlen, spikeT, saccT,
                   eyevid, eyevidT, worldvid, worldvidT, klabels,
                   pbar:ActorHandle,):
    
    font_sz = 18
    t_start = 13  # was 13
    tlen = 3 # was 3
    numFr = np.argmin(np.abs(worldvidT-(t_start+tlen))) - np.argmin(np.abs(worldvidT-t_start))
    
    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    cat_cmap = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey'
    }
    
    fig = plt.figure(constrained_layout=True, figsize=(9,6), dpi=200)
    spec = fig.add_gridspec(ncols=2, nrows=5)

    ax_eyecam = fig.add_subplot(spec[:2,0])
    ax_worldcam = fig.add_subplot(spec[:2,1])
    ax_raster = fig.add_subplot(spec[2:,:])

    worldFr = np.argmin(np.abs(worldvidT-currentT))
    eyeFr = np.argmin(np.abs(eyevidT-currentT))

    for i, sps in spikeT.items():
        sps = sps[(sps>plot_t0) * (sps<(plot_t0+plot_tlen))]
        ax_raster.plot(sps, np.ones(len(sps))*i, '|',
                       color=cat_cmap[klabels[i]], markersize=5)
    ax_raster.set_ylim([len(spikeT.keys())+1, -4])
    ax_raster.set_xlim([plot_t0, plot_t0+plot_tlen])
    ax_raster.set_xticks(np.arange(t_start, t_start+tlen+0.5, 0.5))
    ax_raster.set_xticklabels((np.arange(0, tlen+0.5, 0.5)*1000).astype(int),
                              fontsize=font_sz)
    ax_raster.set_ylabel('cells', fontsize=font_sz)
    ax_raster.set_xlabel('time (ms)', fontsize=font_sz)
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['top'].set_visible(False)
    ax_raster.set_yticks(np.arange(0, 120, 20))
    ax_raster.set_yticklabels(np.arange(0, 120, 20), fontsize=font_sz)

    ax_raster.vlines(currentT, -2, len(spikeT.keys())+2, color='tab:blue')

    # show saccades
    show_saccT = saccT[(saccT>plot_t0) * (saccT<(plot_t0+plot_tlen))]
    ax_raster.plot(show_saccT, np.ones(len(show_saccT)) * -3,
                   '.', color='k', marker='v', markersize=7)

    # eyecam
    ax_eyecam.imshow(eyevid[eyeFr,217:,150:500].astype(np.uint8), cmap='gray')
    ax_eyecam.axis('off')

    # worldcam
    ax_worldcam.imshow(worldvid[worldFr,:,:], cmap='gray')
    ax_worldcam.axis('off')

    plt.tight_layout()

    width, height = fig.get_size_inches() * fig.get_dpi()
    fig.canvas.draw() # draw the canvas, cache the renderer
    images = np.frombuffer(fig.canvas.tostring_rgb(),
                    dtype='uint8').reshape(int(height), int(width), 3)
    
    plt.close()
    pbar.update.remote(1)
    return images



@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter



class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return



def avi_to_arr(path, ds=0.25):
    vid = cv2.VideoCapture(path)
    # array to put video frames into
    # will have the shape: [frames, height, width] and be returned with dtype=int8
    arr = np.empty([int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    # iterate through each frame
    for f in range(0,int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        img_s = cv2.resize(img, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)
    return arr



def drop_repeat_sacc(eventT, onset=True, win=0.020):
    """For saccades spanning multiple camera
    frames, only keep one saccade time. Either first or last.

    If `onset`, keep the first in a sequence (i.e. the onset of
    the movement). otherwise, keep the final saccade in the sequence

    """
    duplicates = set([])
    for t in eventT:
        if onset:
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return out



def psth_modind(psth):
    # modulation in terms of spike rate

    psth = psth.astype(float)
    use = psth - np.mean(psth[0:800].copy())
    mod = np.max(np.abs(use[1000:1250]))

    return mod



def main():

    
    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    cat_cmap = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey'
    }

    basepath = '/home/niell_lab/Data/demo_video'

    print('Reading eyecam')
    eyevid = avi_to_arr(os.path.join(basepath,
                                     '070921_J553RT_control_Rig2_fm1_REYEdeinter.avi'),
                                     ds=1)

    print('Reading worldcam')
    model_data = fme.read_h5(os.path.join(basepath,
                                          'ModelData_dt016_rawWorldCam_2ds.h5'))
    worldvid = model_data['model_vid_sm_shift']
    worldvidT = model_data['model_t']

    print('Reading ephys')
    pickle_path = '/home/niell_lab/Data/freely_moving_ephys/batch_files/062022/hffm_062022_gt.pickle'
    hffm = pd.read_pickle()

    ephys = hffm[hffm['session']=='070921_J553RT_control_Rig2'][hffm['gazecluster']!='unresponsive']
    ephys = ephys.sort_values(by='FmLt_gazeshift_peakT',
                              axis=0,ascending=True).reset_index(drop=True)

    print('Load saccade times and timestamps')
    saccT = np.array(sorted(list(ephys.loc[0,'FmLt_gazeshift_left_saccTimes_dHead1'])
                  + list(ephys.loc[0,'FmLt_gazeshift_right_saccTimes_dHead1'])))

    eyevidT = ephys.loc[0,'FmLt_eyeT'].copy()
    spikeT = ephys['FmLt_spikeT'].copy().to_dict()
    dHead = ephys.loc[0,'FmLt_dHead'].copy()
    dGaze = ephys.loc[0,'FmLt_dGaze'].copy()

    klabels = ephys['gazecluster'].copy().to_numpy()

    print('Load saccade times and timestamps')
    saccT = np.array(sorted(list(ephys.loc[0,'FmLt_gazeshift_left_saccTimes_dHead1'])
            + list(ephys.loc[0,'FmLt_gazeshift_right_saccTimes_dHead1'])))
    
    t = ephys.loc[0, 'FmLt_gazeshift_left_saccTimes_dHead1']

    modinds = []
    for ind, row in ephys.iterrows():
        modinds.append(row['Fm_fr'])

    np.save('/home/niell_lab/Desktop/demo_fr.npy', modinds)

    saccthresh = { # deg/sec
        'head_moved': 60,
        'gaze_stationary': 120,
        'gaze_moved': 240
    }

    eyevidT_ = eyevidT.copy()[:-1]

    # Gaze shifts
    left_gazeshift_times = eyevidT_[(dHead > saccthresh['head_moved'])    \
                                    & (dGaze > saccthresh['gaze_moved'])]
    
    right_gazeshift_times = eyevidT_[(dHead < -saccthresh['head_moved'])   \
                                     & (dGaze < -saccthresh['gaze_moved'])]

    saccT_new = np.sort(np.concatenate([left_gazeshift_times, right_gazeshift_times]))

    saccT_new = drop_repeat_sacc(saccT_new)

    t_start = 13  # was 13
    tlen = 3 # was 3
    numFr = np.argmin(np.abs(worldvidT-(t_start+tlen))) - np.argmin(np.abs(worldvidT-t_start))

    mpl.use('agg')


    pb = ProgressBar(numFr)
    actor = pb.actor

    spikeT_r = ray.put(spikeT)
    saccT_r = ray.put(saccT_new)
    eyevid_r = ray.put(eyevid)
    eyevidT_r = ray.put(eyevidT)
    worldvid_r = ray.put(worldvid)
    worldvidT_r = ray.put(worldvidT)
    klabels_r = ray.put(klabels)

    cmap_v = plt.cm.viridis(np.linspace(0,1,100))

    sf_pref = ephys['sf_pref_cpd'].copy().to_numpy()

    tick_sz = 0.5
    fig, ax_raster = plt.subplots(1,1,figsize=(9,4), dpi=300)
    for i, sps in spikeT.items():
        sps = sps[(sps>t_start) * (sps<(t_start+tlen))]
        ax_raster.plot(sps, np.ones(len(sps))*i, '|', color='k', markersize=4)

    ax_raster.set_ylim([len(spikeT.keys())+1,-4])
    ax_raster.set_xlim([t_start, t_start+tlen])
    ax_raster.set_xticks(np.arange(t_start, t_start+tlen+tick_sz, tick_sz))
    ax_raster.set_xticklabels((np.arange(0, tlen+tick_sz, tick_sz)*1000).astype(int))
    ax_raster.set_ylabel('cells')
    ax_raster.set_xlabel('time (ms)')
    ax_raster.spines['right'].set_visible(False)
    ax_raster.spines['top'].set_visible(False)
    show_saccT = saccT[(saccT>t_start) * (saccT<(t_start+tlen))]

    ax_raster.plot(show_saccT, np.ones(len(show_saccT)) * -3,
                   '.', color='k', marker='v', markersize=7)
    ax_raster.set_yticks(np.arange(0, 120, 20))

    fig.savefig('/home/niell_lab/Desktop/fig2_example_saccades_13_3_black.pdf')

    # Loop over parameters appending process ids
    result_ids = []
    for f in range(numFr):
        startFr = np.argmin(np.abs(worldvidT-t_start))
        currentT = worldvidT[startFr+f]
        
        result_ids.append(plot_frame_img.remote(currentT, t_start, tlen, spikeT_r, saccT_r,
                    eyevid_r, eyevidT_r, worldvid_r, worldvidT_r, klabels_r, actor))
        
    # Progressbar and get results
    pb.print_until_done()
    results_p = ray.get(result_ids)
    images = np.stack([results_p[i] for i in range(len(results_p))])

    # Make video with opencv
    savepath = '/home/niell_lab/Desktop/raster_animation_13_3_cmap_fixfont.mp4'
    FPS = (3/np.size(images,0))*1000 # 60 or 15 for slowed down x4
    out = cv2.VideoWriter(savepath,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          FPS,
                          (images.shape[-2],
                           images.shape[-3]))

    for f in range(np.size(images,0)):
        out.write(cv2.cvtColor(images[f],
                               cv2.COLOR_BGR2RGB))
        
    out.release()



def merge_video_audio():

    video_path = '/home/niell_lab/Desktop/first_2min_x4.mp4'
    audio_path = '/home/niell_lab/Data/demo_video/phil_test4.wav'
    audvid_path = '/home/niell_lab/Data/demo_video/first_2min_x4_merge.mp4'

    subprocess.call(['ffmpeg', '-i', video_path, '-i', audio_path,
                     '-c:v', 'copy', '-c:a', 'aac', '-y', audvid_path])
