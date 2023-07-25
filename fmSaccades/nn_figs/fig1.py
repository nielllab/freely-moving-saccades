
"""
Demo data
for figure 1


"""

import numpy as np
import pandas as pd



def figure1(hffm):
    
    # Set up demo data
    demo = hffm[hffm['session']=='102621_J558NC_control_Rig2'].iloc[0]

    dHead_data = demo['FmLt_dHead']
    dEye_data = demo['FmLt_dEye_dps']
    dGaze_data = demo['FmLt_dGaze']
    eyeT = demo['FmLt_eyeT']

    left = demo['FmLt_gazeshift_left_saccTimes_dHead1']
    right = demo['FmLt_gazeshift_right_saccTimes_dHead1']

    comp = np.hstack([demo['FmLt_comp_left_saccTimes_dHead1'],
                      demo['FmLt_comp_right_saccTimes_dHead1']])

    plotinds = np.sort(np.random.choice(np.arange(eyeT.size),
                                        size=int(np.ceil(eyeT.size/25)),
                                        replace=False))
    
    gazemovs = np.hstack([left, right])

