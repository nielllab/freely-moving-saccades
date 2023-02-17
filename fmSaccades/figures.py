

import numpy as np
import matplotlib.pyplot as plt

def setup():
    colors = {
        'movement': np.array([0.994495, 0.74088, 0.166335, 1.]),
        'early':    np.array([0.95547, 0.533093, 0.28549, 1.]),
        'late':     np.array([0.85975, 0.360588, 0.406917, 1.]),
        'biphasic': np.array([0.640959, 0.116492, 0.602065, 1.]),
        'negative': np.array([0.32515, 0.006915, 0.639512, 1.]),
        'unresponsive': 'dimgrey',
        'gaze':         'firebrick',
        'comp':         'mediumblue',
        'rc':           'indigo'
    }

def fig1(sd):

    