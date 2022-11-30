

import numpy as np
import matplotlib.pyplot as plt


def setup():
    # Create color maps
    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    cat_cmap = {
        'movement': plasma_map[12,:],
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey'
    }
    colors = {
        'gaze': 'firebrick',
        'comp': 'mediumblue',
        'rc': 'indigo'
    }