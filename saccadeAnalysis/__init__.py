"""
saccadeAnalysis

Written by DMM, 2022
"""

# Create dataset.
from .utils.create_dataset import (
    add_stimuli_horizontally,
    add_sessions_vertically,
    stack_dataset
)

# Quantify response properties.
from .utils.response_props import (
    calc_PSTH_modind,
    norm_PSTH,
    calc_PSTH_DSI,
    calc_PSTH_latency,
    get_direction_pref,
    norm_grat_histPSTH,
    calc_grat_histPSTH_modind
)

# Visualization helper functions
from .utils.plt_helpers import (
    propsdict,
    jitter,
    to_color,
    set_plt_params
)

# Generic figures
from .utils.figs import (
    plot_tuning,
    plot_columns,
    plot_PSTH_heatmap,
    plot_regression,
    plot_running_median
)

# Cluster analysis
from .utils.gazeshift_clusters import (
    make_cluster_model_input,
    make_clusters,
    add_labels_to_dataset,
    apply_saved_cluster_models
)

# Seperate excitatory and inhibitory cells (based on
# narrow versus broad spike waveform).
from .utils.spike_waveform import putative_cell_type


# Helper functions for mouse analysis.
from .utils.mouse_helpers import (

)

from .utils.make_HfFm import (


)

# Isolate SBCs.
from .utils.suppressed_by_contrast import isolate_SBCs


# Figure-generating functions.
from .nn_figs import (
    fig1,
    fig2,
    fig3,
    fig4,
    fig5,
    fig6,
    fig7,
    figS1,
    figS2,
    figS3,
    figS4,
    figS5,
    figS6,
    figS7
)

