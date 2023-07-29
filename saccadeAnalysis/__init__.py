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
    make_category_col,
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
    get_norm_FmLt_PSTHs,
    get_norm_FmDk_PSTHs,
    get_norm_Hf_PSTHs,
    FmLtDk_peak_time,
    drop_if_missing
)

from .utils.gratings import (
    gratings_tuning,
    gratings_responsive
)

from .utils.make_HfFm import make_hffm_dataset

# Isolate SBCs.
# from .utils.suppressed_by_contrast import isolate_SBCs


# Figure-generating functions.
from .nn_figs.fig1 import fig1
from .nn_figs.fig2 import fig2
from .nn_figs.fig3 import fig3
from .nn_figs.fig4 import fig4
from .nn_figs.fig5 import fig5
from .nn_figs.fig6 import fig6
# from .nn_figs.fig7 import fig7

from .nn_figs.figS1 import figS1
from .nn_figs.figS2 import figS2
from .nn_figs.figS3 import figS3
from .nn_figs.figS4 import figS4
# from .nn_figs.figS5 import figS5
# from .nn_figs.figS6 import figS6
# from .nn_figs.figS7 import figS7


