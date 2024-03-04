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

from .utils.dark import make_ltdk_dataset


# Figure-generating functions.
from .nn_figs.fig1 import fig1
from .nn_figs.fig2 import fig2
from .nn_figs.fig3 import fig3
from .nn_figs.fig4 import fig4
from .nn_figs.fig5 import fig5
from .nn_figs.fig6 import fig6

from .nn_figs.figS1 import figS1
from .nn_figs.figS2 import figS2
from .nn_figs.figS3 import figS3
from .nn_figs.figS4 import figS4

from .utils.unitsumm_helpers import (
    tuning_modulation_index,
    saccade_modulation_index,
    waveform,
    tuning_curve,
    grat_stim_tuning,
    revchecker_laminar_depth,
    grat_psth,
    lfp_laminar_depth,
    sta,
    stv,
    movement_psth,
    is_empty_index,
    is_empty_cell
)

from .utils.unit_summary import (
    summarize_units
)

from .utils.session_summary import (
    summarize_sessions,
    get_animal_activity
)

from .sessionSummary import sessionSummary
from .unitSummary import unitSummary

from .utils.make_Marm import make_marm_dataset

from .utils.marm_helpers import (
    marm_psth_modind,
    mRaster,
    m_plot_tempseq,
    marm_normalize_psth
)

from .utils.minimal_cluster import apply_minimal_clustering


from .utils.make_HfFm_lim import make_hffm_dataset_onlyFmRc