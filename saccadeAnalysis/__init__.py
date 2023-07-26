"""


Written by DMM, 2022
"""



# Create dataset
from .utils.create_dataset import (
    add_stimuli_horizontally,
    add_sessions_vertically,
    stack_dataset
)

from .utils.response_props import (
    calc_PSTH_modind,
    norm_PSTH,
    calc_PSTH_DSI,
    calc_PSTH_latency,
    get_direction_pref,
    norm_grat_histPSTH,
    calc_grat_histPSTH_modind
)

# Generic figures
from .utils.figs import (
    plot_tuning,
    plot_columns,
    plot_PSTH_heatmap,
    plot_regression,
    plot_running_median
)

from .utils.gazeshift_clusters import (
    make_cluster_model_input,
    make_clusters,
    add_labels_to_dataset,
    apply_saved_cluster_models
)




######


from .utils.auxillary import (
    z_score,
    stderr,
    drop_nan_along
)



from .utils.plot_helpers import (
    set_plt_params,
    make_colors
)

from .utils.psth import (
    
)

from .utils.psth import (
    calc_PSTH_latency,
    calc_PSTH_modind,
    calc_PSTH_DS,
    calc_PSTH_DSI,
    norm_PSTH,
    calc_KDE_PSTH
)

from .utils.marmoset_figs import (
    spike_raster,
)