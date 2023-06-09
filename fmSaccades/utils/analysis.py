
import tqdm as tqdm

import fmSaccades as fms

def normPSTH_all(dataset, psth_keys):

    for ind, row in tqdm(dataset.iterrows()):
        for key, newcol in psth_keys.items():
            if type(row[key]) != float:
                dataset.at[ind, newcol] = fms.norm_PSTH(row[key])