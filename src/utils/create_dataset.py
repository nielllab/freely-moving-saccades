"""

"""

from tqdm import tqdm

import fmEphys.utils as utils

def add_datasets(path_list, existing=None):
    """

    """
    # If there isn't a file for the existing data, add the recordings into a
    # new, empty dictionary.
    if existing is None:
        existing = {}

    for path in path_list:

        new_data = utils.file.read_h5(path)
        use_key = new_data['rname']

        if use_key in existing.keys():
            use_key1 = use_key
            use_key = '{}_repl'.format(use_key)
            print('Recording name {} already in existing data. Adding as {}'.format(use_key1, use_key))
        existing[use_key] = new_data

    return existing

def 