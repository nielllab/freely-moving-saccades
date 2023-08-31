

import os
import PySimpleGUI as sg

sg.theme('Default1')

import fmEphys as fme
import saccadeAnalysis as sacc


def sessionSummary(data_filepath=None):

    if data_filepath is None:
        data_filepath = sg.popup_get_file('Select group h5 file',
                                          keep_on_top=True,
                                          file_types=(('H5 files', '*.h5'),),
                                            no_window=True)

    data = fme.read_group_h5(data_filepath)

    savepath = os.path.split(data_filepath)[0]

    sacc.summarize_sessions(data)


if __name__ == '__main__':

    sessionSummary()