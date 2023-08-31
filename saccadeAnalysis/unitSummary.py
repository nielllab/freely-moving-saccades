

import os
import PySimpleGUI as sg

sg.theme('Default1')

import saccadeAnalysis as sacc


def unitSummary(data_filepath=None):

    if data_filepath is None:

        data_filepath = sg.popup_get_file('Select group h5 file',
                                          keep_on_top=True,
                                          file_types=(('H5 files', '*.h5'),),
                                            no_window=True)

    savepath = os.path.split(data_filepath)[0]

    sacc.summarize_units(data_filepath)


if __name__ == '__main__':

    unitSummary()