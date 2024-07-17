"""
    Module description
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter


import data.preprocessing as prepro
# starting from 0 (!)
PARTICIPANT_TO_CONSIDER = 0

board_id_cyton = BoardIds.CYTON_BOARD
sampling_rate_cyton = BoardShim.get_sampling_rate(board_id_cyton)

# indices for board channels of interest
package_num_channel_cyton = BoardShim.get_package_num_channel(board_id_cyton)
timestamp_channel_cyton = BoardShim.get_timestamp_channel(board_id_cyton)
eeg_channels_cyton = BoardShim.get_eeg_channels(board_id_cyton)[:-1] # drop last channel (as we use it for ECG)
ecg_channel_cyton = BoardShim.get_ecg_channels(board_id_cyton).pop() # NOTE we use last channel (8th) for ECG!

channels_of_interest = [package_num_channel_cyton, timestamp_channel_cyton, ecg_channel_cyton] + eeg_channels_cyton
channels_of_data_only = [ecg_channel_cyton] + eeg_channels_cyton

# cutoffs (in samples) to exclude recording setup/cool-down noise
session_begin_cutoff = 120 * sampling_rate_cyton 
session_end_cutoff = 10 * sampling_rate_cyton

curr_dir = os.path.dirname(__file__)

# recorded data
path_dir_01 = 'ATE_EEG_recordings\\off_proband_01_2024-02-02_12-11-33'
name_file_01 = 'BrainFlow-RAW_2024-02-02_12-11-33_0.csv'
session_id_01 = "off_proband_01"

path_dir_02 = 'ATE_EEG_recordings\off_proband_02_2024-02-05_21-16-18'
name_file_02 = 'BrainFlow-RAW_2024-02-05_21-16-18_0.csv'
session_id_02 = "off_proband_02"

path_dir_03 = 'ATE_EEG_recordings\off_proband_03_2024-02-01_19-31-18'
name_file_03 = 'BrainFlow-RAW_2024-02-01_19-31-18_0.csv'
session_id_03 = "off_proband_03"

path_dir_04 = 'ATE_EEG_recordings\off_proband_04_2024-02-03_18-07-41'
name_file_04 = 'BrainFlow-RAW_2024-02-03_18-07-41_0.csv'
session_id_04 = "off_proband_04"

path_dir_05 = 'ATE_EEG_recordings\off_proband_05_2024-02-09_11-30-48'
name_file_05 = 'BrainFlow-RAW_2024-02-09_11-30-48_0.csv'
session_id_05 = "off_proband_05"


def get_offline_recording_sessions():
    """get_offline_recording_sessions loads all recording sessions into memory as RecordingSession objects
    (these encapsule each recording together with their meta data) and acts as an interface, therefore hiding 
    BrainFlow logic / data format away

    Returns:
        list: list of RecordingSession Objects that encapsule data and meta data of a recording session
    """    
    
    filepaths = []
    filepaths.append((session_id_01, os.path.join(curr_dir, path_dir_01, name_file_01)))
    filepaths.append((session_id_02, os.path.join(curr_dir, path_dir_02, name_file_02)))
    filepaths.append((session_id_03, os.path.join(curr_dir, path_dir_03, name_file_03)))
    filepaths.append((session_id_04, os.path.join(curr_dir, path_dir_04, name_file_04)))
    filepaths.append((session_id_05, os.path.join(curr_dir, path_dir_05, name_file_05)))

    offline_recordings = []

    # NOTE
    # evaluated records were record 01 (Proband01_Good Case) and record 03 (Proband02_Bad Case)

    """
    for (id, path) in filepaths:
        offline_recordings.append(RecordingSession(filepath=path, session_id=id, sampling_rate=sampling_rate_cyton))
    """

    # only extract a single recording here instead
    path = (filepaths[PARTICIPANT_TO_CONSIDER])[1]
    id = (filepaths[PARTICIPANT_TO_CONSIDER])[0]
    offline_recordings.append(RecordingSession(filepath=path, session_id=id, sampling_rate=sampling_rate_cyton))

    return offline_recordings


class RecordingSession:
    """ RecordingSession encapsules recorded data and meta data of a recording session and acts as adapter 
    to BrainFlow data format. Data access is enabled via various getter methods. Timeseries are cutoff at 
    beginning and end of recording as specified.
    """
  
    def __init__(self, filepath, session_id, sampling_rate):
        self.filepath = filepath
        self.sampling_rate = sampling_rate
        self.session_id = session_id
        # load OpenBCI recording (CSV-file in BrainFlow Format)
        self.loaded_data =  DataFilter.read_file(filepath)

    
    
    def get_data(self):
        """get_data _summary_ 

        Returns:
            _type_: _description_
        """

        data_matrix = self.loaded_data[channels_of_interest, session_begin_cutoff:-session_end_cutoff]
        return np.array(data_matrix)
    
    
    def get_preprocessed_data(self):
        """get_preprocessed_data _summary_ 

        Returns:
            _type_: _description_
        """

        # preprocess data (ECG and EEG channels only) first
        data_only = self.loaded_data[channels_of_data_only, :]

        # apply cutoff after preprocessing (to get rid of irregularities at very beginning of the processed signal)
        # and remove transient reponse after filter application
        (prepo_results, prepo_time) = prepro.preprocess_data(data_only)
        prepro_data = np.array(prepo_results[:, session_begin_cutoff:-session_end_cutoff])

        # append channels for package numbers and timestamp values 
        package_num_and_timestamps = self.loaded_data[[package_num_channel_cyton, timestamp_channel_cyton], session_begin_cutoff:-session_end_cutoff]
        data_matrix = np.vstack((np.array(package_num_and_timestamps), prepro_data))

        return (data_matrix, prepo_time)
    


    def get_description(self):
        """get_description provides access to meta data of the recording session

        Returns:
            dict: dictionary of meta data (accessible with keys "session_id" and "sampling_rate")
        """

        descript = {
            "session_id" : self.session_id,
            "sampling_rate" : self.sampling_rate,
        }
        return descript





""" FOR TEST PURPOSES ONLY 
first_record_session = get_offline_recording_sessions()[0]
data = first_record_session.get_data()
prepro_data = first_record_session.get_preprocessed_data()
print("data shape: ", data.shape, " and dtype: ", data.dtype)
print("prepo data shape: ", prepro_data.shape, " and dtype: ", prepro_data.dtype)

# plot the data
fig, axs = plt.subplots(len(channels_of_interest) - 2, 1, sharex = True)

for j in range(2, len(channels_of_interest)):
    axs[j-2].plot(data[j, :])
    axs[j-2].set_title(str(j-2) + "-th channel")

plt.show()

# plot the preprocessed data
fig, axs = plt.subplots(len(channels_of_interest) - 2, 1, sharex = True)

for j in range(2, len(channels_of_interest)):
    axs[j-2].plot(prepro_data[j, :])
    axs[j-2].set_title(str(j-2) + "-th channel (preprocessed)")

plt.show()
"""