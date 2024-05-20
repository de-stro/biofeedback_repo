"""
    TODO Description
"""
import os

from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations

import data.preprocessing as prepro

board_id_cyton = BoardIds.CYTON_BOARD
sampling_rate_cyton = BoardShim.get_sampling_rate(board_id_cyton)
timestamp_channel_cyton = [BoardShim.get_timestamp_channel(board_id_cyton)]
#package_num_channel_cyton = BoardShim.get_package_num_channel(board_id_cyton)
eeg_channels_cyton = BoardShim.get_eeg_channels(board_id_cyton)[:-1] # drop last channel (as it is ECG)
ecg_channel_cyton = [BoardShim.get_ecg_channels(board_id_cyton).pop()] # last channel is ECG

# cutoffs (in samples) to exclude recording setup/cool-down noise
# TODO determine these params visually!! (same for all recordings)
session_begin_cutoff = 120 * sampling_rate_cyton 
session_end_cutoff = 10 * sampling_rate_cyton


# TODO Move data into repository

path_dir_01 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_01_2024-02-01_19-31-18'
name_file_01 = 'BrainFlow-RAW_2024-02-01_19-31-18_0.csv'
session_id_01 = "off_proband_01"

path_dir_02 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_02_2024-02-02_12-11-33'
name_file_02 = 'BrainFlow-RAW_2024-02-02_12-11-33_0.csv'
session_id_02 = "off_proband_02"

path_dir_03 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_03_2024-02-03_18-07-41'
name_file_03 = 'BrainFlow-RAW_2024-02-03_18-07-41_0.csv'
session_id_03 = "off_proband_03"

path_dir_04 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_04_2024-02-05_21-16-18'
name_file_04 = 'BrainFlow-RAW_2024-02-05_21-16-18_0.csv'
session_id_04 = "off_proband_04"

path_dir_05 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_05_2024-02-09_11-30-48'
name_file_05 = 'BrainFlow-RAW_2024-02-09_11-30-48_0.csv'
session_id_05 = "off_proband_05"



# TODO wie steuert das Interface hier die parametrisierung der einzelnen Recording Objekte????
# TODO Data Segmentation und Cutoff ist Problem des F-Eval Scripts 
def get_offline_recording_sessions():
    """get_offline_recording_sessions provides interface to access all offline
    recording sessions (als interface für das ganze script)
    - kapselt brainflow logik (daten format und BF preprocessing) weg
    """
    
    filepaths = []
    filepaths.append((session_id_01, os.path.join(path_dir_01, name_file_01)))
    filepaths.append((session_id_02, os.path.join(path_dir_02, name_file_02)))
    filepaths.append((session_id_03, os.path.join(path_dir_03, name_file_03)))
    filepaths.append((session_id_04, os.path.join(path_dir_04, name_file_04)))
    filepaths.append((session_id_05, os.path.join(path_dir_05, name_file_05)))

    offline_recordings = []

    #for (id, path) in filepaths:
    #    offline_recordings.append(RecordingSession(filepath=path, session_id=id, sampling_rate=sampling_rate_cyton))
    
    # FOR TEST PURPOSES 
    path = (filepaths[1])[1]
    id = (filepaths[1])[0]
    offline_recordings.append(RecordingSession(filepath=path, session_id=id, sampling_rate=sampling_rate_cyton))

    return offline_recordings


class RecordingSession:
    """ kapselt BrainFlow weg (als interface zu den Daten/einzelnen Recording Sessions) und
    provided accesu zu preprocessed (brainflow filter) daten 
    """
    def __init__(self, filepath, session_id, sampling_rate):
        self.filepath = filepath
        # load OpenBCI recording (CSV-file in BrainFlow Format) and cutoff begin/end segment
        self.loaded_data =  DataFilter.read_file(filepath)
        self.eeg_data = self.loaded_data[eeg_channels_cyton, session_begin_cutoff:-session_end_cutoff]
        self.ecg_data = self.loaded_data[ecg_channel_cyton, session_begin_cutoff:-session_end_cutoff]
        self.timestamps = self.loaded_data[timestamp_channel_cyton, session_begin_cutoff:-session_end_cutoff]
        self.sampling_rate = sampling_rate
        self.session_id = session_id

        # TODO ist das so sinnvoller??
        data_channels_cyton = ecg_channel_cyton + eeg_channels_cyton
        self.data = self.loaded_data[data_channels_cyton, session_begin_cutoff:-session_end_cutoff]

    # TODO Check BrainFLow Format ob hier noch transposed werden muss oder nicht!!! Stutze Array auf 1D??
    def get_ECG_data(self):
        """get_ECG_data returns data 2D !!! (See BrainFLow for Format)
        """
         # BrainFlow returns uV!!
        return self.ecg_data
    
    # TODO Check BrainFLow Format ob hier noch transposed werden muss oder nicht!!! stutze Array auf 2D??
    def get_EEG_data(self):
        """get_EEG_data returns data 2D !!! (See BrainFLow for Format)
        """
         # BrainFlow returns uV!!
        return self.eeg_data
    
    # Check Format und stutze auf 1D Array
    def get_preprocessed_ECG_data(self):
        return prepro.preprocess_data(self.ecg_data)

    # Check Format und stzte auf 2D Array
    def get_preprocessed_EEG_data(self):
        return prepro.preprocess_data(self.eeg_data)
    
    # TODO macht das Unterschied ob ich EEG UND ECG oder nur EEG preprocesse etc??
    def get_preprocessed_data(self):
        return prepro.preprocess_data(self.data)
    
    # TODO BRAUCHEN WIR HIER AUCH NOCH TIME STAMPS?
    
    def get_description(self):
        descript = {
            "session_id" : self.session_id,
            "sampling_rate" : self.sampling_rate,
        }
        return descript


