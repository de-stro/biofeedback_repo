"""
    TODO Description
"""
import os

from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG

# TODO Clean this up
#filepaths = []
path_dir_01 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_01_2024-02-01_19-31-18'
name_file_01 = 'BrainFlow-RAW_2024-02-01_19-31-18_0.csv'
#filepaths.append(os.path.join(path_dir_01, name_file_01))
path_dir_02 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_02_2024-02-02_12-11-33'
name_file_02 = 'BrainFlow-RAW_2024-02-02_12-11-33_0.csv'
#filepaths.append(os.path.join(path_dir_02, name_file_02))
path_dir_03 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_03_2024-02-03_18-07-41'
name_file_03 = 'BrainFlow-RAW_2024-02-03_18-07-41_0.csv'
#filepaths.append(os.path.join(path_dir_03, name_file_03))
path_dir_04 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_04_2024-02-05_21-16-18'
name_file_04 = 'BrainFlow-RAW_2024-02-05_21-16-18_0.csv'
#filepaths.append(os.path.join(path_dir_04, name_file_04))
path_dir_05 = r'C:\\Users\\dennis\\Documents\\OpenBCI_GUI\\Recordings\\off_proband_05_2024-02-09_11-30-48'
name_file_05 = 'BrainFlow-RAW_2024-02-09_11-30-48_0.csv'
#filepaths.append(os.path.join(path_dir_05, name_file_05))

def getFilepaths():

    filepaths = []
    filepaths.append(os.path.join(path_dir_01, name_file_01))
    filepaths.append(os.path.join(path_dir_02, name_file_02))
    filepaths.append(os.path.join(path_dir_03, name_file_03))
    filepaths.append(os.path.join(path_dir_04, name_file_04))
    filepaths.append(os.path.join(path_dir_05, name_file_05))

    return filepaths

class RecordingData:
    def __init__(self, filepath):
        self.filepath = filepath
        # load OpenBCI recording (CSV-file in BrainFlow Format)
        self.loaded_data =  DataFilter.read_file(filepath)
        self.eeg_data = self.loaded_data[eeg_channels, :]
        self.ecg_data = self.loaded_data[ecg_channel, :]

    def getECG(self):
         # BrainFlow returns uV!!
        return self.ecg_data
    
    def getEEG(self):
         # BrainFlow returns uV!!
        return self.eeg_data
