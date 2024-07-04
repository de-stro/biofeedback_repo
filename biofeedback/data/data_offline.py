"""
    TODO Description
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations

import data.preprocessing as prepro

board_id_cyton = BoardIds.CYTON_BOARD
sampling_rate_cyton = BoardShim.get_sampling_rate(board_id_cyton)

# indices for board channels of interest
package_num_channel_cyton = BoardShim.get_package_num_channel(board_id_cyton)
timestamp_channel_cyton = BoardShim.get_timestamp_channel(board_id_cyton)
eeg_channels_cyton = BoardShim.get_eeg_channels(board_id_cyton)[:-1] # drop last channel (as we use it for ECG)
ecg_channel_cyton = BoardShim.get_ecg_channels(board_id_cyton).pop() # we use last channel (8th) for ECG! TODO WO ANDERWEITIG VERMERKT?!?!?

channels_of_interest = [package_num_channel_cyton, timestamp_channel_cyton, ecg_channel_cyton] + eeg_channels_cyton
channels_of_data_only = [ecg_channel_cyton] + eeg_channels_cyton

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
    """get_offline_recording_sessions loads all recording sessions into memory as RecordingSession objects
    (these encapsule each recording together with their meta data) and acts as an interface, therefore hiding 
    BrainFlow logic / data format away

    Returns:
        list: list of RecordingSession Objects that encapsule data and meta data of a recording session
    """    
    
    filepaths = []
    filepaths.append((session_id_01, os.path.join(path_dir_01, name_file_01)))
    filepaths.append((session_id_02, os.path.join(path_dir_02, name_file_02)))
    filepaths.append((session_id_03, os.path.join(path_dir_03, name_file_03)))
    filepaths.append((session_id_04, os.path.join(path_dir_04, name_file_04)))
    filepaths.append((session_id_05, os.path.join(path_dir_05, name_file_05)))

    offline_recordings = []

    # NOTE
    # Geeignete Records: Proband 02 (Dennis)
    #for (id, path) in filepaths:
    #    offline_recordings.append(RecordingSession(filepath=path, session_id=id, sampling_rate=sampling_rate_cyton))

    # FOR TEST PURPOSES: Laden aller Recordings aussparen for now! (nur dennis record laden)
    path = (filepaths[1])[1]
    id = (filepaths[1])[0]
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

        """ TODO OMIT THIS
        # first channel is package numbers, second timestamp values, third ECG reference signal and 
        # remaining channels are EEG signal / timeseries data
        # channels_of_interest = [package_num_channel_cyton, timestamp_channel_cyton, ecg_channel_cyton] + eeg_channels_cyton
        # channels_of_data_only = [ecg_channel_cyton] + eeg_channels_cyton

        
        # 2D data maxtrix with channels of interest only, with specified cutoff begin/end of recording applied
        # (rows are channels, columns are values / the actual data packages)
        #self.return_data_only = self.loaded_data[channels_of_interest, session_begin_cutoff:-session_end_cutoff]
        #self.preprocessed_data = prepro.preprocess_data(data_only)

        self.eeg_data = self.loaded_data[eeg_channels_cyton, session_begin_cutoff:-session_end_cutoff]
        self.ecg_data = self.loaded_data[ecg_channel_cyton, session_begin_cutoff:-session_end_cutoff]
        self.timestamps = self.loaded_data[timestamp_channel_cyton, session_begin_cutoff:-session_end_cutoff]
   

        # TODO ist das so sinnvoller??
        data_channels_cyton = [ecg_channel_cyton] + eeg_channels_cyton
        self.return_data_only = self.loaded_data[data_channels_cyton, session_begin_cutoff:-session_end_cutoff]

        # TODO Check BrainFLow Format ob hier noch transposed werden muss oder nicht!!! Stutze Array auf 1D??

    
        """
    
    
    
    def get_data(self):
        """get_data _summary_ # TODO

        Returns:
            _type_: _description_
        """

        data_matrix = self.loaded_data[channels_of_interest, session_begin_cutoff:-session_end_cutoff]
        return np.array(data_matrix)
    
    
    def get_preprocessed_data(self):
        """get_preprocessed_data _summary_ # TODO

        Returns:
            _type_: _description_
        """

        # preprocess data (ECG and EEG channels only) first
        data_only = self.loaded_data[channels_of_data_only, :]
        # TODO ein und ausschwingen des filters rausschneiden
        # TODO apply cutoff after preprocessing (to get rid of irregularities at very beginning of the processed signal)
        (prepo_results, prepo_time) = prepro.preprocess_data(data_only)
        prepro_data = np.array(prepo_results[:, session_begin_cutoff:-session_end_cutoff])

        # append channels for package numbers and timestamp values 
        package_num_and_timestamps = self.loaded_data[[package_num_channel_cyton, timestamp_channel_cyton], session_begin_cutoff:-session_end_cutoff]
        data_matrix = np.vstack((np.array(package_num_and_timestamps), prepro_data))

        return (data_matrix, prepo_time)
    







    # TODO OMIT REST OF Functions
    #def get_ECG_data(self):
        """get_ECG_data returns ECG timeseries data as 1D numpy array (without timestamps!). Values are in uV!
        
        Returns:
            ndarray[Any, dtype[float64]]: ECG timeseries data as 1D numpy array (values in uV)
        """

        # BrainFlow returns uV!!
        #ecg_data = self.loaded_data[ecg_channel_cyton, session_begin_cutoff:-session_end_cutoff]
        #return np.array(ecg_data)
    
    # TODO Check BrainFLow Format ob hier noch transposed werden muss oder nicht!!! stutze Array auf 2D??
    #def get_EEG_data(self):
        """get_EEG_data returns EEG timeseries data as 2D numpy array (without timestamps!). Values are in uV!
        
        Returns:
            ndarray[Any, dtype[float64]]: EEG timeseries data as 2D numpy array (rows are channels and columns are values in uV)
        """
         # BrainFlow returns uV!!
        #eeg_data = self.loaded_data[eeg_channels_cyton, session_begin_cutoff:-session_end_cutoff]
        #return np.array(eeg_data)
    
    # TODO Check Format und stutze auf 1D Array
    #def get_preprocessed_ECG_data(self):
        """get_preprocessed_ECG_data returns preprocessed ECG timeseries data as 1D numpy array (without timestamps!). Values are in uV!
        Preprocessing is provided by BrainFlow
        
        Returns:
            ndarray[Any, dtype[float64]]: Preprocessed ECG timeseries data as 1D numpy array (values in uV)
        """
        #prepro_data = prepro.preprocess_data(self.return_data_only)
        #prepro_ecg = prepro_data[ecg_channel_cyton, session_begin_cutoff:-session_end_cutoff]
        #return np.array(prepro.preprocess_data([new_ecg_data]))

    # TODO Check Format und stzte auf 2D Array
    #def get_preprocessed_EEG_data(self):
        """get_preprocessed_EEG_data returns preprocessed EEG timeseries data as 2D numpy array (without timestamps!). Values are in uV!
        Preprocessing is provided by BrainFlow
        
        Returns:
            ndarray[Any, dtype[float64]]: Preprocessed EEG timeseries data as 2D numpy array (rows are channels and columns are values in uV)
        """
        #return np.array(prepro.preprocess_data(self.eeg_data))
    
    
    # TODO OMIT THIS
    # TODO macht das Unterschied ob ich EEG UND ECG oder nur EEG preprocesse etc??
    #def get_preprocessed_data(self):
        """get_preprocessed_data first channel ECG ref signal, rest is EEG
        already cut-off as specified in script 

        Returns:
            _type_: _description_
        """
        #return np.array(prepro.preprocess_data(self.return_data_only))
    
    # TODO BRAUCHEN WIR HIER AUCH NOCH TIME STAMPS?
    
    #def get_timestamps(self):
        """get_timestamps provides access to data of boards timestamp channel. Board uses UNIX timestamps in seconds,
        this counter starts at the Unix Epoch on January 1st, 1970 at UTC (microsecond precision)

        Returns:
            ndarray[Any, dtype[float64]]: Timestamp values as 1D numpy array (values in seconds)
        """
        #return np.array(self.timestamps)
    
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