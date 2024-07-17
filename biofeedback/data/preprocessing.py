"""
    Module description
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations

import time


board_id_cyton = BoardIds.CYTON_BOARD
sampling_rate_cyton = BoardShim.get_sampling_rate(board_id_cyton)



def preprocess_data(data_2D):

    """preprocessData internal utility function for processing of BrainFlow output (as 2D ndarray data)

    Args:
        data_2D (_type_): _description_

    Returns:
        _type_: _description_
    """

    # perform filtering for each row (= channel) of 2D data array
    for channel in range(data_2D.shape[0]):
        # filters work in-place
        prepo_time = time.time()

        
        DataFilter.detrend(data_2D[channel], DetrendOperations.CONSTANT.value)
        # ICA requires data to be highpass filtered (cutoff at 1.0 Hz, see MNE)
        # DataFilter.perform_highpass(data_2D[channel], sampling_rate_cyton, 1.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandpass(data_2D[channel], sampling_rate_cyton, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        # perform line noise filtering (see samples in BrainFlow docs)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate_cyton, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate_cyton, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        prepo_time = time.time() - prepo_time
    
    return (data_2D, prepo_time)