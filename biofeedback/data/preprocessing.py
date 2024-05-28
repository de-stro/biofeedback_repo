"""
    TODO Description
"""

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.data_filter import FilterTypes
from brainflow.data_filter import DetrendOperations


board_id_cyton = BoardIds.CYTON_BOARD
sampling_rate_cyton = BoardShim.get_sampling_rate(board_id_cyton)





# TODO PreProcess Parametrisierung von außen (aka durch F-Eval Script) erlauben (?)
# TODO Check BRainFLow Daten Format und Nötiges Daten Format für Preprocessing ob hier noch transposed werden muss oder nicht!!
def preprocess_data(data_2D):
    """preprocessData interne hilfsfunktion für BrainFlow processing von 2D numpy array data (according to BrainFlow output)

    Args:
        data_2D (_type_): _description_

    Returns:
        _type_: _description_
    """

    # perform filtering for each row (= channel) of 2D data array
    for channel in range(data_2D.shape[0]):
        # filters work in-place
        
        # TODO WELCHES FILTERING HIER SINNVOLL FÜR ANWENDUNGSFALL (ICA for ECG from EEG)
        # (+ Quellen!!)
        DataFilter.detrend(data_2D[channel], DetrendOperations.CONSTANT.value)
        # TODO ICA requires data to be highpass filtered (cutoff at 1.0 Hz laut MNE) -> alternativ über MNE filtern?
        # DataFilter.perform_highpass(data_2D[channel], sampling_rate_cyton, 1.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandpass(data_2D[channel], sampling_rate_cyton, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate_cyton, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data_2D[channel], sampling_rate_cyton, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    
    return data_2D