import numpy as np

# function to perform ceiling division (a / b rounded up)
def ceildiv(a, b):
    return -(a // -b)



# func checks if values of peak samples are highest of surrounding samples (+/- 10 samples, equals 40 millis with
# sampling rate of 250 Hz), i.e. check if each sample value around detected peak (+/- 10 samples) is highest, 
# correct otherwise
# TODO signal and peak_sample_locations must be numpy arrays? 

def naive_peak_value_surround_checks(signal, peak_sample_locations, sample_check_range:int):

    corrected_peak_locs = []

    for idx, peak in enumerate(peak_sample_locations):
        check_range = list(range(peak-sample_check_range, peak+sample_check_range))
        peak_height = signal[peak]
        checked_peak = peak
        for checkSample in check_range:
            if signal[checkSample] > peak_height:
                print("ERROR!: peak at ", peak, "-th sample (", signal[peak], ") is lower than value (", signal[checkSample], ") of ", checkSample, "-th sample")
                print("MANUAL ARTIFCAT CORRECTION WAS EMPLOYED")
                # change peak sample location to new maxima found
                checked_peak = checkSample
                peak_height = signal[checked_peak]
        corrected_peak_locs.append(checked_peak)
    
    # convert output type to match input type
    if isinstance(peak_sample_locations, np.ndarray):
        corrected_peak_locs = np.array(corrected_peak_locs)
    else:
        assert isinstance(peak_sample_locations, list), "Error: input should either be of type ndarray or list!"
    
    return corrected_peak_locs


# TODO takes and outputs either list or ndarray
def remove_NaN_padding(values):

    values_without_NaN = []
    for element in values:
        if not (element != element):
            values_without_NaN.append(element)

    # convert output type to match input type
    if isinstance(values, np.ndarray):
        values_without_NaN = np.array(values_without_NaN)
    else:
        assert isinstance(values, list), "Error: input should either be of type ndarray or list!"

    return values_without_NaN