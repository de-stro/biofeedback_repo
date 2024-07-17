"""
    Module description
"""

import numpy as np

def ceildiv(a, b):
    """ceildiv performs ceiling division (a / b rounded up)

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    return -(a // -b)




def naive_peak_value_surround_checks(signal, peak_sample_locations, sample_check_range:int):
    """naive_peak_value_surround_checks checks if values of peak samples are highest of surrounding samples (+/- 10 samples, equals 40 millis with
    sampling rate of 250 Hz), i.e. check if each sample value around detected peak (+/- 10 samples) is highest, 
    correct otherwise

    Args:
        signal (_type_): _description_
        peak_sample_locations (_type_): _description_
        sample_check_range (_type_): _description_

    Returns:
        _type_: _description_
    """

    corrected_peak_locs = []

    for idx, peak in enumerate(peak_sample_locations):
        check_range = list(range(peak-sample_check_range, peak+sample_check_range))
        peak_height = signal[peak]
        checked_peak = peak
        for checkSample in check_range:
            # assure that checking samples in check range stops at signal end
            if checkSample < signal.size:
                if signal[checkSample] > peak_height:
                    #print("ERROR!: peak at ", peak, "-th sample (", signal[peak], ") is lower than value (", signal[checkSample], ") of ", checkSample, "-th sample")
                    #print("MANUAL ARTIFCAT CORRECTION WAS EMPLOYED")
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


def remove_NaN_padding(values):
    """remove_NaN_padding removes NaN padding introduced by NeuroKit2

    Args:
        values (_type_): _description_

    Returns:
        _type_: _description_
    """

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


def prune_onsets_and_offsets(onsets_R_list, offsets_R_list):
    """ prune_onsets_and_offsets prunes on and offsets for two np.arrays 

    """

    # ensure that epochs beginn with first R_Offset (hence, a noise_segment) ["begin condition"] and end with last R_Offset ["end condition"],
    # therefore, last detected QRS complex is included, while first QRS complex might be omitted!
    # remove NaN occurences to check ["begin condition"]
    # NOTE Invariant: 1) both onsets and offsets have same length; 2) after onset (time point) comes an offset and the other way around 
    # NOTE "Padding Premise": 1) For each list, only the first and the last element can be NaN (at most)
    
    """TEST: assert that both onsets and offsets have same length """
    assert onsets_R_list.size == offsets_R_list.size, "onsets_R_list and offsets_R_list have not same length!"
    """TEST: assert that only 2 NaN values in array at max (back and front)"""
    assert np.count_nonzero(np.isnan(onsets_R_list)) <= 2
    assert np.count_nonzero(np.isnan(offsets_R_list)) <= 2
    """TEST: assert if first or last element is exclusively NaN (XOR), an element from the other array has to follow """
    if np.isnan(onsets_R_list[0]) and (not np.isnan(offsets_R_list[0])):
        assert offsets_R_list[0] < onsets_R_list[1]
    if np.isnan(offsets_R_list[0]) and (not np.isnan(onsets_R_list[0])):
        assert onsets_R_list[0] < offsets_R_list[1]
    
    if np.isnan(onsets_R_list[-1]) and (not np.isnan(offsets_R_list[-1])):
        assert offsets_R_list[-1] > onsets_R_list[1]
    if np.isnan(offsets_R_list[-1]) and (not np.isnan(onsets_R_list[-1])):
        assert onsets_R_list[0] < offsets_R_list[1]
    
    """FOR TEST PURPOSES ONLY
    TEST:assert that ON-OFF-ON-OFF sequence exists
    pruned_onsets = onsets_R_list[~np.isnan(onsets_R_list)]
    pruned_offsets = offsets_R_list[~np.isnan(offsets_R_list)]
    if pruned_onsets[0] > pruned_offsets[0]:
        for idx, offset in enumerate(pruned_offsets):
            assert pruned_onsets[idx] > pruned_offsets[idx]
    else:
        for idx, onset in enumerate(pruned_onsets):
            assert pruned_offsets[idx] > pruned_onsets[idx]
    """
    
    # clean if both onsets and offsets have NaN entry at end or beginning
    if np.isnan(onsets_R_list[0]) and np.isnan(offsets_R_list[0]):
        onsets_R_list = onsets_R_list[1:]
        offsets_R_list = offsets_R_list[1:]
    if np.isnan(onsets_R_list[-1]) and np.isnan(offsets_R_list[-1]):
        onsets_R_list = onsets_R_list[:-1]
        offsets_R_list = offsets_R_list[:-1]
    
    # ensure that last element is offset (with a preceding onset!)
    if np.isnan(onsets_R_list[-1]) or (onsets_R_list[-1] > offsets_R_list[-1]):
        onsets_R_list = onsets_R_list[:-1]  
        # assert new invariant: offset.size = onset.size + 1
        # assert invariant 2)
        # assert onsets_R_list.size + 1 == offsets_R_list.size, "offsets_R_list is not 1 greater than onsets_R_list!"
        assert (onsets_R_list[-1] < offsets_R_list[-1])
        # check for preceding onset, otherwise also omit the second last offset
        if offsets_R_list[-2] > onsets_R_list[-1]:
            offsets_R_list = offsets_R_list[:-1]
    
    assert (onsets_R_list[-1] < offsets_R_list[-1])
    assert offsets_R_list[-2] < onsets_R_list[-1]
    assert (onsets_R_list.size + 1 == offsets_R_list.size) or (onsets_R_list.size == offsets_R_list.size), "offsets_R_list is not equal or 1 greater than onsets_R_list!"


    # ensure that first element is also offset (followed by an onset)
    if np.isnan(onsets_R_list[0]) or ((onsets_R_list[0] < offsets_R_list[0])):
        onsets_R_list = onsets_R_list[1:]

        assert (onsets_R_list[0] > offsets_R_list[0])
        assert (onsets_R_list[0] < offsets_R_list[1])
    
    # check size invariant 
    assert (onsets_R_list.size + 1 == offsets_R_list.size)
      

    """TEST:assert that ON-OFF-ON-OFF sequence exists"""
    if onsets_R_list[0] > offsets_R_list[0]:
        for idx, onset in enumerate(onsets_R_list):
            assert onsets_R_list[idx] > offsets_R_list[idx]
    else:
        for idx, onset in enumerate(onsets_R_list):
            assert offsets_R_list[idx] > onsets_R_list[idx]

    return (onsets_R_list, offsets_R_list)