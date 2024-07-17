import numpy as np
import scipy.signal as scisig
import sklearn.preprocessing as sklprep
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import data.data_offline as offD
import data.preprocessing as prepro

import ica as icascript
import utility as util

import traceback

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
componentAmountConsidered = 7


def segment_peaks_equidistant_with_ref_peaks(ref_peaks_array, sig_peaks_array):
    """segment_peaks_equidistant_with_ref_peaks segments a signal array in segments with boundries
    equidistiant to reference peaks

    Args:
        ref_peaks_array (_type_): _description_
        sig_peaks_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    segments = []
    ref_peaks_pos = np.nonzero(ref_peaks_array)[0]
    last_end_idx = 0

    # get indexes (equidistant to reference peaks) for segmentation
    for iter_idx, peak_pos in enumerate(ref_peaks_pos):
        segment_begin_idx = last_end_idx

        # check if last peak/segment
        if peak_pos == ref_peaks_pos[-1]:
            segment_end_idx = sig_peaks_array.size - 1     # segment end is last index
        else:
            # segment ends halfway to next reference position/peak
            segment_end_idx = peak_pos + util.ceildiv(ref_peaks_pos[iter_idx + 1] - peak_pos, 2)
        last_end_idx = segment_end_idx

        segment_tuple = (ref_peaks_array[segment_begin_idx:segment_end_idx], sig_peaks_array[segment_begin_idx:segment_end_idx])
        segments.append(segment_tuple)

    return segments


def calculate_metrics_with_reference(ref_peaks, sig_peaks):
    """ calculate_metrics_with_reference calculates the peak detection errors and performance metrics using 
        the reference ECG
    """
    peak_segments = segment_peaks_equidistant_with_ref_peaks(ref_peaks, sig_peaks)

    true_hits = 0
    false_misses = 0
    false_positives = 0
    # displacements with closest distance to reference peak (in samples)
    sample_displacements = []

    for (ref_segment, signal_segment) in peak_segments:


        # ASSERT ref_segment contains exactly one peak (1) and rest is zeros!)
        assert np.count_nonzero(ref_segment) == 1, "ERROR, REF_SEGMENT contains not single peak"

        # check if *truly* a miss / false negative (no peak in signal segment whatsoever) 
        if np.count_nonzero(signal_segment) == 0:
            false_misses += 1
            continue

        # check for presence of *truly* false positives in signal segment (more than one peak)
        if np.count_nonzero(signal_segment) > 1:
            false_positives += (np.count_nonzero(signal_segment) - 1)

        # check if hit (matching peak in both segements)
        if (np.count_nonzero(np.logical_and(ref_segment, signal_segment)) == 1):
            true_hits += 1
            continue

        # at this point there must be a displacement of signal peak to reference peak
        # identify signal peak closest to reference peak (absolute distance!)
        abs_distances_to_ref_peak = np.abs(np.subtract((np.nonzero(ref_segment)[0]), (np.nonzero(signal_segment)[0])))
        # displacement as distance in samples
        min_distance = np.min(abs_distances_to_ref_peak)
        sample_displacements.append(min_distance)

    # assert that ref peaks amount is equal to true hits + true misses + displacements
    assert np.count_nonzero(ref_peaks) == (true_hits + false_misses + len(sample_displacements))

    # convert sample displacements into temporal jitter values in ms 
    millis_jitter_values = list(map(lambda sample_dist: sample_dist / sampling_rate, sample_displacements))

    # add true hits (that have 0 ms jitter) for calc of mean jitter
    millis_jitter_values = millis_jitter_values + ([0] *  true_hits)
                                                   
    # calculate average over all jitter values (including peaks with zero jitter)
    avg_jitter_millis = np.mean(millis_jitter_values)

    # calculate jitter score according to Porr et al. (2024) over all the segments 
    jitter_score = 1 / (1 + (avg_jitter_millis) / 12)

    # calculate the F_1 score over all segments according to  Porr et al. (2024)
    #assert (true_hits + false_positives + false_misses) != 0
    assert (true_hits + len(sample_displacements) + false_positives + false_misses) != 0
    
    amount_true_positives = true_hits + len(sample_displacements)
    try:
        f_1_score = (2 * amount_true_positives) / ((2 * amount_true_positives) + false_positives + false_misses)
    except:
        error_msg = traceback.format_exc()
        f_1_score = 0
        print("Calculating F_1 Score, an Error occured!")
        print(error_msg)

    # calculate the JF-score (in %) according to  Porr et al. (2024)
    jf_score = f_1_score * jitter_score * 100
    

    # calculate further metrics 
    # sensitivity (in %) here regards only (possibly displaced) hits and 
    # misses (with dynamic tolerance window of half distance to next beat)
    try:
        sensitivity = (amount_true_positives / (amount_true_positives + false_misses)) * 100
    except:
        sensitivity = 0
        error_msg = traceback.format_exc()
        print("Calculating Sensitivity, an Error occured!")
        print(error_msg)
    

    results_dict = {
        "true_positives": true_hits,
        "false_negatives": false_misses,
        "false_positives": false_positives,
        "peak_displacements_sample" : sample_displacements,
        #"amount_displacements": len(sample_displacements),
        #"mean_displacement_sample": np.mean(sample_displacements),
        #"std_displacement_sample": np.std(sample_displacements),
        #"avg_jitter_value_millis": avg_jitter_millis,
        "jitter_score": jitter_score,
        "f_1_score": f_1_score,
        "jf_score": jf_score,
        "sensitivity": sensitivity,
        "all_hits": amount_true_positives
    }

    return results_dict




def evaluate_all_peak_detect_methods_on_component(ecg_ic, ecg_sync_ref):
    """ evaluate_all_peak_detect_methods_on_component profiles all R-peak detection methods of NeuroKit2
    on the observation segment
    """

    # first peak correction by neurokit
    neurokit_peak_correction = True
    # afterwards naive peak sample value check / correction
    naive_peak_sample_correction = True

    ################ Parameterize and Profile NeuroKit2. Peak-Detection Algorithms ################################
    # setup dicts for each neurokit peak-detection algorithm (include version with cleaned input if provided by NeuroKit2)
    # (w/ and w/o cleaned data, if applicable)
    peakDetect_dicts = [
        {"method_id": "dirty_neurokit", "method": "neurokit", "clean": False},
        {"method_id": "clean_neurokit", "method": "neurokit", "clean": True},
        {"method_id": "dirty_pantompkins1985", "method": "pantompkins1985", "clean": False},
        {"method_id": "clean_pantompkins1985", "method": "pantompkins1985", "clean": True},
        {"method_id": "dirty_hamilton2002", "method": "hamilton2002", "clean": False},
        {"method_id": "clean_hamilton2002", "method": "hamilton2002", "clean": True},
        # exclude because buggy
        #{"method_id": "dirty_gamboa2008", "method": "gamboa2008", "clean": False},
        #{"method_id": "clean_gamboa2008", "method": "gamboa2008", "clean": True},
        {"method_id": "dirty_elgendi2010", "method": "elgendi2010", "clean": False},
        {"method_id": "clean_elgendi2010", "method": "elgendi2010", "clean": True},
        {"method_id": "dirty_engzeemod2012", "method": "engzeemod2012", "clean": False},
        {"method_id": "clean_engzeemod2012" , "method": "engzeemod2012", "clean": True},
        {"method_id": "dirty_kalidas2017", "method": "kalidas2017", "clean": False},
        # NOTE: Only dummy for cleaning method of kalidas2017 provided by neurokit2, 
        # i.e not fully implemented yet
        #{"method_id": "clean_kalidas2017", "method": "kalidas2017", "clean": True},
        # # Exclude as error is raised: "NeuroKit error: ecg_findpeaks(): 'emrich2023' not implemented"
        #{"method_id": "dirty_emrich2023" , "method": "emrich2023", "clean": False},
        # Exclude as error is raised: "NeuroKit error: ecg_clean(): 'method' should be one of 'neurokit', 'biosppy', 
        # 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012', 'templateconvolution'."
        #{"method_id": "clean_emrich2023", "method": "emrich2023", "clean": True},

        # detection methods with no cleaning method provided yet (by neurokit2) or 
        # not forseen at all (by implementation authors)
        {"method_id": "zong2003", "method": "zong2003", "clean": False},
        # Exclude since persistently buggy
        #{"method_id": "martinez2004", "method": "martinez2004", "clean": False},
        {"method_id": "christov2004", "method": "christov2004", "clean": False},
        {"method_id": "nabian2018" , "method": "nabian2018", "clean": False},
        {"method_id": "rodrigues2021", "method": "rodrigues2021", "clean": False},
        # Exclude since persistently buggy
        #{"method_id": "manikandan2012", "method": "manikandan2012", "clean": False} 
        {"method_id": "promac", "method": "promac", "clean": False}
        ]

    # execute and profile neurokit peak-detection algorithms for REF ECG signal as ground truth

    ### Delineate/Extract R-peaks of ECG_REF for ground truth peak labeling
    # approach 1:   use respective peak-detection method on ecg_reference
    #               limitation: metric evaluation script only measures deterioration of peak detection method
    #               when using dirty signal (ecg_ic) instead of ECG refrence signal
    #
    # approach 2:   use respective peak-detection method on ecg_reference and employ artifact correction provided by neurokit2 
    #               (with artifact correction using the method of Lipponen & Tarvainen (2019) implemented by neurokit2)
    # further:      additionally perform "naive" peak sample checking, i.e if sample value is highest in area of 80ms around the peak 
    #               (+/- 40ms, so +/-10 samples at 250Hz) (since QRS duration in healthy adult lasts around 80-100ms

    for method_dict in peakDetect_dicts:
        """ FOR TEST PURPOSES ONLY """
        print("TEST: PEAK DETECT ON REF SIGNAL BY METHOD: ", method_dict["method_id"])

        if method_dict["clean"]:
            time_start = time.time()
            ref_signal = nk.ecg_clean(ecg_sync_ref, sampling_rate=sampling_rate, method=method_dict["method"])
        else:
            ref_signal = ecg_sync_ref 
            time_start = time.time()
        
        # perform peak detection (with artifact correction provided by neurokit2 if desired)
        ref_results, ref_info = nk.ecg_peaks(ref_signal, sampling_rate = sampling_rate, method = method_dict["method"], correct_artifacts = neurokit_peak_correction, show = False)
        time_total = time.time() - time_start

        # if desired, additionally perform "naive" peak sample checking / correction, by checking surrounding
        # samples for (higher) values
        if naive_peak_sample_correction:
            # fig_peaks_before = nk.events_plot(ref_info["ECG_R_Peaks"], ref_signal, color = "r")
            ref_info["ECG_R_Peaks"] = util.naive_peak_value_surround_checks(ref_signal, ref_info["ECG_R_Peaks"], 40)
            # fig_peaks_after = nk.events_plot(ref_info["ECG_R_Peaks"], ref_signal, color = "g")

        method_dict["gnd_truth_results"] = ref_results
        method_dict["gnd_truth_info"] = ref_info
        method_dict["gnd_truth_exec_time"] = time_total


    # execute and profile neurokit peak-detection algorithms on dirty signal for comparision
    # do not use any artifact / peak correction whatsoever (for comparability)
    for method_dict in peakDetect_dicts:
        """ FOR TEST PURPOSES ONLY """
        print("TESTEST: PEAK DETECT ON ECG_IC_SIGNAL BY METHOD: ", method_dict["method_id"])
        if method_dict["clean"]:
            time_start = time.time()
            ecg_signal = nk.ecg_clean(ecg_ic, sampling_rate=sampling_rate, method=method_dict["method"])
        else:
            ecg_signal = ecg_ic 
            time_start = time.time()

        sig_results, sig_info = nk.ecg_peaks(ecg_signal, sampling_rate = sampling_rate, method = method_dict["method"], correct_artifacts = False, show = False)
        time_total = time.time() - time_start
        
        method_dict["sig_results"] = sig_results
        method_dict["sig_info"] = sig_info
        method_dict["sig_exec_time"] = time_total


    # calculate quality metrics regarding comparision to ground truth
    for method_dict in peakDetect_dicts:
        """ FOR TEST PURPOSES ONLY """
        print("TESTEST: CALC METRICS BY METHOD: ", method_dict["method_id"])
        gnd_truth_beats = np.array((method_dict["gnd_truth_results"])["ECG_R_Peaks"])
        gnd_truth_beat_amount = np.count_nonzero(gnd_truth_beats)
        signal_beats = np.array((method_dict["sig_results"])["ECG_R_Peaks"])
        signal_beat_amount = np.count_nonzero(signal_beats)
        metrics_dict = calculate_metrics_with_reference(gnd_truth_beats, signal_beats)
      
        metrics_dict["compTime_sec"] =  method_dict["sig_exec_time"]
        
        # store quality metrics
        method_dict["metrics_dict"] = metrics_dict
        
    return peakDetect_dicts
    


