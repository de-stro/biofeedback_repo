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

# Overview of TODO
# TODO 1: Error Handling (weiterreichen, wo Verantwortlichkeiten?)
# TODO 2: Naive Peak Correction für Ground Truth ??
# TODO 3: Kaputte Neurokit Methoden checken (-> Neurokit2 import updaten?)
# TODO 4: Testfälle schreiben wo sinnvoll (insb segmenting und metric calculation)
# TODO 5: perform_peak_detection Service Methode implementieren
# TODO 6: Chose PD method based on quality metrics?

def segment_peaks_equidistant_with_ref_peaks(ref_peaks_array, sig_peaks_array):
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
    """calculate_metrics_with_reference calculates
    """
    # TODO Wie errors handeln? in welcher Funktion? Eher high level funktionen die
    # mit den konkreten methoden umgehen (hier wird angenommen alles passt)

    peak_segments = segment_peaks_equidistant_with_ref_peaks(ref_peaks, sig_peaks)

    true_hits = 0
    true_misses = 0
    true_false_positives = 0
    # displacements with closest distance to reference peak (in samples)
    sample_displacements = []

    for (ref_segment, signal_segment) in peak_segments:


        # TODO ASSERT ref_segment contains exactly one peak (1) and rest is zeros!)
        # TODO otherwise raise error or skip segment (???)
        assert np.count_nonzero(ref_segment) == 1, "ERROR, REF_SEGMENT contains not single peak"
        # if np.count_nonzero(ref_segment) != 1:
            # raise Exception("ERROR, REF_SEGMENT contains not single peak")

        # check if *true* miss (no peak in signal segment whatsoever) 
        if np.count_nonzero(signal_segment) == 0:
            true_misses += 1
            continue

        # check for presence of false positives in signal segment (more than one peak)
        if np.count_nonzero(signal_segment) > 1:
            true_false_positives += (np.count_nonzero(signal_segment) - 1)

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

    # convert sample displacements into temporal jitter values in ms 
    millis_jitter_values = list(map(lambda sample_dist: sample_dist / sampling_rate, sample_displacements))

    # calculate average over all jitter values 
    avg_jitter_millis = np.mean(millis_jitter_values)

    # calculate jitter score according to TODO PAPER Von BERND PORR over all the segments 
    jitter_score = 1 / (1 + (avg_jitter_millis) / 12)

    # calulate the F_1 score over all segments TODO according to BERN PORR PREPRINT
    try:
        f_1_score = (2 * true_hits) / ((2 * true_hits) + true_false_positives + true_misses)
    except:
        error_msg = traceback.format_exc()
        f_1_score = 0
        print("Calculating F_1 Score, an Error occured!")
        print(error_msg)

    # calculate the JF-score according to TODO PAPER BERN PORR 
    jf_score = f_1_score * jitter_score * 100

    # calculate further metrics 
    # sensitivity here regards only (possibly displaced) hits and 
    # misses (with dynamic tolerance window of half distance to next beat)
    try:
        sensitivity = true_hits / (true_hits + true_misses)
    except:
        sensitivity = 0
        error_msg = traceback.format_exc()
        print("Calculating Sensitivity, an Error occured!")
        print(error_msg)

    results_dict = {
        "true_positives": true_hits,
        "false_negatives": true_misses,
        "false_positives": true_false_positives,
        "amount_displacements": len(sample_displacements),
        "mean_displacement_sample": np.mean(sample_displacements),
        "std_displacement_sample": np.std(sample_displacements),
        "avg_jitter_value_millis": avg_jitter_millis,
        "jitter_score": jitter_score,
        "f_1_score": f_1_score,
        "jf_score": jf_score,
        "sensitivity": sensitivity
    }

    return results_dict



def evaluate_all_peak_detect_methods(ecg_ic, ecg_sync_ref):

    # TODO variablen auslagern in methoden rumpf
    # first peak correction by neurokit
    neurokit_peak_correction = True
    # afterwards naive peak sample value check / correction
    naive_peak_sample_correction = True

    ################ Parameterize and Profile NeuroKit2. Peak-Detection Algorithms ################################
    # setup dicts for each neurokit peak-detection algorithm (include version with cleaned input if provided by NeuroKit2)
    # TODO (w/ and w/o cleaned data, if applicable) -> macht das Sinn, das jeweils mit und ohne?
    # TODO Parameterize the methods independently??? -> geht über Scope hinaus
    # TODO Untersuche Neurokit Fehler bei Benutzung mancher Methoden
    peakDetect_dicts = {
        "dirty_neurokit" : {"method": "neurokit", "clean": False},
        "clean_neurokit" : {"method": "neurokit", "clean": True},
        "dirty_pantompkins1985" : {"method": "pantompkins1985", "clean": False},
        "clean_pantompkins1985" : {"method": "pantompkins1985", "clean": True},
        "dirty_hamilton2002" : {"method": "hamilton2002", "clean": False},
        "clean_hamilton2002" : {"method": "hamilton2002", "clean": True},
        # TODO gamboa2008 Fehler untersuchen (index out of bounds)
        #"dirty_gamboa2008" : {"method": "gamboa2008", "clean": False},
        #"clean_gamboa2008" : {"method": "gamboa2008", "clean": True},
        "dirty_elgendi2010" : {"method": "elgendi2010", "clean": False},
        "clean_elgendi2010" : {"method": "elgendi2010", "clean": True},
        "dirty_engzeemod2012" : {"method": "engzeemod2012", "clean": False},
        "clean_engzeemod2012" :  {"method": "engzeemod2012", "clean": True},
        "dirty_kalidas2017" : {"method": "kalidas2017", "clean": False},
        # TODO Remark: Only dummy for cleaning method of kalidas2017 provided by neurokit2, 
        # i.e not fully implemented yet (as of 16.05.2024)
        "clean_kalidas2017" : {"method": "kalidas2017", "clean": True},
        # TODO emrich2023 Fehler untersuchen (not implemented bzw. "vg" method (nur für clean?))
        # TODO UPDATE THE NEUROKIT IMPORT
        #"dirty_emrich2023" :  {"method": "emrich2023", "clean": False},
        #"clean_emrich2023" : {"method": "emrich2023", "clean": True},

        # detection methods with no cleaning method provided yet (by neurokit2) or 
        # not forseen at all (by implementation authors)
        "zong2003" : {"method": "zong2003", "clean": False},
        "martinez2004" : {"method": "martinez2004", "clean": False},
        "christov2004" : {"method": "christov2004", "clean": False},
        "nabian2018" :  {"method": "nabian2018", "clean": False},
        "rodrigues2021" : {"method": "rodrigues2021", "clean": False},
        "manikandan2012" : {"method": "manikandan2012", "clean": False},
        "promac" : {"method": "promac", "clean": False}
    }

    # execute and profile neurokit peak-detection algorithms for REF ECG signal as ground truth

    ### Delineate/Extract R-peaks of ECG_REF for ground truth peak labeling
    # approach 1:   use respective peak-detection method on ecg_reference
    #               TODO adress limitation: metric evaluation script only measures deterioration of peak detection method
    #               when using dirty signal (ecg_ic) instead of ECG refrence signal
    #
    # approach 2:   use respective peak-detection method on ecg_reference and employ artifact correction provided by neurokit2 
    #               (TODO CITE with artifact correction using the method of Lipponen & Tarvainen (2019) implemented by neurokit2)
    # further:      additionally perform "naive" peak sample checking, i.e if sample value is highest in area of 80ms around the peak 
    #               (+/- 40ms, so +/-10 samples at 250Hz) (since QRS duration in healthy adult lasts around 80-100ms TODO CITE wikipedia)

    for algo_name, algo_dict in peakDetect_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ref_signal = nk.ecg_clean(ecg_sync_ref, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ref_signal = ecg_sync_ref 
            time_start = time.time()
        
        # perform peak detection (with artifact correction provided by neurokit2 if desired)
        ref_results, ref_info = nk.ecg_peaks(ref_signal, sampling_rate = sampling_rate, method = algo_dict["method"], correct_artifacts = neurokit_peak_correction, show = False)
        time_total = time.time() - time_start

        # if desired, additionally perform "naive" peak sample checking / correction, by checking surrounding
        # samples for (higher) values
        if naive_peak_sample_correction:
            # fig_peaks_before = nk.events_plot(ref_info["ECG_R_Peaks"], ref_signal, color = "r")
            ref_info["ECG_R_Peaks"] = util.naive_peak_value_surround_checks(ref_signal, ref_info["ECG_R_Peaks"], 40)
            # fig_peaks_after = nk.events_plot(ref_info["ECG_R_Peaks"], ref_signal, color = "g")

        algo_dict["gnd_truth_results"] = ref_results
        algo_dict["gnd_truth_info"] = ref_info
        algo_dict["gnd_truth_exec_time"] = time_total


    # execute and profile neurokit peak-detection algorithms on dirty signal for comparision
    # do not use any artifact / peak correction whatsoever (for comparability)
    for algo_name, algo_dict in peakDetect_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ecg_signal = nk.ecg_clean(ecg_ic, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ecg_signal = ecg_ic 
            time_start = time.time()

        sig_results, sig_info = nk.ecg_peaks(ecg_signal, sampling_rate = sampling_rate, method = algo_dict["method"], correct_artifacts = False, show = False)
        time_total = time.time() - time_start
        
        algo_dict["sig_results"] = sig_results
        algo_dict["sig_info"] = sig_info
        algo_dict["sig_exec_time"] = time_total

    
    
    # calculate quality metrics regarding comparision to ground truth

    for algo_name, algo_dict in peakDetect_dicts.items():

        gnd_truth_beats = np.array((algo_dict["gnd_truth_results"])["ECG_R_Peaks"])
        gnd_truth_beat_amount = np.count_nonzero(gnd_truth_beats)
        signal_beats = np.array((algo_dict["sig_results"])["ECG_R_Peaks"])
        signal_beat_amount = np.count_nonzero(signal_beats)

        metrics_dict = calculate_metrics_with_reference(gnd_truth_beats, signal_beats)
        # store quality metrics
        algo_dict["metrics_dict"] = metrics_dict
        
    return peakDetect_dicts

    


def main():

    ############################ get data and perform ICA on it (analogous to ICA script)
    # TODO Outsource this completely to ICA script (Interface for ICA-ing needed)
    
    # load and select sample / window from test recordings (eeg and ecg channels seperately)
    # initiate test recording data objects as list
    dataRecordings = []
    filepaths = offD.getFilepaths()
    '''
    for path in filepaths:
        dataRecordings.append(RecordingData(path))
    '''
    dataRecordings.append(offD.RecordingData(filepaths[1]))
    ########################################
    
    ### perform offline ICA Analysis on loaded and (pre-) filtered (BrainFlow) EEG data of Test Recording No.1
    rawEEG = icascript.createObjectMNE('eeg', prepro.preprocessData(dataRecordings[0].getEEG()))
    rawECG = icascript.createObjectMNE('ecg', prepro.preprocessData(dataRecordings[0].getECG()))

    # get user template from ECG recording (not simulated)
    userTemplate = mne.io.Raw.copy(rawECG).crop(tmin=360.0, tmax=420.0)


    # Here we'll crop to 'tmax- tmin' seconds (240-180 = 60 seconds)
    # TODO Rausfinden warum float_NaN to int conversion Fehler, wenn tmin 300 und tmax 360
    # TODO Rausfinden warum broadcast / shape error wenn 
    rawEEG.crop(tmin = 180.0, tmax=240.0)
    rawECG.crop(tmin = 180.0, tmax=240.0)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)

    # apply ICA on the filtered raw EEG data
    global componentAmountConsidered

    # perform ICA comparing the different algorithmic performance on the data (regarding pTp SNR of ECG-related IC)
    method_key, ica_dicts = icascript.choseBestICA(rawEEG, rawECG, amountICs = componentAmountConsidered, maxIter = "auto")
    bestICA = ica_dicts[method_key]

    # TEST plot results (fusedECG) to visually check results (time series of ICs) of best ICA chosen
    
    rawResult = bestICA["sources"]
    #rawECG.add_channels([rawResult])
    #rawECG.plot(title = "rawECG (fused)")

    # extract components to perform cross-correlation on
    components = []
    timeseries = rawResult.get_data()
    for component in timeseries:
        components.append(component.flatten())

    ecg_related_index = bestICA["ECG_related_index_from_0"]
    ecg_related_timeseries = bestICA["ECG_related_Timeseries"]
    ecg_reference = (rawECG.get_data()).flatten()

    ################ Parameterize and Profile NeuroKit2. Peak-Finding Algorithms ################################

    print("################################################################################")
    print("NEEEEEEEEEEEEEEEWWWWWWWWWWWWWWWWWWWWWWWWWWWW MEEEEEEEEEEEEEEEEEEETHHHHHHHHHHHHHH")

    methods_dict = evaluate_all_peak_detect_methods(ecg_related_timeseries, ecg_reference)
    for algo_name, algo_dict in methods_dict.items():
        metrics_dict = algo_dict["metrics_dict"]
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")
        print("##########", algo_name, "##########")
        ###############
        print("### TIMING RESULTS ###")
        print("Ground truth took in total (sec): ", algo_dict["gnd_truth_exec_time"])
        print("Dirty signal took in total (sec): ", algo_dict["sig_exec_time"])
        ###############
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++ COMPARISION METRICS +++++++++++++++++++++++")
        print("REF BEAT COUNT: ", np.count_nonzero((algo_dict["gnd_truth_results"])["ECG_R_Peaks"]))
        print("SIG BEAT COUNT: ", np.count_nonzero((algo_dict["sig_results"])["ECG_R_Peaks"]))
        print("TRUE Positives total amount: ", metrics_dict["true_positives"])
        print("True Misses total amount: ", metrics_dict["false_negatives"])
        print("True False Positives total amount: ", metrics_dict["false_positives"])
        print("Peak displacements total amount: ", metrics_dict["amount_displacements"])
        print("Sample Displacement Mean: ", metrics_dict["mean_displacement_sample"])
        print("Sample Displacement Standard Deviation: ", metrics_dict["std_displacement_sample"])
        print("Jitter Value (Millis) Average: ", metrics_dict["avg_jitter_value_millis"])
        print("Jitter Score: ", metrics_dict["jitter_score"])
        print("F_1 Score: ", metrics_dict["f_1_score"])
        print("Resulting JF_Score: ", metrics_dict["jf_score"])
        print("Sensitivity reported: ", metrics_dict["sensitivity"])
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")




    # TODO Implement Method Choosing based on Quality Metrics
    
    
if __name__ == "__main__":
    main()