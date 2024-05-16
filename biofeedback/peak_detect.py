import numpy as np
import scipy.signal as scisig
import sklearn.preprocessing as sklprep
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import data.offlinedata as offD
import data.preprocessing as prepro

import ica as icascript
import utility as util

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
componentAmountConsidered = 7


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
        f_1_score = 0
        print("Calculating F_1 Score, an Error occured!")

    # calculate the JF-score according to TODO PAPER BERN PORR 
    jf_score = f_1_score * jitter_score * 100

    # calculate further metrics 
    # sensitivity here regards only (possibly displaced) hits and 
    # misses (with dynamic tolerance window of half distance to next beat)
    try:
        sensitivity = true_hits / (true_hits + true_misses)
    except:
        sensitivity = 0
        print("Calculating Sensitivity, an Error occured!")

    results = {
        "true_positives": true_hits,
        "false_negatives": true_misses,
        "false positives": true_false_positives,
        "mean_displacement_sample": np.mean(sample_displacements),
        "avg_jitter_value_millis": avg_jitter_millis,
        "jitter_score": jitter_score,
        "f_1_score": f_1_score,
        "jf_score": jf_score,
        "sensitivity": sensitivity
    }

    return results



    """
    # TODO ASSERT ref_segment contains exactly one peak (1) and rest is zeros!)
    # TODO otherwise raise error or skip segment (???)

    match = (np.count_nonzero(np.logical_and(ref_segment, signal_segment)) == 1)  # True if hit
    # all peaks that are either a displaced hit or false positives
    unmatched_signal_peaks = np.logical_xor(ref_segment, signal_segment)

    # if true, redundant peaks are present (false positives), besides one displaced hit
    redundancy = (np.count_nonzero(unmatched_signal_peaks) > 1)

    # check if *true* miss (no peak in signal segment whatsoever) 
    if np.count_nonzero(signal_segment) == 0:
        true_misses += 1
        continue

    # check if hit (possibly with further false positives present)
    if match:
        true_hits += 1
        
        # check if further false positives are present
        if redundancy:
            true_false_positives += np.count_nonzero(unmatched_signal_peaks)

        continue
        

    # at this point there must be a displacement (possibly with further false positives present)
    # identify signal peak closest to reference peak
    distances_to_ref_peak = np.abs(np.subtract((np.nonzero(ref_segment)[0]), (np.nonzero(signal_segment)[0])))
    # displacement as distance in samples
    min_distance = np.min(distances_to_ref_peak)
    sample_displacements.append(min_distance)

    # if present in signal segment, keep track of false positives (all unmatched peaks except displaced hit)
    true_false_positives += (np.count_nonzero(unmatched_signal_peaks) - 1)
    """


def evaluate_all_peak_detect_methods(ecg_related_ic, ecg_reference):

    ################ Parameterize and Profile NeuroKit2. Peak-Detection Algorithms ################################
    # setup dicts for each neurokit peak-detection algorithm 
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
        "zong2003" : {"method": "zong2003", "clean": False},
        "martinez2004" : {"method": "martinez2004", "clean": False},
        "christov2004" : {"method": "christov2004", "clean": False},
        # TODO Fehler untersuchen (index out of bounds)
        #"dirty_gamboa2008" : {"method": "gamboa2008", "clean": False},
        #"clean_gamboa2008" : {"method": "gamboa2008", "clean": True},
        "dirty_elgendi2010" : {"method": "elgendi2010", "clean": False},
        "clean_elgendi2010" : {"method": "elgendi2010", "clean": True},
        "dirty_engzeemod2012" : {"method": "engzeemod2012", "clean": False},
        "clean_engzeemod2012" :  {"method": "engzeemod2012", "clean": True},
        # TODO Fehler untersuchen (not implemented)
        #"manikandan2012" : {"method": "manikandan2012", "clean": False},
        "dirty_kalidas2017" : {"method": "kalidas2017", "clean": False},
        "clean_kalidas2017" : {"method": "kalidas2017", "clean": True},
        "nabian2018" :  {"method": "nabian2018", "clean": False},
        "rodrigues2021" : {"method": "rodrigues2021", "clean": False},
        # TODO Fehler untersuchen (not implemented bzw. "vg" method (nur für clean?))
        #"dirty_emrich2023" :  {"method": "emrich2023", "clean": False},
        #"clean_emrich2023" : {"method": "emrich2023", "clean": True},
        "promac" : {"method": "promac", "clean": False}
    }

    # execute and profile neurokit peak-detection algorithms for REF signal as ground truth

    ### Delineate/Extract R-peaks of ECG_REF for ground truth peak labeling
    # approach:     use respective peak-detection method on ecg_reference (TODO CITE with artifact correction using the method of Lipponen & Tarvainen (2019) implemented by neurokit2)
    #               and check additionally if sample value is highest in area of 80ms around the peak (+/- 40ms, so +/-10 samples at 250Hz)
    #               (since QRS duration in healthy adult lasts around 80-100ms TODO CITE wikipedia)

    for algo_name, algo_dict in peakDetect_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ref_signal = nk.ecg_clean(ecg_reference, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ref_signal = ecg_reference 
            time_start = time.time()
        
        ref_results, ref_info = nk.ecg_peaks(ref_signal, sampling_rate = sampling_rate, method = algo_dict["method"], correct_artifacts = True, show = False)
        time_total = time.time() - time_start

        # TODO Implement manual peak-correction (?)
        '''
        ####################
        # check, if each sample value around detected peak (+/- 10 samples) is highest, correct otherwise
        ####################
        ref_peakSamples = ref_info["ECG_R_Peaks"]
        nk.events_plot(ref_peakSamples, ref_signal)
        plt.show()
        for peak in ref_peakSamples:
            peak_height = ref_signal[peak]
            check_range = list(range(peak-10, peak+10))
            for checkSample in check_range:
                if ref_signal[checkSample] > peak_height:
                    print("ERROR FOR ", algo_name, " in REF Peak Detection")
                    print("ERROR!: peak at ", peak, "-th sample (", ref_signal[peak], ") is lower than value (", ref_signal[checkSample], ") of ", checkSample, "-th sample")
                    print("MANUAL ARTIFCAT CORRECTION WAS EMPLOYED")
                    # change peak location to new maxima found
                    peak = checkSample
        ####################
        '''

        algo_dict["gnd_truth_results"] = ref_results
        algo_dict["gnd_truth_info"] = ref_info
        algo_dict["gnd_truth_exec_time"] = time_total


    # execute and profile neurokit peak-detection algorithms on dirty signal for comparision
    for algo_name, algo_dict in peakDetect_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ecg_signal = nk.ecg_clean(ecg_related_ic, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ecg_signal = ecg_related_ic 
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

        metric_dict = calculate_metrics_with_reference(gnd_truth_beats, signal_beats)


        # store quality metrics
        algo_dict["true_misses"] = truly_missed_beats
        algo_dict["true_false_pos"] = truly_false_positives
        algo_dict["true_displace"] = truly_displacements
        algo_dict["displace_mean"] = mean_displacements
        algo_dict["displace_std"] = std_displacements
        

    """
    # print metric reports
    for algo_name, algo_dict in peakDetect_dicts.items():
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")
        print("##########", algo_name, "##########")
        ###############
        print("### GROUND TRUTH RESULTS ###")
        print("Took in total (sec): ", algo_dict["gnd_truth_exec_time"])
        print("Ref_Beat_Amount: ", np.count_nonzero((algo_dict["gnd_truth_results"])["ECG_R_Peaks"]))
        ###############
        print("### DIRTY SIGNAL RESULTS ###")
        print("### DIRTY SIGNAL RESULTS ###")
        print("Took in total (sec): ", algo_dict["sig_exec_time"])
        print("Sig_Beat_Amount: ", np.count_nonzero((algo_dict["sig_results"])["ECG_R_Peaks"]))
        ###############
        print("### COMPARISION METRICS NAIVELY ###")
        print("Amount matches found: ", algo_dict["matches_amount"])
        print("Amount misses found: ", algo_dict["misses_amount"])
        print("Amount false + found: ", algo_dict["false_pos_amount"])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++ Comparision Metrics Truly +++++++++++++++++++++++")
        print("WDH REF BEAT COUNT: ", np.count_nonzero((algo_dict["gnd_truth_results"])["ECG_R_Peaks"]))
        print("WDH SIG BEAT COUNT: ", np.count_nonzero((algo_dict["sig_results"])["ECG_R_Peaks"]))
        print("WDH MATCH TOTAL: ", algo_dict["matches_amount"])
        print("True Misses Amount: ", algo_dict["true_misses"])
        print("True False Positives Amount: ", algo_dict["true_false_pos"])
        print("True Displacements Amount: ", algo_dict["true_displace"])
        print("Mean of Displacements in Samples: ", algo_dict["displace_mean"])
        print("Standard Deviation of Displacements: ", algo_dict["displace_std"])

        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")

    """

    


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

    # setup dicts for each neurokit peak-finding algorithm 
    # TODO (w/ and w/o cleaned data, if applicable) -> macht das Sinn, das jeweils mit und ohne?
    # TODO Parameterize the methods independently??? -> geht über Scope hinaus
    # TODO Untersuche Neurokit Fehler bei Benutzung mancher Methoden
    peakFind_dicts = {
        "dirty_neurokit" : {"method": "neurokit", "clean": False},
        "clean_neurokit" : {"method": "neurokit", "clean": True},
        "dirty_pantompkins1985" : {"method": "pantompkins1985", "clean": False},
        "clean_pantompkins1985" : {"method": "pantompkins1985", "clean": True},
        "dirty_hamilton2002" : {"method": "hamilton2002", "clean": False},
        "clean_hamilton2002" : {"method": "hamilton2002", "clean": True},  
        "zong2003" : {"method": "zong2003", "clean": False},
        "martinez2004" : {"method": "martinez2004", "clean": False},
        "christov2004" : {"method": "christov2004", "clean": False},
        # TODO Fehler untersuchen (index out of bounds)
        #"dirty_gamboa2008" : {"method": "gamboa2008", "clean": False},
        #"clean_gamboa2008" : {"method": "gamboa2008", "clean": True},
        "dirty_elgendi2010" : {"method": "elgendi2010", "clean": False},
        "clean_elgendi2010" : {"method": "elgendi2010", "clean": True},
        "dirty_engzeemod2012" : {"method": "engzeemod2012", "clean": False},
        "clean_engzeemod2012" :  {"method": "engzeemod2012", "clean": True},
        # TODO Fehler untersuchen (not implemented)
        #"manikandan2012" : {"method": "manikandan2012", "clean": False},
        "dirty_kalidas2017" : {"method": "kalidas2017", "clean": False},
        "clean_kalidas2017" : {"method": "kalidas2017", "clean": True},
        "nabian2018" :  {"method": "nabian2018", "clean": False},
        "rodrigues2021" : {"method": "rodrigues2021", "clean": False},
        # TODO Fehler untersuchen (not implemented bzw. "vg" method (nur für clean?))
        #"dirty_emrich2023" :  {"method": "emrich2023", "clean": False},
        #"clean_emrich2023" : {"method": "emrich2023", "clean": True},
        "promac" : {"method": "promac", "clean": False}
    }

    # execute and profile neurokit peak-finding algorithms for REF signal as ground truth

    ### Delineate/Extract R-peaks of ECG_REF for ground truth peak labeling
    # approach 1:   use neurokit peak-extraction (TODO CITE with artifact correction using the method by Lipponen & Tarvainen (2019))
    #               and check if sample value is highest in area of 80ms around the peak (+/- 40ms, so +/-10 samples at 250Hz)
    #               (since QRS duration in healthy adult lasts around 80-100ms TODO CITE wikipedia)

    for algo_name, algo_dict in peakFind_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ref_signal = nk.ecg_clean(ecg_reference, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ref_signal = ecg_reference 
            time_start = time.time()
        
        ref_results, ref_info = nk.ecg_peaks(ref_signal, sampling_rate = sampling_rate, method = algo_dict["method"], correct_artifacts = True, show = False)
        time_total = time.time() - time_start

        # TODO Implement manual peak-correction (?)
        '''
        ####################
        # check, if each sample value around detected peak (+/- 10 samples) is highest, correct otherwise
        ####################
        ref_peakSamples = ref_info["ECG_R_Peaks"]
        nk.events_plot(ref_peakSamples, ref_signal)
        plt.show()
        for peak in ref_peakSamples:
            peak_height = ref_signal[peak]
            check_range = list(range(peak-10, peak+10))
            for checkSample in check_range:
                if ref_signal[checkSample] > peak_height:
                    print("ERROR FOR ", algo_name, " in REF Peak Detection")
                    print("ERROR!: peak at ", peak, "-th sample (", ref_signal[peak], ") is lower than value (", ref_signal[checkSample], ") of ", checkSample, "-th sample")
                    print("MANUAL ARTIFCAT CORRECTION WAS EMPLOYED")
                    # change peak location to new maxima found
                    peak = checkSample
        ####################
        '''

        algo_dict["gnd_truth_results"] = ref_results
        algo_dict["gnd_truth_info"] = ref_info
        algo_dict["gnd_truth_exec_time"] = time_total


    # execute and profile neurokit peak-finding algorithms on dirty signal for comparision
    for algo_name, algo_dict in peakFind_dicts.items():

        if algo_dict["clean"]:
            time_start = time.time()
            ecg_signal = nk.ecg_clean(ecg_related_timeseries, sampling_rate=sampling_rate, method=algo_dict["method"])
        else:
            ecg_signal = ecg_related_timeseries 
            time_start = time.time()

        sig_results, sig_info = nk.ecg_peaks(ecg_signal, sampling_rate = sampling_rate, method = algo_dict["method"], correct_artifacts = False, show = False)
        time_total = time.time() - time_start
        
        algo_dict["sig_results"] = sig_results
        algo_dict["sig_info"] = sig_info
        algo_dict["sig_exec_time"] = time_total

    
    
    # calculate quality metrics (matches, missed beats, false positives)
    # regarding comparision to ground truth

    for algo_name, algo_dict in peakFind_dicts.items():

        gnd_truth_beats = np.array((algo_dict["gnd_truth_results"])["ECG_R_Peaks"])
        gnd_truth_beat_amount = np.count_nonzero(gnd_truth_beats)
        signal_beats = np.array((algo_dict["sig_results"])["ECG_R_Peaks"])
        signal_beat_amount = np.count_nonzero(signal_beats)

    
    for algo_name, algo_dict in peakFind_dicts.items():

        gnd_truth_beats = np.array((algo_dict["gnd_truth_results"])["ECG_R_Peaks"])
        gnd_truth_beat_amount = np.count_nonzero(gnd_truth_beats)
        signal_beats = np.array((algo_dict["sig_results"])["ECG_R_Peaks"])
        signal_beat_amount = np.count_nonzero(signal_beats)

        # calculate matches (all common ones in ground_truth and signal beats)
        matches = np.logical_and(gnd_truth_beats, signal_beats)
        matches_amount = np.count_nonzero(matches)
        algo_dict["matches_amount"] = matches_amount

        # calculate naive (!) misses (TODO ASSERT? assume gnd_truth_beat_amount >= matches)
        # TODO calculate boolean array with misses (?)
        naive_misses = np.logical_and(np.logical_xor(gnd_truth_beats, signal_beats), gnd_truth_beats)
        misses_amount = np.count_nonzero(naive_misses)

        
        # TODO ASSERT misses_amount == gnd_truth_beat_amount - matches_amount
        if misses_amount != (gnd_truth_beat_amount - matches_amount):
            print("ERROR misses_amount != (gnd_truth_beat_amount - matches_amount)")
        algo_dict["misses_amount"] = misses_amount

        # calculate false positives OR displacements TODO EXPLAIN SUM[(REF XOR SIG) AND not(REF)]
        false_positives = np.logical_xor(gnd_truth_beats, signal_beats)
        false_positives = np.logical_and(false_positives, np.logical_not(gnd_truth_beats))
        false_pos_amount = np.count_nonzero(false_positives)
        algo_dict["false_pos_amount"] = false_pos_amount

        # calculate peak displacements 
        # TODO Explain
        
        # TODO Discern between displacement and combination of miss & false positive

        # segment signal beats array with peak locations into (presumably) missed beats
        # each segment has width of half the distance to next beat (left/right)
        true_peak_locations = (algo_dict["gnd_truth_info"])["ECG_R_Peaks"]
        missed_beat_segments = []

        truly_missed_beats = 0
        displacements = []
        truly_false_positives = 0

        last_end_idx = 0
        for iter_idx, peak_location in enumerate(true_peak_locations):
            segment_begin = last_end_idx
            peak_idx = peak_location
            # check if last peak
            if peak_location == true_peak_locations[-1]:
                segment_end = signal_beats.size - 1     # segment end is last index
            else:
                # segment end is halfway to next beat
                segment_end = peak_location + util.ceildiv(true_peak_locations[iter_idx + 1] - peak_location, 2)
            last_end_idx = segment_end

            # check if missed beat (naively) and iff, add to list of missed beat segments
            if naive_misses[peak_location]:
                beat_segment_tuple = (gnd_truth_beats[segment_begin:segment_end], signal_beats[segment_begin:segment_end])
                missed_beat_segments.append(beat_segment_tuple)
            elif np.count_nonzero(signal_beats[segment_begin:segment_end]) > 1:
                # check if hit with false positives
                truly_false_positives += (np.count_nonzero(signal_beats[segment_begin:segment_end]) - 1)

        # iterate all missed beat segements and discern displacements, false positives and missed beats
        
        for segment_tuple in missed_beat_segments:
            gnd_truth_segment = segment_tuple[0]
            #gnd_truth_beat_index = (np.nonzero(gnd_truth_segment))[0][0]
            signal_segment = segment_tuple[1]

            # check if truly missed beat (no peak in signal segment whatsoever)
            if np.count_nonzero(signal_segment) == 0:
                truly_missed_beats = truly_missed_beats + 1
                continue
            else:
                # identify closest signal peak to ground truth beat (as displaced beat)
                distances_to_beat = np.abs(np.subtract((np.nonzero(gnd_truth_segment)[0]), (np.nonzero(signal_segment)[0])))
                # displacement as distance in samples
                min_distance = np.min(distances_to_beat)
                displacements.append(min_distance)

                # check if also false positives present (for 1 beat in gnd truth segment, more than 1 peak in signal segment)
                if np.count_nonzero(signal_segment) > 1:
                    truly_false_positives = truly_false_positives + (np.count_nonzero(signal_segment) - 1)


        # calculate mean displacement distance in samples and standard deviation
        truly_displacements = len(displacements)
        mean_displacements = np.mean(displacements)
        std_displacements = np.std(displacements)

        # store quality metrics
        algo_dict["true_misses"] = truly_missed_beats
        algo_dict["true_false_pos"] = truly_false_positives
        algo_dict["true_displace"] = truly_displacements
        algo_dict["displace_mean"] = mean_displacements
        algo_dict["displace_std"] = std_displacements
        

        
        # COMPARE WITH NEW METHODS
        ergebnisse = calculate_metrics_with_reference(gnd_truth_beats, signal_beats)
        try:
            assert algo_dict["matches_amount"] == ergebnisse["true_positives"], "Matches Amount does not Match"
        except:
            print("QÄQÄQÄQÄQÄÄQQÄQÄÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄ Matches Amount does not Match")
        try:
            assert algo_dict["true_misses"] == ergebnisse["false_negatives"], "Misses Amount does not match"
        except:
            print("QÄQÄQÄQÄQÄÄQQÄQÄÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄ Misses Amount does not match")
        try:
            assert algo_dict["true_false_pos"] == ergebnisse["false positives"], " False Positives Amount does not match"
        except:
            print("QÄQÄQÄQÄQÄÄQQÄQÄÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄ False Positives Amount does not match")
        try:
            assert algo_dict["displace_mean"] == ergebnisse["mean_displacement_sample"], "Mean Displacement in Samples does not match"
        except:
            print("QÄQÄQÄQÄQÄÄQQÄQÄÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄQÄ Mean Displacement in Samples does not match")
        """
        assert algo_dict["matches_amount"] == ergebnisse["true_positives"], "Matches Amount does not Match"
        assert algo_dict["true_misses"] == ergebnisse["false_negatives"], "Misses Amount does not match"
        assert algo_dict["true_false_pos"] == ergebnisse["false positives"], " False Positives Amount does not match"
        assert algo_dict["displace_mean"] == ergebnisse["mean_displacement_sample"], "Mean Displacement in Samples does not match"
        """
    

    # print metric reports
    for algo_name, algo_dict in peakFind_dicts.items():
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")
        print("##########", algo_name, "##########")
        ###############
        print("### GROUND TRUTH RESULTS ###")
        print("Took in total (sec): ", algo_dict["gnd_truth_exec_time"])
        print("Ref_Beat_Amount: ", np.count_nonzero((algo_dict["gnd_truth_results"])["ECG_R_Peaks"]))
        ###############
        print("### DIRTY SIGNAL RESULTS ###")
        print("### DIRTY SIGNAL RESULTS ###")
        print("Took in total (sec): ", algo_dict["sig_exec_time"])
        print("Sig_Beat_Amount: ", np.count_nonzero((algo_dict["sig_results"])["ECG_R_Peaks"]))
        ###############
        print("### COMPARISION METRICS NAIVELY ###")
        print("Amount matches found: ", algo_dict["matches_amount"])
        print("Amount misses found: ", algo_dict["misses_amount"])
        print("Amount false + found: ", algo_dict["false_pos_amount"])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++ Comparision Metrics Truly +++++++++++++++++++++++")
        print("WDH REF BEAT COUNT: ", np.count_nonzero((algo_dict["gnd_truth_results"])["ECG_R_Peaks"]))
        print("WDH SIG BEAT COUNT: ", np.count_nonzero((algo_dict["sig_results"])["ECG_R_Peaks"]))
        print("WDH MATCH TOTAL: ", algo_dict["matches_amount"])
        print("True Misses Amount: ", algo_dict["true_misses"])
        print("True False Positives Amount: ", algo_dict["true_false_pos"])
        print("True Displacements Amount: ", algo_dict["true_displace"])
        print("Mean of Displacements in Samples: ", algo_dict["displace_mean"])
        print("Standard Deviation of Displacements: ", algo_dict["displace_std"])

        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")



    # TODO Implement Method Choosing based on Quality Metrics
    
    
if __name__ == "__main__":
    main()