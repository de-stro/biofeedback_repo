"""
    script to perform offline ICA Analysis in MNE on local (earlier recorded) EEG-Session (with parallel chest-ECG on
    channel 8, in BrainFlow Format (CSV)
"""

import math 
import numpy as np
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import data.data_offline as offD
import data.preprocessing as prepro
import utility
import icselect

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG

flip = 1
ecg_related_index = 0
componentAmountConsidered = len(eeg_channels)

'''
    Define a function to create epochs based on / around R-peak locations
    TODO Wie die Epochs createn? Wieviel ms um die R-Peaks to consider?
'''
def extract_heartbeats(cleaned, peaks, sampling_rate=None): 
    heartbeats = nk.epochs_create(cleaned, 
                                  events=peaks, 
                                  epochs_start=-0.3, 
                                  epochs_end=0.4, 
                                  sampling_rate=sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    return heartbeats

# Creating MNE objects from brainflow data array
def createObjectMNE(type: str, data_2D):

    data_2D = data_2D / 1000000  # BrainFlow returns uV, convert to V for MNE!1!1!1!

    if type == 'eeg':
        ch_types = ['eeg'] * len(eeg_channels)
        info = mne.create_info(ch_names=len(eeg_channels), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)
    else:
        ch_types = ['ecg']
        info = mne.create_info(ch_names=len(ecg_channel), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)


def calculate_corr_sign(ref_signal, noisy_signal):
    mittelwert = (np.max(ref_signal) + np.min(ref_signal)) / 2
    normalized_ref = ref_signal - mittelwert
    normalized_ref = normalized_ref / np.max(normalized_ref)

    mittelwert = (np.max(noisy_signal) + np.min(noisy_signal)) / 2
    normalized_noisy = noisy_signal - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy) 

    similarity = normalized_noisy * normalized_ref

    return np.sign(np.mean(similarity))



def calculate_snr_wR(clean_signal, noisy_signal):
    mittelwert = (np.max(clean_signal) + np.min(clean_signal)) / 2
    normalized_clean = clean_signal - mittelwert
    normalized_clean = normalized_clean / np.max(normalized_clean)

    clean_power = np.mean(np.square(normalized_clean))

    mittelwert = (np.max(noisy_signal) + np.min(noisy_signal)) / 2
    normalized_noisy = noisy_signal - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy) 

    similarity = normalized_noisy * normalized_clean

    normalized_noisy = normalized_noisy * np.sign(np.mean(similarity))

    noise = normalized_noisy - normalized_clean
    
    # TODO richtige SNR Formel (insb. Vorfaktor 20 oder 10?)
    noise_power = np.mean(np.square(noise))
    snr = 10 * np.log10(clean_power / noise_power)
    return snr

''' Correlation-based Method: Compute the correlation coefficient between the noisy signal and the clean reference signal. 
    Return Pearson product-moment correlation coefficients
    The closer the correlation coefficient is to 1, the higher the SNR. 
    TODO Code Explaination in detail needed! How is Corr caluculated??
'''
def calculate_snr_correlation_wR_CGPT(ref_signal, noisy_signal):
    correlation = np.corrcoef(ref_signal, noisy_signal)[0, 1]
    snr = correlation ** 2
    return snr

''' segment the data into epochs around QRS complexes detected
    TODO peak_finding methode mit in methoden kopf nehmen!

'''
def epoch_for_QRS_and_noise(noisy_signal, reference_signal = None):

    naive_peak_correction = True

    # if passed, use reference signal for delineation (to mark QRS-complexes etc) 
    if reference_signal is not None:
        # detect R-peaks in reference signal
        ref_peaks_signal, ref_peaks_info = nk.ecg_peaks(reference_signal, sampling_rate = sampling_rate, 
                                                        method = 'neurokit', correct_artifacts = True, show = False)
        
        peak_locations = ref_peaks_info["ECG_R_Peaks"]
        # additionally employ "naive" peak correction, by checking if peak sample has highest value compared to 
        # surrounding samples (only reasonable for clean ref ecg with no noisy peaks inbetween R-peaks)
        # TODO Determine appropriate check range for peak correction!!
        if naive_peak_correction:
            peak_locations = utility.naive_peak_value_surround_checks(reference_signal, peak_locations, 10)
            # TEST VISUAL!!!
            #nk.events_plot(corrected_peak_locs, reference_signal)
            #plt.show()

        # Delineate cardiac cycle of reference signal (with peak locations checked additionally)
        ref_signals, waves = nk.ecg_delineate(reference_signal, peak_locations, sampling_rate = sampling_rate,
                                                  method='dwt', show=False, show_type = "peaks")
        
    else:
        # detect R-peaks in noisy signal
        noisy_peaks_signal, noisy_peaks_info = nk.ecg_peaks(noisy_signal, sampling_rate = sampling_rate, 
                                                        method ='neurokit', correct_artifacts = True, show = False)
        # Delineate cardiac cycle of noisy signal
        noisy_signals, waves = nk.ecg_delineate(noisy_signal, noisy_peaks_info["ECG_R_Peaks"], sampling_rate = sampling_rate,
                                                  method = 'dwt', show = False, show_type = "bounds_R")
        
    onsets_R_list = np.array(waves["ECG_R_Onsets"])
    offsets_R_list = np.array(waves["ECG_R_Offsets"])
    """TEST onsets und offset immer gleich lang trotz evtl. NaN padding und NaN padding nur max ein element
    am anfang und/oder am ende"""
    # TESTE DIE INVARIANTEN UND PADDING PRÄMISSEN
    assert onsets_R_list.size == offsets_R_list.size
    assert np.count_nonzero(np.isnan(onsets_R_list)) <= 2
    assert np.count_nonzero(np.isnan(offsets_R_list)) <= 2
    
    if np.isnan(onsets_R_list[0]) and (not np.isnan(offsets_R_list[0])):
        assert offsets_R_list[0] < onsets_R_list[1]
    if np.isnan(offsets_R_list[0]) and (not np.isnan(onsets_R_list[0])):
        assert onsets_R_list[0] < offsets_R_list[1]
    
    if np.isnan(onsets_R_list[-1]) and (not np.isnan(offsets_R_list[-1])):
        assert offsets_R_list[-1] > onsets_R_list[1]
    if np.isnan(offsets_R_list[-1]) and (not np.isnan(onsets_R_list[-1])):
        assert onsets_R_list[0] < offsets_R_list[1]
    
    # prune onsets and offsets to remove NaN entries introduced by neurokit2
    onsets_R_list, offsets_R_list = utility.prune_onsets_and_offsets(onsets_R_list, offsets_R_list)
    
    # TODO Delete: remove NaN padding introduced by neurokit2 ecg_delineation
    #onsets_R_list = utility.remove_NaN_padding(onsets_R_list)
    #offsets_R_list = utility.remove_NaN_padding(offsets_R_list)

    # ensure that epochs beginn with first R_Offset (hence, a noise_segment) and end with last R_Offset,
    # therefore, last detected QRS complex is included, while first QRS complex might be omitted!
    """
    if onsets_R_list[0] < offsets_R_list[0]:
        onsets_R_list = onsets_R_list[1:]
    if onsets_R_list[-1] > offsets_R_list[-1]:
        onsets_R_list = onsets_R_list[:-1]
    
    # prune offset and onsets before comparision to remove NaN padding introduced by neurokit2 (otherwise always False)
    
    pruned_onsets = onsets_R_list[~np.isnan(onsets_R_list)]
    pruned_offsets = offsets_R_list[~np.isnan(offsets_R_list)]

    if pruned_onsets[0] < pruned_offsets[0]:
        onsets_R_list = onsets_R_list[1:]
    if pruned_onsets[-1] > pruned_offsets[-1]:
        onsets_R_list = onsets_R_list[:-1]

    """

    # now size offsets_R_list must be equal or 1 greater than size onsets_R_list!
    #print("TEST:: Size of onset_R_list ", len(onsets_R_list), " and of offset_R_list: ", len(offsets_R_list))
    #print("TEST:: OFFSET_R_LIST MUST BE EQUAL OR 1 GREATER THAN THE SIZE OF ONSET_R_LIST !!")
    print("$§$§$§$§$§$§$§$§$§$§$§$$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$")
    # calculate duration between R_offset and R_onset in seconds (omit last R_Offset for noise epochs) TODO WHY?!? 
    # but if onsets starts with NaN, also prune onsets (to truly calculate noise epochs duration)
    # TODO DOCUMENT Segmentation Procedure! visuell in documentation!
    noise_epochs_duration = onsets_R_list - (offsets_R_list[:-1])
    noise_epochs_duration = noise_epochs_duration / sampling_rate

    # calculate duration between R_onset and R_offset in seconds (omit first R_Offset for QRS epochs)
    qrs_epochs_duration = (offsets_R_list[1:]) - onsets_R_list
    qrs_epochs_duration = qrs_epochs_duration / sampling_rate
    #print("TEST:: Length of epochs_duration arrays must be the same!!!")
    #print("TEST::Length of qrs_epochs_duration: ", qrs_epochs_duration.size, " and of noise_epochs_duration: ", noise_epochs_duration.size)
    ''' TEST'''
    print("mean duration for QRS complex (in sec): ", np.mean(qrs_epochs_duration))
    print("duration of 10th QRS complex (in sec): ", ((waves["ECG_R_Offsets"])[9] - (waves["ECG_R_Onsets"])[9]) / sampling_rate)
    
    # remove NaN padding introduced by neurokit2 ecg_delineation
    #noise_epochs_duration = utility.remove_NaN_padding(noise_epochs_duration)
    #qrs_epochs_duration = utility.remove_NaN_padding(qrs_epochs_duration)
    #onsets_R_list = utility.remove_NaN_padding(onsets_R_list)
    #offsets_R_list = utility.remove_NaN_padding(offsets_R_list)

    # TODO FIX BUG with new neurokit2 version (0.2.7 instead of 0.2.3) regarding conversion float NaN to integer
    # BUG with 0.2.3 version another bug occurs earlier at LOC::116 from neurokit2 signal_interpolate ("operands could not be broadcast together with shapes (69,) (15001,)")
    # signal delineation to create epochs of noisy signal around QRS-complexes found 
    qrs_epochs = nk.epochs_create(noisy_signal, events = onsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = qrs_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
    
    # do the same epoching for epochs of noise parts of noisy signal (all except for QRS complexes)
    noise_epochs = nk.epochs_create(noisy_signal, events = offsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = noise_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
        
    """ TEST 
    # TODO Plotte hier das ganze Signal (QRS + dazugehöriges Noise Segment), drunter QRS und Noise einzel
    # Iterate through epoch data
    i = 0
    for qrs_epoch in qrs_epochs.values():
        if i == 10:
            # Plot scaled signals
            nk.signal_plot(qrs_epoch['Signal']) 
        i = i + 1
    
    i = 0
    for noise_epoch in noise_epochs.values():
        if i == 10:
            # Plot scaled signals
            nk.signal_plot(noise_epoch['Signal']) 
        i = i + 1

    plt.show()
    """
    return (qrs_epochs, noise_epochs)
    
def epoch_peak_to_peak_amplitude(epoch):
    signal = epoch['Signal']
    max_peak = np.max(signal)
    min_peak = np.min(signal)
    return max_peak - min_peak

def epoch_RSSQ(epoch):
    signal = epoch['Signal']
    RSSQ = math.sqrt(np.sum(np.square(signal)))
    return RSSQ



''' Calculate SNR of noisy signal epoched around heart beats / R-peaks detected (in noisy signal, no reference)
'''
def calculate_snr_PeakToPeak_from_epochs(qrs_epochs, noise_epochs):
    # get avg peak-to-peak-amplitude of signal epochs (qrs)
    qrs_pTp_amplitude_sum = 0
    for qrs_epoch in qrs_epochs.values():
        qrs_pTp_amplitude_sum = qrs_pTp_amplitude_sum + epoch_peak_to_peak_amplitude(qrs_epoch)
    qrs_pTp_amplitude_avg = qrs_pTp_amplitude_sum / len(qrs_epochs)

    # get avg peak-to-peak-amplitude of noise epochs (noise)
    noise_pTp_amplitude_sum = 0
    for noise_epoch in noise_epochs.values():
        noise_pTp_amplitude_sum = noise_pTp_amplitude_sum + epoch_peak_to_peak_amplitude(noise_epoch)
    noise_pTp_amplitude_avg = noise_pTp_amplitude_sum / len(noise_epochs)

    # calculate SNR in dB
    # TODO genaue Formel (insb Vorfaktor) für milliVolts???! (siehe unten)
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power
    snr = 20 * np.log10(qrs_pTp_amplitude_avg / noise_pTp_amplitude_avg)
    return snr

def calc_snr_RSSQ_from_epochs(qrs_epochs, noise_epochs):
    # get avg RSSQ of signal epochs (qrs)
    qrs_RSSQ_sum = 0
    for qrs_epoch in qrs_epochs.values():
        qrs_RSSQ_sum = qrs_RSSQ_sum + epoch_RSSQ(qrs_epoch)
    qrs_RSSQ_avg = qrs_RSSQ_sum / len(qrs_epochs)

    noise_RSSQ_sum = 0
    for noise_epoch in noise_epochs.values():
        noise_RSSQ_sum = noise_RSSQ_sum + epoch_RSSQ(noise_epoch)
    noise_RSSQ_avg = noise_RSSQ_sum / len(noise_epochs)

    # calculate SNR in dB
    # TODO genaue Formel (insb Vorfaktor) für milliVolts???! (siehe unten)
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power
    snr = 20 * np.log10(qrs_RSSQ_avg / noise_RSSQ_avg)
    return snr

# TODO OMIT THIS METHOD
def choseBestICA(eeg_data:mne.io.Raw, ecg_ref:mne.io.Raw, amountICs, maxIter):
    """Given an eeg_signal, this methods computes ICA using the best performing algorithm based on (pTp) SNR and 
    returns dictionary with the results after fitting (key and dict of ICA dicts)
    """
    
    ''' TODO assume the data is already preprocessed and as (cropped) MNE Object
    # perform preprocessing on raw data
    prep_eeg = prepro.preprocessData(eeg_data)
    prep_ecg = prepro.preprocessData(ecg_ref)

    # create MNE Raw Objects to perform ICA on
    mne_eeg = createObjectMNE("eeg", prep_eeg)
    mne_ecg = createObjectMNE("ecg", prep_ecg)
    '''
    # TODO was this already done (in preprocessing)??
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filtBP_eeg = eeg_data.copy().filter(l_freq=1.0, h_freq=50.0)

    # compute and compare the different MNE ICA algorithm implementations (comp time and SNR of ECG-IC)
    
    # configure the ica objects for fitting
    picard_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard')}
    infomax_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax')}
    fastica_dict = {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica')}
    ica_dicts = {"picard": picard_dict, "infomax": infomax_dict, "fastica": fastica_dict}

    # fit the ICA objects and time the fitting
    for methodName, ica_dict in ica_dicts.items():
        ica_dict["rawCopy"] = eeg_data.copy()
        ica_dict["filterCopy"] = filtBP_eeg.copy()

        startTime = time.time()
        ica_dict["ica"].fit(ica_dict["filterCopy"])
        fitTime = time.time() - startTime

        ica_dict["fitTime"] = fitTime
        ica_dict["sources"] = ica_dict["ica"].get_sources(ica_dict["rawCopy"]) # TODO can we also *not* use Raw Obj Instance here?

        # identify ECG-related IC (using ecg_ref) and compute the resulting SNR
        timeseries, times = (ica_dict["sources"]).get_data(return_times=True)
        timeseries.flatten()
        snr_correlations_CGPT = []
        ecg_ref_series = (ecg_ref.get_data()).flatten()
        
        for component in timeseries:
            snr_correlations_CGPT.append(calculate_snr_correlation_wR_CGPT(ecg_ref_series, component))

        ecg_related_index = snr_correlations_CGPT.index(max(snr_correlations_CGPT))
        ica_dict["ECG_related_index_from_0"] = ecg_related_index

        print("ChoseBest_TEST: ECG_RELATED INDEX")
        print("Method used: ",  methodName)
        print("ECG-related Component calulated: ", ecg_related_index, "-th component (starting from 0!)")
        print("with SNR_corr value of: ", snr_correlations_CGPT[ecg_related_index])

        # TODO Plot the sources / results to visually control calculated ECG_component

        # calulate peak-to-peak SNR of ECG-related component
        ecg_related_timeseries = timeseries[ecg_related_index]
        # flip ecg-related component if negatively correlated with ref_ecg
        ecg_related_timeseries *= calculate_corr_sign(ecg_ref_series, ecg_related_timeseries)

        ica_dict["ECG_related_Timeseries"] = ecg_related_timeseries

        # segment the signal into indivdual heartbeats to calculate pTp SNR / RSSQ SNR
        epoch = epoch_for_QRS_and_noise(ecg_related_timeseries, ecg_ref_series)
        #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

        snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
        print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")
        ica_dict["pTp_SNR_dB"] = snr_ptp

        snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
        print("SNR (avg RSSQ) is ", snr_rssq, "dB")
        ica_dict["rssq_SNR_dB"] = snr_rssq
    
    # chose best performing ICA algorithm and return results (based on SNR alone)
    # TODO calculate metric (optimal time/SNR ratio) to rank algorithm performance
    maxSNR_pTp = (ica_dicts["picard"])["pTp_SNR_dB"]
    bestMethod_pTp = "picard"
    maxSNR_rssq = (ica_dicts["picard"])["rssq_SNR_dB"]
    bestMethod_rssq = "picard"

    for methodName, ica_dict in ica_dicts.items():
        if  ica_dict["pTp_SNR_dB"] > maxSNR_pTp:
            maxSNR_pTp = ica_dict["pTp_SNR_dB"]
            bestMethod_pTp = methodName
        if ica_dict["rssq_SNR_dB"] > maxSNR_rssq:
            maxSNR_rssq = ica_dict["rssq_SNR_dB"]
            bestMethod_rssq = methodName
    
    print("########## TEST_choseBestICA: RESULTS ###################")
    print("Best method regarding pTp_SNR: ", bestMethod_pTp, "with ", maxSNR_pTp, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_pTp])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_pTp])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_pTp])["ica"].n_iter_)
    print("###")
    print("Best method regarding rssq_SNR: ", bestMethod_rssq, "with ", maxSNR_rssq, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_rssq])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_rssq])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_rssq])["ica"].n_iter_)

    return (bestMethod_pTp, ica_dicts)

# TODO OMIT THIS METHOD
def evaluate_all_ICA_variants(eeg_data, ecg_ref, amountICs:int, maxIter:int):
    """Given an eeg_signal and synchronized reference ecg_signal, this methods computes ICA using all variants provided by MNE and
    calulates metrics (pTp-SNR etc) for ECG-related IC (component having highest correlation with reference ecg) 
    returns dictionary with the metrics and results after fitting (key and dict of ICA dicts)
    """
    # takes as inputs brainflow data arrays !!! (2D)
    # data is already assumed to be appropriately preprocessed !!!  
    # TODO Welche übergreifenden Params noch auslagern? (für direkten Zugriff aus Eval-Script!)

    # transfrom input ecg_ref into 2D array for MNE object creation
    ecg_ref = [ecg_ref]

    # create mne.io.Raw Objects from input data to perform mne.preprocessing.ica on
    start_objCreateTime = time.time()
    mne_rawEEG = createObjectMNE('eeg', eeg_data)
    mne_rawECG = createObjectMNE('ecg', ecg_ref)
    objCreateTime = time.time() - start_objCreateTime

    "FOR TEST ONLY"
    print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")
    print("CREATION OF MNE RAW OBJ (for ICA evaluation) took in total (sec: ", objCreateTime)
    print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")

    # TODO OMIT as this was already done (in preprocessing)??
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    # If you intend to fit ICA on Epochs, it is recommended to high-pass filter, 
    # but not baseline correct the data for good ICA performance. A warning will be emitted otherwise.
    # TODO warum hier copy nötig? verändert MNE filter was an den daten (inplace?)???
    raw_EEG_copy = mne_rawEEG.copy()
    start_MNEfilterTime = time.time()
    filtBP_eeg = raw_EEG_copy.filter(l_freq=1.0, h_freq=50.0)
    MNEfilterTime = time.time() - start_MNEfilterTime

    # compute and compare the different MNE ICA algorithm implementations (comp time and SNR of ECG-IC)
    ##################################################################
    # TODO Chapter Limitations! : welche obvious param variations noch abdeckbar und sinnvoll?
    #################################################################
    # configure the ica objects for fitting
    ica_dicts = {
        "picard" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard')},
        "infomax" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax')},
        "fastica" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica')}

        # TODO similar to infomax (according to MNE https://mne.tools/stable/generated/mne.preprocessing.ICA.html)
        #"picard" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=False, extended=False))},
        #"picard_o" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=True, extended=False))},
        # TODO similar to extended Infomax (accord to MNE)
        #"ext_picard" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=False, extended=True))},
        # TODO similar to FASTICA according to MNE
        #"ext_picard_o" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=True, extended=True))},
        #"infomax" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax')},
        #"ext_infomax" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax', fit_params=dict(extended=True))},
        #"fastica" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica')},
        #"fastica_par" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica', fit_params=dict(algorithm='parallel'))},
        #"fastica_defl" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica', fit_params=dict(algorithm='deflation'))}
    }


    # fit the MNE ICA objects and time the fitting
    for methodName, ica_dict in ica_dicts.items():
        
        # store MNE overhead time (raw obj creation and bandpass filter)
        ica_dict["MNE_rawObjCreationTime"] = objCreateTime
        ica_dict["MNE_bandPassFilterTime"] = MNEfilterTime

        # TODO warum hier copy notwendig, passiert fit inplace bzw. verändert die Daten??
        filtBP_eeg_copy = filtBP_eeg.copy()
        startTime = time.time()
        ica_dict["ica"].fit(filtBP_eeg_copy)
        fitTime = time.time() - startTime
        ica_dict["icaFitTime"] = fitTime

        # prepare stuff for IC-Select (using ecg_ref)
        # TODO warum hier copy notwendig, verändert get_sources was an den Daten?
        mne_rawEEG_copy = mne_rawEEG.copy()
        sources = ica_dict["ica"].get_sources(mne_rawEEG_copy) # TODO can we also *not* use Raw Obj Instance here?
        ic_timeseries, times = sources.get_data(return_times=True)
        ic_timeseries.flatten()
        ecg_ref_series = (mne_rawECG.get_data()).flatten()

        # TODO Omit This as its unneccessary? ist REF_ECG nach MNE ICA nicht mehr synchronized mit resulting ICs timeseries?
        # store timeseries of resulting independent components, as well as ECG reference signal
        ica_dict["ic_timeseries"] = ( ic_timeseries, times)
        ica_dict["ref_ecg_timeseries"] = ecg_ref_series

    # TODO outsource this to IC-Select Script !?
    # identify ECG-related IC (using ecg_ref) and calculate resulting metrics
    for methodName, ica_dict in ica_dicts.items():

        timeseries = (ica_dict["ic_timeseries"])[0]
        ecg_ref_series = ica_dict["ref_ecg_timeseries"]

        # compute correlation of components with ECG reference to identify ECG-related IC
        correlations_with_ref = []
        for component in timeseries:
            correlations_with_ref.append(np.abs(np.corrcoef(ecg_ref_series, component)[0, 1]))
            
        ecg_related_index = correlations_with_ref.index(max(correlations_with_ref))
        ica_dict["ECG_related_index_from_0"] = ecg_related_index

        # TEST TSTE
        #identify_ecg_related_ic(ecg_ref_series, timeseries)

        ####### TEST TEST TEST ####################################################
        print("ChoseBest_TEST: ECG_RELATED INDEX")
        print("Method used: ",  methodName)
        print("ECG-related Component calulated: ", ecg_related_index, "-th component (starting from 0!)")
        print("with correlation value of: ", correlations_with_ref[ecg_related_index])
        #### TEST VISUALLY ########################################################
        # Plot the sources / results to visually control calculated ECG_component
        fig, axs = plt.subplots(amountICs + 1, 1, sharex = True)
        axs[0].plot(ecg_ref_series)
        axs[0].set_title('clean ecg')

        for idx, component in enumerate(timeseries):
            axs[idx+1].plot(component)
            axs[idx+1].set_title(str(idx+1) + "-th component")
        plt.show()
        #### TEST VISUALLY ########################################################
        

        # calulate peak-to-peak SNR of ECG-related component
        ecg_related_timeseries = timeseries[ecg_related_index]
        # flip ecg-related component if negatively correlated with ref_ecg
        ecg_related_timeseries *= calculate_corr_sign(ecg_ref_series, ecg_related_timeseries)

        # TODO OMIT this (brauchen wir das wirklich?)
        #ica_dict["ECG_related_Timeseries"] = ecg_related_timeseries

        # segment the signal into indivdual heartbeats to calculate pTp SNR / RSSQ SNR
        epoch = epoch_for_QRS_and_noise(ecg_related_timeseries, ecg_ref_series)
        #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

        snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
        print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")
        ica_dict["pTp_SNR_dB"] = snr_ptp

        snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
        print("SNR (avg RSSQ) is ", snr_rssq, "dB")
        ica_dict["rssq_SNR_dB"] = snr_rssq
    
    # chose best performing ICA algorithm and return results (based on SNR alone)
    # TODO calculate metric (optimal time/SNR ratio) to rank algorithm performance
    maxSNR_pTp = (ica_dicts["picard"])["pTp_SNR_dB"]
    bestMethod_pTp = "picard"
    maxSNR_rssq = (ica_dicts["picard"])["rssq_SNR_dB"]
    bestMethod_rssq = "picard"

    for methodName, ica_dict in ica_dicts.items():
        if  ica_dict["pTp_SNR_dB"] > maxSNR_pTp:
            maxSNR_pTp = ica_dict["pTp_SNR_dB"]
            bestMethod_pTp = methodName
        if ica_dict["rssq_SNR_dB"] > maxSNR_rssq:
            maxSNR_rssq = ica_dict["rssq_SNR_dB"]
            bestMethod_rssq = methodName
    
    print("########## TEST_choseBestICA: RESULTS ###################")
    print("Best method regarding pTp_SNR: ", bestMethod_pTp, "with ", maxSNR_pTp, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_pTp])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_pTp])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_pTp])["ica"].n_iter_)
    print("###")
    print("Best method regarding rssq_SNR: ", bestMethod_rssq, "with ", maxSNR_rssq, " dB")
    print("with ECG_related component being the ", (ica_dicts[bestMethod_rssq])["ECG_related_index_from_0"], "-th component (from 0!)")
    print("Took in sec: ", (ica_dicts[bestMethod_rssq])["fitTime"], "with amount iterations: ", (ica_dicts[bestMethod_rssq])["ica"].n_iter_)

    return (bestMethod_pTp, ica_dicts)


def calulate_ICA_metrics_with_ecg_ref(ecg_ref_signal, ecg_related_component):

    # segment the signal into indivdual heartbeats to calculate pTp SNR / RSSQ SNR
    epoch = epoch_for_QRS_and_noise(ecg_related_component, ecg_ref_signal)

    snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")

    snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    
    return (snr_ptp, snr_rssq)


def evaluate_all_MNE_ICA_on_segment(ecg_ref_signal, eeg_signals, component_amount, max_iterations):
    """evaluate_all_MNE_ICA_on_segment _summary_

    Args:
        ecg_ref_signal (_type_): _description_
        eeg_signals (_type_): _description_
        component_amount (_type_): _description_
        max_iterations (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    # TODO methode soll ecg_ref als numpy array und eeg_signls als 2d np-array nehmen und numpa array mit ECG-related ICS raushauen
    #sowie nparray?? oder list whatever mit ica_metric_dicts raushauen
    
    
    # TODO data is assumed to already be preprocessed appropriately 
    # TODO : 
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    # If you intend to fit ICA on Epochs, it is recommended to high-pass filter, 
    # but not baseline correct the data for good ICA performance. A warning will be emitted otherwise.
    # TODO warum hier copy nötig? verändert MNE filter was an den daten (inplace?)???
    

    # create mne.io.Raw Objects from input eeg data to perform MNE ICA on
    start_rawObj_create_time = time.time()
    mne_rawEEG = createObjectMNE('eeg', eeg_signals)
    mne_rawECG = createObjectMNE('ecg', ecg_ref_signal.reshape(1, ecg_ref_signal.size))
    rawObj_create_time = time.time() - start_rawObj_create_time

    # highpass filter the data using MNE filter # TODO also for ECG needed ie. to keep synchrony??
    start_mne_filter_time = time.time()
    mne_filtEEG = mne_rawEEG.copy().filter(l_freq=1.0, h_freq=None)
    # mne_filtECG = mne_rawECG.copy().filter(l_freq=1.0, h_freq=None)
    mne_filter_time = time.time() - start_mne_filter_time

    """FOR TEST ONLY """
    mne_object_length = ((((createObjectMNE('ecg', ecg_ref_signal.reshape(1, ecg_ref_signal.size))).get_data()).flatten())).size
    assert ecg_ref_signal.size == mne_object_length, "MNE alters ECG timeseries length"
    # TODO if assert error then we have to use mne raw also on ECG timeseries
    # ecg_ref_signal = mne_filtECG.get_data().flatten()

    ######################################################################
    # compute and compare the different MNE ICA algorithm implementations
    ######################################################################
    # TODO Chapter Limitations! : welche obvious param variations noch abdeckbar und sinnvoll?
    
    # setup list of ica dicts that store the metrics
    # TODO what about random_state variabel??
    ica_dicts = [
        {"method_id" : "picard", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='picard')},
        {"method_id" : "infomax", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='infomax')},
        {"method_id" : "fastica", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='fastica')}

        # TODO similar to infomax (according to MNE https://mne.tools/stable/generated/mne.preprocessing.ICA.html)
        #"picard" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=False, extended=False))},
        #"picard_o" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=True, extended=False))},
        # TODO similar to extended Infomax (accord to MNE)
        #"ext_picard" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=False, extended=True))},
        # TODO similar to FASTICA according to MNE
        #"ext_picard_o" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='picard', fit_params=dict(ortho=True, extended=True))},
        #"infomax" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax')},
        #"ext_infomax" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='infomax', fit_params=dict(extended=True))},
        #"fastica" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica')},
        #"fastica_par" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica', fit_params=dict(algorithm='parallel'))},
        #"fastica_defl" : {"ica": ICA(n_components=amountICs, max_iter=maxIter, random_state=97, method='fastica', fit_params=dict(algorithm='deflation'))}
        ]

    # setup list to store the ecg_related timeseries for each MNE ICA variant
    seg_ecg_related_ics = []

    # setup lists for ica metrics (numerical values)
    seg_computation_times_sec = []
    seg_pTp_snrs_dB = []
    seg_rssq_snrs_dB = []
    seg_corrs_with_refECG = []

    # fit the MNE ica objects and time the fitting
    for ica_dict in ica_dicts:

        # TODO ist hier copy wirklich notwendig? verändert MNE ICA fitting die original daten???
        eeg_fit_copy = mne_filtEEG.copy() #mne_rawEEG.copy()

        start_fit_time = time.time()
        ica_dict["ica"].fit(eeg_fit_copy)
        fit_time = time.time() - start_fit_time

        # store fitting time and Raw Obj creation time and MNE Filtering time
        ica_dict["time_fit"] = fit_time
        ica_dict["rawObj_create_time"] = rawObj_create_time
        ica_dict["mne_filter_time"] = mne_filter_time
        
        seg_computation_times_sec.append(fit_time + rawObj_create_time + mne_filter_time)

        # store the resulting independent components
        # TODO Warum hier copy nötig?? verändert get_sources die Daten???
        sources = ica_dict["ica"].get_sources(mne_rawEEG.copy()) 
        ic_timeseries, times = sources.get_data(return_times=True)
        ic_timeseries.flatten()
        
        # TODO ist das wirklich nötig?
        ica_dict["independent_components"] = ic_timeseries

        # identify the ecg-related independent component (by correlating with ecg reference)
        ecg_related_component, corr_value = icselect.identify_ecg_component_with_ref(ecg_ref_signal, ic_timeseries)
        seg_ecg_related_ics.append(ecg_related_component)
        seg_corrs_with_refECG.append(corr_value)

        # if negatively correlated, flip the ecg-related component around
        if corr_value < 0:
            ecg_related_component *= np.sign(corr_value)

        """ TEST: if flip successful / correctly
        fig, axs = plt.subplots(2, 1, sharex = True)
        axs[0].plot(ecg_ref_signal)seg_ecg_related_ics
        axs[0].set_title('ECG Reference Signal')
        axs[1].plot(ecg_related_component)
        axs[1].set_title('Flipped ECG-related IC')
        plt.show()
        """

        # compute and store ICA quality metrics (using ECG reference signal for epochs centered around actual QRS complexes)
        snr_ptp, snr_rssq = calulate_ICA_metrics_with_ecg_ref(ecg_ref_signal, ecg_related_component)
        ica_dict["snr_ptp"] = snr_ptp
        ica_dict["snr_rssq"] = snr_rssq
        
        seg_pTp_snrs_dB.append(snr_ptp)
        seg_rssq_snrs_dB.append(snr_rssq)

    return (ica_dicts, seg_computation_times_sec, seg_pTp_snrs_dB, seg_rssq_snrs_dB, seg_corrs_with_refECG, seg_ecg_related_ics)

# TODO REFACTOR THIS
def main():

    global componentAmountConsidered
    
    ########## load and select sample / window from test recordings (eeg and ecg channels seperately)
    # load offline (brainflow) recording sessions 
    dataRecordings = offD.get_offline_recording_sessions()


    ########################################
    

    ### perform offline ICA Analysis on loaded and (pre-) filtered (BrainFlow) EEG data of Test Recording No.1
    startTime_prepro = time.time()
    preproEEG = prepro.preprocess_data(dataRecordings[0].get_EEG_data())
    preproECG = prepro.preprocess_data(dataRecordings[0].get_ECG_data())
    endTime_prepro = time.time() - startTime_prepro

    startTime_objCreate = time.time()
    rawEEG = createObjectMNE('eeg', preproEEG)
    rawECG = createObjectMNE('ecg', preproECG)
    endTime_objCreate = time.time() - startTime_objCreate

    # Here we'll crop to 'tmax' seconds 
    # Note that the new tmin is assumed to be t=0 for all subsequently called functions
    # TODO Data Cropping auslagern wenn verschiebung der sample label ein problem
    #clean data
    rawEEG.crop(tmin = 200.0, tmax=260.0)
    rawECG.crop(tmin = 200.0, tmax=260.0)
    # noisy data (channel 6 dies in the end)
    #rawEEG.crop(tmin = 630.0, tmax=690.0)
    #rawECG.crop(tmin = 630.0, tmax=690.0)

    #choseBestICA(rawEEG, rawECG, componentAmountConsidered, "auto")

    #evaluate_all_ICA_variants(dataRecordings[0].get_EEG_data(), dataRecordings[0].get_ECG_data(), amountICs= componentAmountConsidered, maxIter = "auto")

    startTime_filter = time.time()
    # TODO Check if this is already done by BrainFlow Filtering 
    # TODO Note down that this is necessary (bandpass filter EEG before ICA to prevent slow downward shift) (?)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)
    endTime_filter = time.time() - startTime_filter

    print("Preprocessing (BrainFlow Filter + uV-V Conversion + RawObj Creation + additional MNE_filter)")
    print("Preprocessing (BrainFlow Filter + uV conversion) took (in sec): ", endTime_prepro)
    print("MNE RawObj Creation took (in sec) ", endTime_objCreate)
    print("Additional MNE Bandpass Filter took (in sec): ", endTime_filter)

    # apply ICA on the filtered raw EEG data

    startTime_ICA = time.time()
    # using picard
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='picard')

    # using fastICA
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='fastica')

    # using infoMax
    ica = ICA(n_components=componentAmountConsidered, max_iter="auto", random_state=97, method='infomax')
    
    ica.fit(filt_rawEEG)
    ica

    icaTime = time.time() - startTime_ICA

    print("ICA took (in sec): ", icaTime)
    print("ICA took in total:", ica.n_iter_, "iterations")

    # retrieve the fraction of variance in the original data 
    # that is explained by our ICA components in the form of a dictionary
    explained_var_ratio = ica.get_explained_variance_ratio(filt_rawEEG) 
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )

    # plot_sources will show the time series of the ICs. Note that in our call to plot_sources 
    # we can use the original, unfiltered Raw object
    filt_rawEEG.load_data()
    #ica.plot_sources(rawEEG, title = "ICA Sources" )

    
    rawResult = ica.get_sources(rawEEG)
    print("len (timepoints) of rawResult: ", len(rawResult))
    #rawResult.plot()
    rawECG.add_channels([rawResult])
    rawECG.plot(title = "rawECG (fused)")
    
    #rawComponents.plot()
    #rawResult.plot() 

    #comps = ica.get_sources(raw)
    #comps.plot()

    #ica.plot_components()

    # get all IC components for SNR Evaluation (each)
    resultData, resultTimes = rawECG[:]
    new_ecg = resultData[0][:]
    components = []
    for i in range(1, componentAmountConsidered + 1):
            components.append(resultData[i][:])
 

    # print SNR values for ECG-related component only (compared to reference ECG)
    correlations = []
    for curr_component in components:
            correlations.append(calculate_snr_correlation_wR_CGPT(new_ecg, curr_component))
    
    global ecg_related_index
    ecg_related_index = correlations.index(max(correlations))

    print("##########################")
    print("SNR results for the", (ecg_related_index+1), "th component")
    print("SNR:", calculate_snr_wR(new_ecg, components[ecg_related_index]), "dB")
    #print("SNR (RMS):", calculate_snr_RMS(components[0], components[i]), "dB")
    #print("SNR (peak-to-peak):", calculate_snr_peak_to_peak(components[i]))
    #print("SNR (wavelet):", calculate_snr_wavelet(components[i]))
    print("SNR (correlation with REF ECG):", calculate_snr_correlation_wR_CGPT(new_ecg, components[ecg_related_index]))
    #print("SNR (IDUN) for 1 comp:", calculate_snr_Idun(components[i]))
    print("##########################")

    # TEST if simple np.corrcoef is sufficient
    correlations_numpy = []
    for curr_component in components:
            correlations_numpy.append(np.abs(np.corrcoef(new_ecg, curr_component)[0, 1]))
    
    assert ecg_related_index == correlations_numpy.index(max(correlations_numpy))

    #identify_ecg_related_ic(new_ecg, components[ecg_related_index])
  


    
    # plot the extracted ecg related component
    fig, axs = plt.subplots(componentAmountConsidered + 1, 1, sharex = True)
    axs[0].plot(new_ecg)
    axs[0].set_title('clean ecg')

    for j in range(componentAmountConsidered):
        axs[j+1].plot(components[j])
        axs[j+1].set_title(str(j+1) + "-th component")

    plt.show()

    

    # employ NeuroKit Heartbeat Visualization and Quality Assessment
    # TODO über den plot mit markierten R-Peaks nochmal Herzrate drüberlegen für visuelle Kontrolle
    # TODO mit individual heartbeats plot spielen um guten Wert für epoch segmentierung zu finden (bzgl. SNR Formel Güler/PluxBio)

    ecg_related_comp = components[ecg_related_index] * calculate_corr_sign(new_ecg, components[ecg_related_index])
    nk_signals, nk_info = nk.ecg_process(ecg_related_comp, sampling_rate=sampling_rate)

    # Extract ECG-related IC and R-peaks location
    nk_rpeaks = nk_info["ECG_R_Peaks"]
    nk_cleaned_ecg = ecg_related_comp #nk_signals["ECG_Clean"]
    
    # assess ECG signal quality using nk_function ecg_quality()
    print("NK2. Signal Quality Assessment (Zhao2018, simple approach):", nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "zhao2018",  approach="simple"))
    print("NK2. Signal Quality Assessment (Zhao2018, fuzzy approach):", nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "zhao2018",  approach="fuzzy"))
    quality = nk.ecg_quality(nk_cleaned_ecg, sampling_rate=sampling_rate, method = "averageQRS")
    print("NK2. Signal Quality Assessment (QRS Avg):", np.mean(quality))

    #nk.signal_plot([nk_cleaned_ecg, quality], sampling_rate = sampling_rate, subplots = True, standardize=False)


    # Visualize R-peaks in ECG signal
    nk_plot = nk.events_plot(nk_rpeaks, nk_cleaned_ecg)
    
    # Plotting all the heart beats
    #nk_epochs = nk.ecg_segment(nk_cleaned_ecg, rpeaks=None, sampling_rate=sampling_rate, show=True)

    ###################
    
    heartbeats = extract_heartbeats(nk_cleaned_ecg, peaks=nk_rpeaks, sampling_rate=sampling_rate)
    heartbeats.head()

    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
    heartbeats_pivoted.head()
    '''
        # Prepare figure
        nk_fig, ax = plt.subplots()

        ax.set_title("Individual Heart Beats")
        ax.set_xlabel("Time (seconds)")

        # Aesthetics
        labels = list(heartbeats_pivoted)
        labels = ['Channel ' + x for x in labels] # Set labels for each signal
        cmap = iter(plt.cm.YlOrRd(np.linspace(0,1, int(heartbeats["Label"].nunique())))) # Get color map
        lines = [] # Create empty list to contain the plot of each signal

        for i, x, color in zip(labels, heartbeats_pivoted, cmap):
            line, = ax.plot(heartbeats_pivoted[x], label='%s' % i, color=color)
            lines.append(line)

        plt.show()
'''
    
    
    epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg, new_ecg)
    #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

    snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")

    snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    
    
if __name__ == "__main__":
    main()