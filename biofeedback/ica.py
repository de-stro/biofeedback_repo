"""
    script to perform offline ICA Analysis in MNE on local (earlier recorded) EEG-Session (with parallel chest-ECG on
    channel 8, in BrainFlow Format (CSV)
"""

import math 
import numpy as np
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BoardIds

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



def extract_heartbeats(cleaned, peaks, sampling_rate=None): 
    """extract_heartbeats creates epochs based on / around R-peak locations

    Args:
        cleaned (_type_): _description_
        peaks (_type_): _description_
        sampling_rate (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    heartbeats = nk.epochs_create(cleaned, 
                                  events=peaks, 
                                  epochs_start=-0.3, 
                                  epochs_end=0.4, 
                                  sampling_rate=sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    return heartbeats

def createObjectMNE(type: str, data_2D):
    """createObjectMNE creates MNE objects from brainflow data array

    Args:
        type (_type_): _description_
        data_2D (_type_): _description_

    Returns:
        _type_: _description_
    """

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
    """calculate_corr_sign calculates correlation between clean and noisy signal,
    as degree of linear similarity

    Args:
        ref_signal (_type_): _description_
        noisy_signal (_type_): _description_

    Returns:
        _type_: _description_
    """
    mittelwert = (np.max(ref_signal) + np.min(ref_signal)) / 2
    normalized_ref = ref_signal - mittelwert
    normalized_ref = normalized_ref / np.max(normalized_ref)

    mittelwert = (np.max(noisy_signal) + np.min(noisy_signal)) / 2
    normalized_noisy = noisy_signal - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy) 

    similarity = normalized_noisy * normalized_ref

    return np.sign(np.mean(similarity))



def calculate_snr_wR(clean_signal, noisy_signal):
    """calculate_snr_wR calculates signal to noise ratio of ATE-ECG using the reference ECG
    as clean signal

    Args:
        clean_signal (_type_): _description_
        noisy_signal (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    
    # prefactor is 10 since we express signal and noise in power
    noise_power = np.mean(np.square(noise))
    snr = 10 * np.log10(clean_power / noise_power)
    return snr



def epoch_for_QRS_and_noise(noisy_signal, reference_signal = None):
    """epoch_for_QRS_and_noise segments the data into epochs around QRS complexes detected
    """

    naive_peak_correction = True

    # if passed, use reference signal for delineation (to mark QRS-complexes etc) 
    if reference_signal is not None:
        # detect R-peaks in reference signal
        ref_peaks_signal, ref_peaks_info = nk.ecg_peaks(reference_signal, sampling_rate = sampling_rate, 
                                                        method = 'neurokit', correct_artifacts = True, show = False)
        
        peak_locations = ref_peaks_info["ECG_R_Peaks"]
        # additionally employ "naive" peak correction, by checking if peak sample has highest value compared to 
        # surrounding samples (only reasonable for clean ref ecg with no noisy peaks inbetween R-peaks)
        # NOTE Appropriate check range for peak correction has to be determined
        if naive_peak_correction:
            peak_locations = utility.naive_peak_value_surround_checks(reference_signal, peak_locations, 10)
            
            """FOR TEST PURPOSES ONLY
            TEST VISUAL
            nk.events_plot(corrected_peak_locs, reference_signal)
            plt.show()
            """

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
    
    # assert that invariants and padding premise hold true
    # (onsets snd offset need same length, even if NaN padding and NaN padding one element
    # at the end and/or the beginning (at MAX))
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
    
    """ TODO OMIT THIS
    # TODO Delete: remove NaN padding introduced by neurokit2 ecg_delineation
    #onsets_R_list = utility.remove_NaN_padding(onsets_R_list)
    #offsets_R_list = utility.remove_NaN_padding(offsets_R_list)

    # ensure that epochs beginn with first R_Offset (hence, a noise_segment) and end with last R_Offset,
    # therefore, last detected QRS complex is included, while first QRS complex might be omitted!
    
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
    
    """ FOR TEST PURPOSES ONLY
    # now size offsets_R_list must be equal or 1 greater than size onsets_R_list!
    print("TEST:: Size of onset_R_list ", len(onsets_R_list), " and of offset_R_list: ", len(offsets_R_list))
    print("TEST:: OFFSET_R_LIST MUST BE EQUAL OR 1 GREATER THAN THE SIZE OF ONSET_R_LIST !!")
    print("$§$§$§$§$§$§$§$§$§$§$§$$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$§$")
    """

    # calculate duration between R_offset and R_onset in seconds (omit last R_Offset for noise epochs)  
    # but if onsets starts with NaN, also prune onsets (to truly calculate noise epochs duration)
    noise_epochs_duration = onsets_R_list - (offsets_R_list[:-1])
    noise_epochs_duration = noise_epochs_duration / sampling_rate

    # calculate duration between R_onset and R_offset in seconds (omit first R_Offset for QRS epochs)
    qrs_epochs_duration = (offsets_R_list[1:]) - onsets_R_list
    qrs_epochs_duration = qrs_epochs_duration / sampling_rate
    
    """FOR TEST PURPOSES ONLY
    print("TEST:: Length of epochs_duration arrays must be the same!!!")
    print("TEST::Length of qrs_epochs_duration: ", qrs_epochs_duration.size, " and of noise_epochs_duration: ", noise_epochs_duration.size)
    ''' TEST'''
    print("mean duration for QRS complex (in sec): ", np.mean(qrs_epochs_duration))
    print("duration of 10th QRS complex (in sec): ", ((waves["ECG_R_Offsets"])[9] - (waves["ECG_R_Onsets"])[9]) / sampling_rate)
    """

    # signal delineation to create epochs of noisy signal around QRS-complexes found 
    qrs_epochs = nk.epochs_create(noisy_signal, events = onsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = qrs_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
    
    # do the same epoching for epochs of noise parts of noisy signal (all except for QRS complexes)
    noise_epochs = nk.epochs_create(noisy_signal, events = offsets_R_list, sampling_rate = sampling_rate, epochs_start = 0, 
                                epochs_end = noise_epochs_duration.tolist(), event_labels = None, event_conditions = None, baseline_correction = False)
        
    """ FOR TEST PURPOSES ONLY 
    # Plot the whole signal (QRS + according noise segment), also QRS and noise separate
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




def calculate_snr_PeakToPeak_from_epochs(qrs_epochs, noise_epochs):
    """calculate_snr_PeakToPeak_from_epochs Calculate SNR of noisy signal epoched around 
    heart beats / R-peaks detected (in noisy signal, no reference)

    Args:
        qrs_epochs (_type_): _description_
        noise_epochs (_type_): _description_

    Returns:
        _type_: _description_
    """

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
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power (therefore prefactor of 20)
    snr = 20 * np.log10(qrs_pTp_amplitude_avg / noise_pTp_amplitude_avg)
    return snr


def calc_snr_RSSQ_from_epochs(qrs_epochs, noise_epochs):
    """calc_snr_RSSQ_from_epochs calculates SNR based on RSSQ approach (see SotA2, Guler et al.)

    Args:
        qrs_epochs (_type_): _description_
        noise_epochs (_type_): _description_

    Returns:
        _type_: _description_
    """

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
    # when signal and noise are measured in volts (V) or amperes (A) (measures of amplitude) 
    # they must first be squared to obtain a quantity proportional to power (therefore prefactor of 20)
    snr = 20 * np.log10(qrs_RSSQ_avg / noise_RSSQ_avg)
    return snr


def calulate_ICA_metrics_with_ecg_ref(ecg_ref_signal, ecg_related_component):
    """calulate_ICA_metrics_with_ecg_ref calculates all metrics using the reference ECG as ground truth

    Args:
        ecg_ref_signal (_type_): _description_
        ecg_related_component (_type_): _description_

    Returns:
        _type_: _description_
    """

    # segment the signal into indivdual heartbeats to calculate pTp SNR / RSSQ SNR
    epoch = epoch_for_QRS_and_noise(ecg_related_component, ecg_ref_signal)

    snr_ptp = calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")

    snr_rssq = calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    
    return (snr_ptp, snr_rssq)


def evaluate_all_MNE_ICA_on_segment(ecg_ref_signal, eeg_signals, component_amount, max_iterations):
    """evaluate_all_MNE_ICA_on_segment profiles all ICA algorithms on the observation segment

    Args:
        ecg_ref_signal (_type_): _description_
        eeg_signals (_type_): _description_
        component_amount (_type_): _description_
        max_iterations (_type_): _description_

    Returns:
        _type_: _description_
    """    
    

    # create mne.io.Raw Objects from input eeg data to perform MNE ICA on
    start_rawObj_create_time = time.time()
    mne_rawEEG = createObjectMNE('eeg', eeg_signals)
    rawObj_create_time = time.time() - start_rawObj_create_time

    # highpass filter the data using MNE filter
    start_mne_filter_time = time.time()
    mne_filtEEG = mne_rawEEG.copy().filter(l_freq=1.0, h_freq=None)
    mne_filter_time = time.time() - start_mne_filter_time

    #mne_rawECG = createObjectMNE('ecg', ecg_ref_signal.reshape(1, ecg_ref_signal.size))
    #mne_filtECG = mne_rawECG.copy().filter(l_freq=1.0, h_freq=None)
    #ecg_ref_signal = mne_rawECG.get_data().flatten()

    """FOR TEST PURPOSES ONLY 
    mne_object_length = ((((createObjectMNE('ecg', ecg_ref_signal.reshape(1, ecg_ref_signal.size))).get_data()).flatten())).size
    assert ecg_ref_signal.size == mne_object_length, "MNE alters ECG timeseries length"
    """

    ######################################################################
    # compute and compare the different MNE ICA algorithm implementations
    ######################################################################
    
    # setup list of ica dicts that store the metrics
    ica_dicts = [

        # similar to infomax (according to MNE https://mne.tools/stable/generated/mne.preprocessing.ICA.html)
        {"method_id": "picard", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='picard', fit_params=dict(ortho=False, extended=False))},  
        {"method_id": "picard_o", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='picard', fit_params=dict(ortho=True, extended=False))},
        # similar to extended Infomax (accord to MNE)
        {"method_id": "ext_picard", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='picard', fit_params=dict(ortho=False, extended=True))},
        # similar to FASTICA according to MNE
        {"method_id": "ext_picard_o", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='picard', fit_params=dict(ortho=True, extended=True))},  
        {"method_id": "infomax", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='infomax')},
        {"method_id": "ext_infomax", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='infomax', fit_params=dict(extended=True))},
        # the same as ICA_parallel {"method_id": "fastica", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='fastica')},
        {"method_id": "fastica_par", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='fastica', fit_params=dict(algorithm='parallel'))},
        {"method_id": "fastica_defl", "ica": ICA(n_components=component_amount, max_iter=max_iterations, random_state=97, method='fastica', fit_params=dict(algorithm='deflation'))}
    ]

    # setup list to store the ecg_related timeseries for each MNE ICA variant
    seg_ecg_related_ics = []

    # setup lists for ica metrics (numerical values)
    seg_fitting_times_sec = []
    seg_filter_times_sec = []
    seg_mne_createRawObj_times_sec = []
    seg_actual_iterations_amount = []
    seg_seconds_per_iteration = []
    seg_pTp_snrs_dB = []
    seg_rssq_snrs_dB = []
    seg_pTp_snrs_dB_refECG = []
    seg_rssq_snrs_dB_refECG = []
    seg_corrs_with_refECG = []
    seg_dist_to_restCorrs = []
    seg_variances_explained = []

    # fit the MNE ica objects and time the fitting
    for ica_dict in ica_dicts:

        eeg_fit_copy = mne_filtEEG.copy() #mne_rawEEG.copy()

        start_fit_time = time.time()
        ica_dict["ica"].fit(eeg_fit_copy)
        fit_time = time.time() - start_fit_time

        # store fitting time and Raw Obj creation time and MNE Filtering time
        ica_dict["time_fit"] = fit_time
        ica_dict["rawObj_create_time"] = rawObj_create_time
        ica_dict["mne_filter_time"] = mne_filter_time
     
        seg_fitting_times_sec.append(fit_time)
        seg_filter_times_sec.append(mne_filter_time)
        seg_mne_createRawObj_times_sec.append(rawObj_create_time)

        # extract the amount of iterations actually performed (until algorithm converged, if lower than maxIter)
        iteration_amount = ica_dict["ica"].n_iter_
        seg_actual_iterations_amount.append(iteration_amount)

        # calculate the amount of seconds needed (to fit ICA) for one iteration
        seg_seconds_per_iteration.append(fit_time / iteration_amount)

        # store the resulting independent components
        sources = ica_dict["ica"].get_sources(mne_rawEEG.copy()) 
        ic_timeseries, times = sources.get_data(return_times=True)
        ic_timeseries.flatten()
        
        ica_dict["independent_components"] = ic_timeseries

        # identify the ecg-related independent component (by correlating with ecg reference)
        ecg_related_component, flipped, ecg_related_index, corr_value, dist_to_rest_corrs = icselect.identify_ecg_component_with_ref(ecg_ref_signal, ic_timeseries)
        seg_ecg_related_ics.append(ecg_related_component)
        seg_corrs_with_refECG.append(corr_value)
        seg_dist_to_restCorrs.append(dist_to_rest_corrs)

        # retrieve the proportion of data variance explained by the EGG-related component 
        # (of the original, ica input data)
        variance_dict = ica_dict["ica"].get_explained_variance_ratio(inst = mne_filtEEG, components = [ecg_related_index], ch_type= 'eeg')
        seg_variances_explained.append(variance_dict["eeg"])

        # compute and store ICA quality metrics (using ECG reference signal for epochs centered around actual QRS complexes)
        snr_ptp_component, snr_rssq_component = calulate_ICA_metrics_with_ecg_ref(ecg_ref_signal, ecg_related_component)
        ica_dict["snr_ptp"] = snr_ptp_component
        ica_dict["snr_rssq"] = snr_rssq_component
        
        seg_pTp_snrs_dB.append(snr_ptp_component)
        seg_rssq_snrs_dB.append(snr_rssq_component)

        # calculate pTp and rssq SNR of reference chest-ECG for comparison
        snr_ptp_reference, snr_rssq_reference = calulate_ICA_metrics_with_ecg_ref(ecg_ref_signal, ecg_ref_signal)
        seg_pTp_snrs_dB_refECG.append(snr_ptp_reference)
        seg_rssq_snrs_dB_refECG.append(snr_rssq_reference)

        seg_SNRs_dB_reference = (seg_pTp_snrs_dB_refECG, seg_rssq_snrs_dB_refECG)

    seg_computation_times_sec = (seg_mne_createRawObj_times_sec, seg_filter_times_sec, seg_fitting_times_sec)

    return (ica_dicts, seg_computation_times_sec, seg_actual_iterations_amount, seg_seconds_per_iteration, seg_pTp_snrs_dB, seg_rssq_snrs_dB, seg_SNRs_dB_reference, seg_corrs_with_refECG, seg_dist_to_restCorrs, seg_variances_explained, seg_ecg_related_ics)
