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

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG

flip = 1
ecg_related_index = 0
componentAmountConsidered = len(eeg_channels)

def calculate_snr_correlation(clean_signal, noisy_signal):
    correlation = np.corrcoef(clean_signal, noisy_signal)[0, 1]
    correlation = np.correlate(clean_signal, noisy_signal)
    correlation = correlation ** 2
    return correlation

"""
def calc_crosscorr(signal, noisy):
    # clean the signals first 
    mittelwert = (np.max(signal) + np.min(signal)) / 2
    normalized_signal = signal - mittelwert
    normalized_signal = normalized_signal / np.max(normalized_signal)

    mittelwert = (np.max(noisy) + np.min(noisy)) / 2
    normalized_noisy = noisy - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy)

    # determine if negative correlated
    similarity = normalized_noisy * normalized_signal
    normalized_noisy = normalized_noisy * np.sign(np.mean(similarity))
    

    correlation = scis.correlate(normalized_signal, normalized_noisy, mode = "full")
    corr = correlation / np.max(correlation)

    lags = scis.correlation_lags(len(normalized_signal), len(normalized_noisy), mode = "full")
    lag = lags[np.argmax(correlation)]

    print("Calculated Lag ", lag)
    print("Calculated Max Corr ", np.max(corr))

    return np.max(corr)
"""

def compute_cross_correlation(clean_signal, noisy_signal):
    # Compute cross-correlation
    cross_corr = scisig.correlate(clean_signal, noisy_signal, mode='full')
    double_cross = scisig.correlate(clean_signal, cross_corr, mode='full')
    triple_cross = scisig.correlate(clean_signal, double_cross, mode='full')
    fourth_cross = scisig.correlate(clean_signal, triple_cross, mode='full')

    # Normalize cross-correlation (using L2 Norm)
    # norm_cross_corr = cross_corr / (np.linalg.norm(clean_signal) * np.linalg.norm(noisy_signal))

    norm_cross_corr = sklprep.minmax_scale(cross_corr, feature_range= (-1, 1))
    akf_noisy = scisig.correlate(noisy_signal, noisy_signal, mode='same')
    fft_series = np.fft.fft(akf_noisy)
    fft_series = np.fft.fftshift(fft_series)
    fft_series = np.square(np.abs(fft_series))

    # employ minimalistic peak detection


    # normalize cross-correlation 
    fig, axs = plt.subplots(5, 1)
    axs[0].plot(clean_signal, label = 'Clean Signal')
    axs[1].plot(noisy_signal, label = 'Noisy Signal')
    axs[2].plot(cross_corr, label = 'Cross-Correlation of Noisy')
    axs[3].plot(scisig.correlate(cross_corr, cross_corr, mode='same'), label = 'AKF of Cross-Corr')
    axs[4].plot(fourth_cross, label = 'Fourth CrossCor of Noisy')

    #axs[3].plot(np.square(np.abs(np.fft.fftshift(np.fft.fft(cross_corr)))), label = "FFT Cross-Corr")
    #axs[3].plot(akf_noisy, label = 'AKF of Noisy')
    #axs[3].plot(fft_series, label = 'FFT of akf noisy_signal')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    #plt.show()
    
    #return norm_cross_corr
    return np.max(np.abs(norm_cross_corr))
    #return np.max(norm_cross_corr)


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
    dataRecordings.append(offD.RecordingSession(filepaths[1]))
    ########################################
    
    ### perform offline ICA Analysis on loaded and (pre-) filtered (BrainFlow) EEG data of Test Recording No.1
    rawEEG = icascript.createObjectMNE('eeg', prepro.preprocessData(dataRecordings[0].getEEG()))
    rawECG = icascript.createObjectMNE('ecg', prepro.preprocessData(dataRecordings[0].getECG()))

    # get user template from ECG recording (not simulated)
    userTemplate = mne.io.Raw.copy(rawECG).crop(tmin=360.0, tmax=420.0)


    # Here we'll crop to 'tmax- tmin' seconds (240-180 = 60 seconds)
    # TODO Rausfinden warum float_NaN to int conversion Fehler, wenn tmin 300 und tmax 360
    # TODO Rausfinden warum broadcast / shape error wenn 
    rawEEG.crop(tmin = 900.0, tmax=960.0)
    rawECG.crop(tmin = 900.0, tmax=960.0)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)

    # apply ICA on the filtered raw EEG data
    global componentAmountConsidered

    # perform ICA comparing the different algorithmic performance on the data (regarding pTp SNR of ECG-related IC)
    method_key, ica_dicts = icascript.choseBestICA(rawEEG, rawECG, amountICs = componentAmountConsidered, maxIter = "auto")
    bestICA = ica_dicts[method_key]

    # TEST plot results (fusedECG) to visually check results (time series of ICs) of best ICA chosen
    
    rawResult = bestICA["sources"]
    rawECG.add_channels([rawResult])
    rawECG.plot(title = "rawECG (fused)")

    # extract components to perform cross-correlation on
    components = []
    timeseries = rawResult.get_data()
    for component in timeseries:
        components.append(component.flatten())

    ecg_related_index = bestICA["ECG_related_index_from_0"]
    ecg_ref = (rawECG.get_data()).flatten()

    ############## Create Templates for Cross-Correlation ###############

    # simulate ideal ECG signal for (ideal) template
    ideal_template = nk.data(dataset="ecg_1000hz")
    #plt.figure()
    #plt.plot(np.square(np.abs(np.fft.fftshift(np.fft.fft(ideal_template)))), label = "FFT Template")
    ideal_template_resamp = nk.signal_resample(ideal_template, sampling_rate = 1000, desired_sampling_rate = sampling_rate)
    # TODO extract exactly one beat / two beats ... ACURATELY using neurokit methods (instead of magic numbers)
    ideal_template_1B = ideal_template_resamp[400:700]
    ideal_template_2B = ideal_template_resamp[400:950]
    ideal_template_3B = ideal_template_resamp[400:1150]
    ideal_template_4B = ideal_template_resamp[400:1450]
    #nk.signal_plot(ideal_template_1B, sampling_rate = sampling_rate)
    #plt.show()

    # TODO use ecg_ref to create individual user template
    # -> sinnvoll, wenn letzten endes eh ein anderer Herzschlag beim template matching verwendet wird?
    '''
    # create user ecg_template from copied Raw_ecg object
    templateData, resultTimes = userTemplate[:]
    user_ecg_template = templateData[0][:]
    #user_ecg_template = user_ecg_template[500:750]
    plt.plot(user_ecg_template)
    plt.show()
    '''
    #######################################################################

    ############## Compute Cross-Correlation with Ideal Template (simulated) #############################

    print("####### RESULTS: Ideal_TEMPL_CROSSCORR_CGPT ###############")
    ideal_temp_crosscorr_CGPT = []
    it = 0 
    ideal_temp_crosscorr_CGPT_timeBegin = time.time()
    for curr_component in components:
         crosscorr = compute_cross_correlation(ideal_template_1B, curr_component)
         print("(Ideal_Temp_CGPT) Cross-Correlation of ", it, "-th component (from 0!) is: ", crosscorr)
         ideal_temp_crosscorr_CGPT.append(crosscorr)
         it += 1
    ideal_temp_crosscorr_CGPT_timeTotal = time.time() - ideal_temp_crosscorr_CGPT_timeBegin
    print("Component with max crosscorr is: ", np.argmax(ideal_temp_crosscorr_CGPT), " component (from 0!), with crosscorr of ", np.max(ideal_temp_crosscorr_CGPT))
    print("Ideal_TEMPL_CROSSCORR_CGPT took ", ideal_temp_crosscorr_CGPT_timeTotal, " seconds in total")

    plt.show()
    
    ############## Compute Cross-Correlation with User Template (from ecg_ref) #############################
    '''
    user_template_correlation_CGPT = []
    it = 0 
    user_corr_CGPT_timeBegin = time.time()
    for curr_component in components:
         corr = compute_cross_correlation_CGPT(user_ecg_template, curr_component)
         print("(USER_Template_CGPT) Cross-Correlation of ", it + 1, "-th component is: ", corr)
         user_template_correlation_CGPT.append(corr)
         it += 1
    user_corr_CGPT_time_total = time.time() - user_corr_CGPT_timeBegin
    print("Component with max corr_CGPT calculated is: ", np.argmax(user_template_correlation_CGPT) + 1, " component")
    print("with cross correlation of ", np.max(user_template_correlation_CGPT))
    print("CGPT_Corr took ", user_corr_CGPT_time_total, " seconds in total")
    '''

    ########################################################
    
    ############### AUTOCORRELATION APPROACH #################
    ''' TODO Auto-Corr Ansatz prüfen (ob als invalide Alternative) oder omitten
    no_ref_auto_correlations = []
    it = 0 
    auto_corr_timeBegin = time.time()
    autocorr_threshold = 0.4
    for curr_component in components:
        # this is basically scipy correlation of the signal with itself
        # TODO Find out appropriate time lag (duration of one heart beat??) to correlate against
        #autocorr, info = nk.signal_autocor(curr_component, lag = 250, show = True)

        autocorr = scis.correlate(curr_component, curr_component, mode = "full")
        # normalize autocorrelation values 
        autocorr /= np.max(autocorr)

        corr_peaks = np.where(autocorr > autocorr_threshold)[0]
        no_ref_auto_correlations.append(corr_peaks.size)

        #print("(NK) Auto)-Correlation of ", it + 1, "-th component is: ", autocorr)
        #no_ref_auto_correlations.append(autocorr)
        it += 1
    auto_corr_time_total = time.time() - auto_corr_timeBegin

    print("Component with max (NK) Auto-Corr PEAKS calculated is: ", np.argmax(no_ref_auto_correlations) + 1, " component")
    print("with auto correlation PEAKS of ", np.max(no_ref_auto_correlations))
    print("NK auto corr took ", auto_corr_time_total, " seconds in total")
    '''

    ########################################################
    #######################################################
    '''
    it = 0 
    templ_corr_timeBegin = time.time()
    no_ref_templ_correlations = []
    for curr_component in components:
        snr_corr = calculate_snr_correlation(user_ecg_template, curr_component)
        print("(USER_Template_CGPT) SNR-Correlation of ", it + 1, "-th component is: ", snr_corr)
        no_ref_templ_correlations.append(snr_corr)
        it += 1
    templ_corr_time_total = time.time() - templ_corr_timeBegin
    print("Component with max (CGPT) User Template SNR-Corr calculated is: ", np.argmax(no_ref_templ_correlations) + 1, " component")
    print("with snr correlation of ", np.max(no_ref_templ_correlations))
    print("(CGPT) SNR corr took ", templ_corr_time_total, " seconds in total")
    '''

def identify_ecg_component_with_ref(ecg_ref_signal, ic_signals):
    """identify_ecg_component_with_ref _summary_TODO

    Args:
        ecg_ref_signal (_type_): _description_
        ic_signals (_type_): _description_

    Returns:
        _type_: _description_
    """
    flipped = 0
    # compute correlation of components with ECG reference to identify ECG-related IC
    correlations_with_ref = []
    for ind_component in ic_signals:
        correlations_with_ref.append(np.corrcoef(ecg_ref_signal, ind_component)[0, 1])
    
    # identify component with highest correlation to reference ECG (i.e. ECG-related IC)
    ecg_related_index = np.argmax(np.abs(correlations_with_ref))
    ecg_related_component = ic_signals[ecg_related_index]
    ecg_related_correlation = correlations_with_ref[ecg_related_index]

    # if negatively correlated, flip the ECG-related component around and keep track of flip
    if ecg_related_correlation < 0:
        ecg_related_component *= -1
        flipped = 1

    # calculate distance of ecg_related_correlation (highest abs corr value) to second highest corr value
    correlations_with_ref.pop(ecg_related_index)
    second_highest_corr_index = np.argmax(np.abs(correlations_with_ref))
    second_highest_corr = correlations_with_ref[second_highest_corr_index]

    distance_to_rest_corrs = abs(ecg_related_correlation) - abs(second_highest_corr)

    """ TEST: if ECG-related component was correctly identified
    print("IC_SELECT TEST: ECG_RELATED INDEX")
    print("ECG-related Component calulated: ", ecg_related_index, "-th component (starting from 0!)")
    print("with correlation value of: ", correlation)
    #### TEST VISUALLY ########################################################
    # Plot the sources / results to visually control calculated ECG_related_component
    fig, axs = plt.subplots(len(ic_signals) + 1, 1, sharex = True)
    axs[0].plot(ecg_ref_signal)
    axs[0].set_title('ECG Reference Signal')

    for idx, component in enumerate(ic_signals):
        axs[idx+1].plot(component)
        axs[idx+1].set_title(str(idx) + "-th independent component")
    plt.show()
    #### TEST VISUALLY ########################################################
    """

    return (ecg_related_component, flipped, ecg_related_index, abs(ecg_related_correlation), distance_to_rest_corrs)

    
if __name__ == "__main__":
    main()