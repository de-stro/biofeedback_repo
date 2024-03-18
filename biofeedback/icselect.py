import operator
import math 
import pywt
import numpy as np
import scipy.signal as scis
import time 
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

import data.offlinedata as offD
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
    #correlation = np.corrcoef(clean_signal, noisy_signal)[0, 1]
    correlation = np.correlate(clean_signal, noisy_signal)
    correlation = correlation ** 2
    return correlation

''' TODO Delete
def calc_cross_cor_stackof(signal, noisy):
    "Plot cross-correlation (full) between two signals."
    # clean signal and noise first
    mittelwert = (np.max(signal) + np.min(signal)) / 2
    normalized_signal = signal - mittelwert
    normalized_signal = normalized_signal / np.max(normalized_signal)

    mittelwert = (np.max(noisy) + np.min(noisy)) / 2
    normalized_noisy = noisy - mittelwert
    normalized_noisy = normalized_noisy / np.max(normalized_noisy)

    # determine if negative correlated
    similarity = normalized_noisy * normalized_signal
    normalized_noisy = normalized_noisy * np.sign(np.mean(similarity))

    signal = normalized_signal
    noisy = normalized_noisy

    #########################################

    N = max(len(signal), len(noisy)) 
    n = min(len(signal), len(noisy)) 

    if N == len(noisy): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 

    #c = scis.correlate(signal / np.std(signal), noisy / np.std(noisy), 'full')
    c = scis.correlate(signal, noisy, 'full') 
    correlation_value = np.max(c / n) # (?) -> TODO Check if true
    print("Calculated correlation (normalized in [0,1]) ", np.max(c / n))
    plt.plot(lags, c / n) 
    plt.show() 

    return correlation_value
    #return np.corrcoef(signal, noisy)[0, 1]
'''

'''
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
'''

def compute_cross_correlation_CGPT(clean_signal, noisy_signal):
    # Compute cross-correlation
    cross_corr = scis.correlate(clean_signal, noisy_signal, mode='full')
    
    # Normalize cross-correlation (using L2 Norm)
    norm_cross_corr = cross_corr / (np.linalg.norm(clean_signal) * np.linalg.norm(noisy_signal))
    
    #return norm_cross_corr
    return np.max(np.abs(norm_cross_corr))


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
    rawEEG.crop(tmin = 900.0, tmax=960.0)
    rawECG.crop(tmin = 900.0, tmax=960.0)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)

    # apply ICA on the filtered raw EEG data
    global componentAmountConsidered

    startTime_ICA = time.time()
    # using picard
    #ica = ICA(n_components=componentAmountConsidered, max_iter="auto", random_state=97, method='picard')

    # using fastICA
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='fastica')

    # using infoMax
    ica = ICA(n_components=componentAmountConsidered, max_iter="auto", random_state=97, method='infomax')
    
    ica.fit(filt_rawEEG)
    ica

    icaTime = time.time() - startTime_ICA

    print("ICA took (in sec): ", icaTime)
    print("ICA took in total:", ica.n_iter_, "iterations")

    # plot_sources will show the time series of the ICs. Note that in our call to plot_sources 
    # we can use the original, unfiltered Raw object
    rawResult = ica.get_sources(rawEEG)
    print("len (timepoints) of rawResult: ", len(rawResult))
    #rawResult.plot()
    rawECG.add_channels([rawResult])
    rawECG.plot(title = "rawECG (fused)")

    # get all IC components to calculate correlation (including REF ECG)
    resultData, resultTimes = rawECG[:]
    new_ecg = resultData[0][:]
    components = []
    for i in range(1, componentAmountConsidered + 1):
            components.append(resultData[i][:])
 

    # print SNR values for ECG-related component only (compared to reference ECG)
    correlations = []
    for curr_component in components:
            correlations.append(calculate_snr_correlation(new_ecg, curr_component))
    
    global ecg_related_index
    ecg_related_index = correlations.index(max(correlations))

    print("##########################")
    print("SNR results for the", (ecg_related_index+1), "th component (with REF ECG)")
    print("SNR (correlation with REF ECG):", calculate_snr_correlation(new_ecg, components[ecg_related_index]))
    print("SNR (with REF ECG):", icascript.calculate_snr_wR(new_ecg, components[ecg_related_index], ecg_related_index), "dB")

    epoch = icascript.epoch_for_QRS_and_noise(components[ecg_related_index], new_ecg)
    snr_ptp = icascript.calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")
    
    snr_rssq = icascript.calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    print("##########################")

    ################ CALCULATE CROSS CORRELATION WITH REF SIGNAL TO DETERMINE ECG-RELATED IC ##############
    '''
    cross_correlations = []
    for curr_component in components:
         corr = calc_cross_cor_stackof(new_ecg, curr_component)
         cross_correlations.append(corr)
    print("Component with max corr calculated is: ", np.argmax(cross_correlations) + 1, " component")
    print("with cross correlation of ", np.max(cross_correlations))
    '''

    
    cross_corr_CGPT = []
    it = 0
    for curr_component in components:
         corr = compute_cross_correlation_CGPT(new_ecg, curr_component)
         print("(CGPT) Cross-Correlation of ", it + 1, "-th component is: ", corr)
         cross_corr_CGPT.append(corr)
         it += 1
    
    print("Component with max corr_CGPT calculated is: ", np.argmax(cross_corr_CGPT) + 1, " component")
    print("with cross correlation of ", np.max(cross_corr_CGPT))
    


    ################################# CALCULATE CORRELATION WITHOUT REF ECG ##################################

    # Create accurate ecg template signal to cross correlate noisy signal with
    '''
    ecg_template = nk.data(dataset="ecg_1000hz")
    ecg_template_resamp = nk.signal_resample(ecg_template, sampling_rate = 1000, desired_sampling_rate = sampling_rate)
    ecg_template = ecg_template_resamp[400:650]
    nk.signal_plot(ecg_template, sampling_rate = sampling_rate)
    plt.show()

    template_correlation_CGPT = []
    it = 0 
    corr_CGPT_timeBegin = time.time()
    for curr_component in components:
         corr = compute_cross_correlation_CGPT(ecg_template, curr_component)
         print("(Template_CGPT) Cross-Correlation of ", it + 1, "-th component is: ", corr)
         template_correlation_CGPT.append(corr)
         it += 1
    corr_CGPT_time_total = time.time() - corr_CGPT_timeBegin
    print("Component with max corr_CGPT calculated is: ", np.argmax(template_correlation_CGPT) + 1, " component")
    print("with cross correlation of ", np.max(template_correlation_CGPT))
    print("CGPT_Corr took ", corr_CGPT_time_total, " seconds in total")
    '''

    ###################################################
    #####################################################

    # create user ecg_template from copied Raw_ecg object
    templateData, resultTimes = userTemplate[:]
    user_ecg_template = templateData[0][:]
    #user_ecg_template = user_ecg_template[500:750]
    plt.plot(user_ecg_template)
    plt.show()

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

    ########################################################
    #######################################################
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
    
    # plot the extracted ecg related components
    fig, axs = plt.subplots(componentAmountConsidered + 1, 1)
    axs[0].plot(new_ecg)
    axs[0].set_title('clean ecg')

    for j in range(componentAmountConsidered):
        axs[j+1].plot(components[j])
        axs[j+1].set_title(str(j+1) + "-th component")

    plt.show()

    

    # employ NeuroKit Heartbeat Visualization and Quality Assessment
    # TODO über den plot mit markierten R-Peaks nochmal Herzrate drüberlegen für visuelle Kontrolle
    # TODO mit individual heartbeats plot spielen um guten Wert für epoch segmentierung zu finden (bzgl. SNR Formel Güler/PluxBio)

    ecg_related_comp = components[ecg_related_index] * flip
    nk_signals, nk_info = nk.ecg_process(ecg_related_comp, sampling_rate=sampling_rate)

    # Extract ECG-related IC and R-peaks location
    nk_rpeaks = nk_info["ECG_R_Peaks"]
    nk_cleaned_ecg = ecg_related_comp #nk_signals["ECG_Clean"]

    #nk.signal_plot([nk_cleaned_ecg, quality], sampling_rate = sampling_rate, subplots = True, standardize=False)

    # Visualize R-peaks in ECG signal
    nk_plot = nk.events_plot(nk_rpeaks, nk_cleaned_ecg)
    
    # Plotting all the heart beats
    #nk_epochs = nk.ecg_segment(nk_cleaned_ecg, rpeaks=None, sampling_rate=sampling_rate, show=True)

    ###################
    
    
if __name__ == "__main__":
    main()