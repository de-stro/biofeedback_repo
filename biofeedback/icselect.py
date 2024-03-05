import operator
import math 
import pywt
import numpy as np
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

    # Here we'll crop to 'tmax' seconds 
    rawEEG.crop(tmin = 180.0, tmax=240.0)
    rawECG.crop(tmin = 180.0, tmax=240.0)
    # apply bandpass filter(?) on EEG data (before ICA) to prevent slow downward(?) shift (?)
    filt_rawEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)

    # apply ICA on the filtered raw EEG data
    global componentAmountConsidered

    startTime_ICA = time.time()
    # using picard
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='picard')

    # using fastICA
    #ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='fastica')

    # using infoMax
    ica = ICA(n_components=componentAmountConsidered, max_iter=1, random_state=97, method='infomax')
    
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

    # get all IC components for SNR Evaluation (each)
    resultData, resultTimes = rawECG[:]
    new_ecg = resultData[0][:]
    components = []
    for i in range(1, componentAmountConsidered + 1):
            components.append(resultData[i][:])
 

    # print SNR values for ECG-related component only (compared to reference ECG)
    correlations = []
    for curr_component in components:
            correlations.append(icascript.calculate_snr_correlation(new_ecg, curr_component))
    
    global ecg_related_index
    ecg_related_index = correlations.index(max(correlations))

    print("##########################")
    print("SNR results for the", (ecg_related_index+1), "th component")
    print("SNR:", icascript.calculate_snr(new_ecg, components[ecg_related_index], ecg_related_index), "dB")
    #print("SNR (RMS):", calculate_snr_RMS(components[0], components[i]), "dB")
    #print("SNR (peak-to-peak):", calculate_snr_peak_to_peak(components[i]))
    #print("SNR (wavelet):", calculate_snr_wavelet(components[i]))
    print("SNR (correlation with REF ECG):", icascript.calculate_snr_correlation(new_ecg, components[ecg_related_index]))
    #print("SNR (IDUN) for 1 comp:", calculate_snr_Idun(components[i]))
    print("##########################")

    ################################# CALCULATE CORRELATION WITHOUT REF ECG ##################################
    ecg_template = nk.signal_resample(nk.data(dataset="ecg_1000hz"), desired_length = 1, sampling_rate = 1000, desired_sampling_rate = 250, method = "FFT")
    nk.signal_plot(ecg_template[0:10000], sampling_rate=1000)
    plt.show()

    no_ref_templ_correlations = []
    for curr_component in components:
            no_ref_templ_correlations.append(icascript.calculate_snr_correlation(new_ecg, curr_component))

    no_ref_auto_correlations = []
    for curr_component in components:
            no_ref_auto_correlations.append(icascript.calculate_snr_correlation(new_ecg, curr_component))

    no_ref_template_matching_correlations = []
    for curr_component in components:
            no_ref_template_matching_correlations.append(icascript.calculate_snr_correlation(new_ecg, curr_component))

    no_ref_ssp_correlations = []
    for curr_component in components:
            no_ref_ssp_correlations.append(icascript.calculate_snr_correlation(new_ecg, curr_component))


    
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

    
    ''' TODO Delete this
    epoch = icascript.epoch_for_QRS_and_noise(nk_cleaned_ecg, new_ecg)
    #epoch = epoch_for_QRS_and_noise(nk_cleaned_ecg)

    snr_ptp = icascript.calculate_snr_PeakToPeak_from_epochs(epoch[0], epoch[1])
    print("SNR (avg peak-to-peak-amplitude) is ", snr_ptp, "dB")

    snr_rssq = icascript.calc_snr_RSSQ_from_epochs(epoch[0], epoch[1])
    print("SNR (avg RSSQ) is ", snr_rssq, "dB")
    '''
    
    
if __name__ == "__main__":
    main()