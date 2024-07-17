"""
    Module description
"""

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


def compute_cross_correlation(clean_signal, noisy_signal):
    """compute_cross_correlation computes and visualizes cross correlations between the independent components
    and reference ECG

    Args:
        clean_signal (_type_): _description_
        noisy_signal (_type_): _description_

    Returns:
        _type_: _description_
    """
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



def identify_ecg_component_with_ref(ecg_ref_signal, ic_signals):
    """identify_ecg_component_with_ref identifies the ECG-related IC using the reference ECG as ground truth

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

    """ FOR TEST PURPOSES ONLY
    TEST: if ECG-related component was correctly identified
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
