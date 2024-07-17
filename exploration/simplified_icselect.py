import numpy as np
import scipy.signal as scisig
import sklearn.preprocessing as sklprep
import matplotlib.pyplot as plt
import time

from brainflow.board_shim import BoardShim, BoardIds

import neurokit2 as nk
import mne
from mne.preprocessing import ICA

import biofeedback.data.data_offline as offD
import biofeedback.data.preprocessing as prepro

board_id = BoardIds.CYTON_BOARD
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)[:-1] # drop last channel (as it is ECG)
ecg_channel = [BoardShim.get_ecg_channels(board_id).pop()] # last channel is ECG


def createObjectMNE(type: str, data_2D):
    """ Create MNE objects from brainflow data array """
    data_2D = data_2D / 1000000  # BrainFlow returns uV, convert to V for MNE!1!1!1!

    if type == 'eeg':
        ch_types = ['eeg'] * len(eeg_channels)
        info = mne.create_info(ch_names=len(eeg_channels), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)
    else:
        ch_types = ['ecg']
        info = mne.create_info(ch_names=len(ecg_channel), sfreq=sampling_rate, ch_types=ch_types)
        return mne.io.RawArray(data_2D, info)

def compute_cross_correlation(clean_signal, noisy_signal):
    """ Compute cross-correlation """
    cross_corr = scisig.correlate(clean_signal, noisy_signal, mode='full')

    # Normalize cross-correlation (using L2 Norm)
    # norm_cross_corr = cross_corr / (np.linalg.norm(clean_signal) * np.linalg.norm(noisy_signal))

    norm_cross_corr = sklprep.minmax_scale(cross_corr, feature_range= (-1, 1))
    akf_noisy = scisig.correlate(noisy_signal, noisy_signal, mode='same')
    fft_series = np.fft.fft(akf_noisy)
    fft_series = np.fft.fftshift(fft_series)
    fft_series = np.square(np.abs(fft_series))


    fig, axs = plt.subplots(5, 1)
    axs[0].plot(clean_signal, label = 'Clean Signal')
    axs[1].plot(noisy_signal, label = 'Noisy Signal')
    axs[2].plot(cross_corr, label = 'Cross-Correlation of Noisy')
    axs[3].plot(scisig.correlate(cross_corr, cross_corr, mode='same'), label = 'AKF of Cross-Corr')

    #axs[3].plot(np.square(np.abs(np.fft.fftshift(np.fft.fft(cross_corr)))), label = "FFT Cross-Corr")
    #axs[3].plot(akf_noisy, label = 'AKF of Noisy')
    #axs[3].plot(fft_series, label = 'FFT of akf noisy_signal')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    #plt.show()
    
    return np.max(np.abs(norm_cross_corr))

def main():
    ## Get data

    recording = offD.RecordingSession(offD.getFilepaths()[1])

    rawEEG = createObjectMNE('eeg', prepro.preprocessData(recording.getEEG()))
    rawECG = createObjectMNE('ecg', prepro.preprocessData(recording.getECG()))

    rawEEG.crop(tmin = 900.0, tmax=960.0)
    rawECG.crop(tmin = 900.0, tmax=960.0)

    # get user template from ECG recording (not simulated)
    # userTemplate = mne.io.Raw.copy(rawECG).crop(tmin=360.0, tmax=420.0)

    filteredEEG = rawEEG.copy().filter(l_freq=1.0, h_freq=50.0)

    ## Perform ICA
    
    ica = ICA(n_components=len(eeg_channels), max_iter='auto', random_state=97, method='picard')

    startTime = time.time()
    ica.fit(filteredEEG)
    fitTime = time.time() - startTime
    print(f'ICA fit time: {fitTime} s')

    rawResult = ica.get_sources(rawEEG.copy()) # TODO can we also *not* use Raw Obj Instance here?

    # TEST plot results (fusedECG) to visually check results (time series of ICs) of best ICA chosen
    # rawECG.add_channels([rawResult])
    # rawECG.plot(title = "rawECG (fused)")

    # extract components
    components = [component.flatten() for component in rawResult.get_data()]

    ## Create ideal templates

    # simulate ideal ECG signal for (ideal) template
    ideal_template = nk.data(dataset="ecg_1000hz")
    ideal_template = nk.signal_resample(
        ideal_template,
        sampling_rate = 1000,
        desired_sampling_rate = sampling_rate
    )
    # templates of 1 to 4 beats
    ideal_template_1B = ideal_template[400:700]
    ideal_template_2B = ideal_template[400:950]
    ideal_template_3B = ideal_template[400:1150]
    ideal_template_4B = ideal_template[400:1450]

    ## Compute cross-correlation with ideal template (simulated)
    
    print("####### RESULTS: Ideal_TEMPL_CROSSCORR_CGPT ###############")
    ideal_temp_crosscorr = []
    for idx, curr_component in enumerate(components):
         crosscorr = compute_cross_correlation(ideal_template_1B, curr_component)
         print("Cross-correlation of ", idx, "-th component (from 0!) is: ", crosscorr)
         ideal_temp_crosscorr.append(crosscorr)
    print("Component with max crosscorr is:",
          np.argmax(ideal_temp_crosscorr),
          "component (from 0!), with crosscorr of",
          np.max(ideal_temp_crosscorr)
    )
    
    ############## Compute Cross-Correlation with User Template (from ecg_ref) #############################
    # TODO
    ########################################################
    
    ############### AUTOCORRELATION APPROACH #################
    # TODO
    ########################################################

    plt.show()
    
if __name__ == "__main__":
    main()
