import numpy as np

from brainflow.board_shim import BoardShim, BoardIds

import neurokit2 as nk
import mne

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


if __name__ == "__main__":
    recording = offD.RecordingSession(offD.getFilepaths()[1]) # choose recording to save

    rawEEG = createObjectMNE('eeg', prepro.preprocessData(recording.getEEG()))
    rawECG = createObjectMNE('ecg', prepro.preprocessData(recording.getECG()))

    t_offset = 200
    rawEEG.crop(tmin = t_offset)#, tmax = t_offset + 5 * 60)
    rawECG.crop(tmin = t_offset)#, tmax = t_offset + 5 * 60)

    eegs = rawEEG.get_data() # [channels, samples]
    ecg = rawECG.get_data()[0]
    # Normalize to (-1, 1)
    eegs /= np.max(np.abs(eegs))
    ecg /= np.max(np.abs(ecg))
    # save
    np.savez('dennis.npz', eegs=eegs, ecg=ecg, sampling_rate=sampling_rate)

    save_template = True
    if save_template:
        # simulate ideal ECG signal for (ideal) template
        ideal_template = nk.data(dataset="ecg_1000hz")
        ideal_template = nk.signal_resample(
            ideal_template,
            sampling_rate = 1000,
            desired_sampling_rate = sampling_rate
        )
        ideal_template = ideal_template[400:700] # extract a single beat
        np.save('ideal_beat.npy', ideal_template)

