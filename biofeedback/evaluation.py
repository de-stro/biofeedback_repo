import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utility
import data.data_offline as data_off
import ica

segment_length_sec = 60

# ica parameters
ic_amount_to_consider = 7
ica_max_iterations = 10

def main():


    segmented_recording_sessions = []

    # load offline (brainflow) recording sessions 
    for recording_session in data_off.get_offline_recording_sessions():

        sampling_rate = recording_session.get_description()["sampling_rate"]
        # TODO does it make difference if ECG is also processed or not?
        session_data = recording_session.get_preprocessed_data()
        # session length in sec is amount of samples divided by sampling rate (in Hz)
        recording_length_samples = session_data[0].size

        # segment the session data into equally long segments
        segment_length_samples = segment_length_sec * sampling_rate
        #segment_amount = recording_length_samples // segment_length_samples
        segment_amount = utility.ceildiv(recording_length_samples, segment_length_samples)
        # split the session data column-wise (along second axis), NOTE: this returns VIEWs of the original array
        segments = np.split(session_data, range(segment_length_samples, (segment_amount)*segment_length_samples, segment_length_samples), axis=1)

        # omit last segment if shorter than specified segment_length
        if (segments[-1])[0].size < segment_length_samples:
            segments.pop()

        segmented_recording_sessions.append((recording_session, segments))
    
    """ TODO TEST For test purposes only
    for (session, segments) in segmented_recording_sessions:
        for idx, segment in enumerate(segments):

            data_frame = pd.DataFrame(np.transpose(segment))
            print('Data From the File')
            print(data_frame.head(61))

            # plot the segment
            fig, axs = plt.subplots(len(segment) - 2, 1, sharex = True)

            for j in range(2, len(segment)):
                axs[j-2].plot(segment[j, :])
                axs[j-2].set_title(str(j-2) + "-th channel (" + str(idx) + "-th segment)")

            plt.show()
    """

    # setup data and metric matrices to store results in
    ica_matrix = ...
    ic_select_matrix = ...
    peak_detect_matrix = ...


    # employ ICA on every segment of every recording session

    for (recording_session, segments) in segmented_recording_sessions:
        for segment in segments:

            # TODO Evaluate FIRSTLY ON 1 RECORDING ONLY
            ecg_signal = segment[2]
            eeg_signals = segment[3:]
            ica.evaluate_all_MNE_ICA_on_segment(ecg_signal, eeg_signals, component_amount=7, max_iterations=50)

   
    # evaluate all ICA variants on 2D data of ECG (only 1 timeseries) and 2D data of EEG
    #ica.evaluate_all_ICA_variants(segment_eeg, segment_ecg, amountICs=7, maxIter=10)



     

    # preprocess data appropriately for ICA

    # select ECG-related component (resulting from ICA)

    # employ R-peak detection on timeseries of ECG-related IC

    # evaluate feasibility of all combinations of MNE.ICA x Neurokit.Peak-Detection-Methods 

if __name__ == "__main__":
    main()