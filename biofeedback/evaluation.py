import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import utility
import data.data_offline as data_off
import ica
import peak_detect

SEGMENT_LENGTH_SEC = 60
ICA_VAR_TO_EVAL = 3
PEAK_DETECT_VAR_TO_EVAL = 12

# ica parameters
IC_AMOUNT_TO_CONSIDER = 7
ICA_MAX_ITERATIONS = 10

def main():

    start_segmentation_time = time.time()

    segmented_recording_sessions = []

    # load offline (brainflow) recording sessions 
    for recording_session in data_off.get_offline_recording_sessions():

        sampling_rate = recording_session.get_description()["sampling_rate"]
        # TODO does it make difference if ECG is also processed or not?
        session_data = recording_session.get_preprocessed_data()
        # session length in sec is amount of samples divided by sampling rate (in Hz)
        recording_length_samples = session_data[0].size

        # segment the session data into equally long segments
        segment_length_samples = SEGMENT_LENGTH_SEC * sampling_rate
        #segment_amount = recording_length_samples // segment_length_samples
        segment_amount = utility.ceildiv(recording_length_samples, segment_length_samples)
        # split the session data column-wise (along second axis), NOTE: this returns VIEWs of the original array
        segments = np.split(session_data, range(segment_length_samples, (segment_amount)*segment_length_samples, segment_length_samples), axis=1)

        # omit last segment if shorter than specified segment_length
        if (segments[-1])[0].size < segment_length_samples:
            segments.pop()

        segmented_recording_sessions.append((recording_session, segments))
    
    segmentation_time = time.time() - start_segmentation_time

    """TEST: if all segments have the same length"""

    
    """ TEST: if segmentation was done correctly
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
    ### APPROACH 1: NAIVE IMPLEMENTATION

    start_evaluation_time_naive = time.time()

    # allocate 3D (4D) data matrices to store results of all recording sessions
    # TODO make it numpy arrays
    total_ica_matrix = []
    total_ic_select_matrix = []
    total_peak_detect_matrix = []


    # employ ICA and peak detection on every segment of every recording session
    for (recording_session, segments) in segmented_recording_sessions:

        # setup 2D (3D) matrices to store results of the recording session
        session_ica_matrix = []
        session_ic_select_matrix = []
        session_peak_detect_matrix = []

        # apply ICA and store metrics for each segment of the recording session
        for segment in segments:

            ecg_signal = segment[2]
            eeg_signals = segment[3:]
            ica_dicts, ecg_related_ics, comp_times, pTP_snrs, rssq_snrs = ica.evaluate_all_MNE_ICA_on_segment(ecg_signal, eeg_signals, component_amount=7, max_iterations=50)

            # store results of ICA on the segment in session metric matrix (as a row)
            session_ica_matrix.append(ica_dicts)
            session_ic_select_matrix.append(ecg_related_ics)

        # apply peak detection and store metrics for each segment of the recording session
        for segment in segments:

            ecg_signal = segment[2]

            # set up 2D matrix to store peak detection metrics of the segment 
            segment_peak_detect_matrix = []

            # apply peak detection and store metrics for each ECG-related IC (ie. executed ICA-Variant) of the segment
            for ecg_component in ecg_related_ics:
                
                peak_detect_dicts = peak_detect.evaluate_all_peak_detect_methods(ecg_component, ecg_signal)
                
                # store results of peak detection (as row in segment_peak_detect_matrix)
                segment_peak_detect_matrix.append(peak_detect_dicts)
            
            session_peak_detect_matrix.append(segment_peak_detect_matrix)



    evaluation_time_naive = time.time() - start_evaluation_time_naive

    print("§§§ Evaluation for all recordings took in total: §§§")
    print(segmentation_time, " (segmentation time in sec)")
    print(evaluation_time_naive, " (evaluation time in sec)")
    # preprocess data appropriately for ICA

    # select ECG-related component (resulting from ICA)

    # employ R-peak detection on timeseries of ECG-related IC

    # evaluate feasibility of all combinations of MNE.ICA x Neurokit.Peak-Detection-Methods 

if __name__ == "__main__":
    main()