import numpy as np
import pandas as pd

import data.data_offline as data_off

segment_length_sec = 60

def main():

    segmented_recording_sessions = []

    # load offline (brainflow) recording sessions 
    for session in data_off.get_offline_recording_sessions():
        sampling_rate = session.get_description()["sampling_rate"]
        # TODO does it make difference if ECG is also processed or not?
        session_data = session.get_preprocessed_data()
        # session length in sec is amount of samples divided by sampling rate (in Hz)
        recording_length_samples = session_data[0].size

        # segment the session data into equally long segments
        segment_length_samples = segment_length_sec * sampling_rate
        segment_amount = recording_length_samples // segment_length_samples
        # split the session data column-wise (along second axis), NOTE: this returns VIEWs of the original array
        segments = np.array_split(session_data, range(0, segment_amount*segment_length_samples, segment_length_samples), axis=1)

        # omit last segment if shorter than specified segment_length
        if recording_length_samples % segment_length_samples != 0:
            segments.pop()

        segmented_recording_sessions.append((session, segments))

    """ TODO For test purposes only
    for (session, segments) in segmented_recording_sessions:
        for segment in segments:

            data_frame = pd.DataFrame(np.transpose(segment))
            print('Data From the File')
            print(data_frame.head(61))
    """

    # employ ICA on every segment

    
     

    # preprocess data appropriately for ICA

    # select ECG-related component (resulting from ICA)

    # employ R-peak detection on timeseries of ECG-related IC

    # evaluate feasibility of all combinations of MNE.ICA x Neurokit.Peak-Detection-Methods 

if __name__ == "__main__":
    main()