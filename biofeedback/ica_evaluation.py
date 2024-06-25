import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import utility
import data.data_offline as data_off
import ica
import peak_detect

SEGMENT_LENGTH_SEC = 60
ICA_VAR_TO_EVAL = 8
PEAK_DETECT_VAR_TO_EVAL = 16

# ica parameters
PC_AMOUNT_TO_CONSIDER = 7
ICA_MAX_ITERATIONS = "auto"

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
    # combined performance evaluation
    total_ica_matrix = []
    total_ic_select_matrix = []
    total_peak_detect_matrix = []


    # employ ICA and peak detection on every segment of every recording session
    for (recording_session, segments) in segmented_recording_sessions:

        #######################################################################################################
        ####################################### ICA ###########################################################
        #######################################################################################################


        # setup 2D matrices to store ICA results of the recording session
        sesh_ica_dicts_matrix = []
        
        sesh_ica_mneRawObjCreate_times_matrix = []
        sesh_ica_mneFiltering_times_matrix = []
        sesh_ica_fitting_times_matrix = []
        sesh_ica_actual_iterations_amount_matrix = []
        sesh_ica_seconds_per_iteration_matrix = []
        sesh_ica_pTpSNRs_matrix = []
        sesh_ica_rssqSNRs_matrix = []
        sesh_ica_refcorrs_matrix = []
        sesh_ica_dist_to_restCorrs_matrix = []
        sesh_ica_explained_variances_matrix = []
        sesh_ica_ecgICs_matrix = []

        sesh_ica_dicts_matrix = []

        # setup list of ref ecg (we have one for each segement)
        sesh_ref_ecg = []


        # apply ICA and store metrics for each segment of the recording session
        for seg_idx, segment in enumerate(segments):
            """ TEST PURPOSE for shorter ICA eval um peak_detect zu debuggen
            if seg_idx < 4:
                continue
            #################
                
            ##################
            if seg_idx >= 2: 
                break
            ##################
            """

            seg_ref_ecg_signal = segment[2]
            seg_eeg_signals = segment[3:]
            seg_ica_dicts, seg_computation_times_sec, seg_actual_iterations_amount, seg_seconds_per_iteration, seg_pTp_snrs_dB, seg_rssq_snrs_dB, seg_corrs_with_refECG, seg_dist_to_restCorrs, seg_variances_explained, seg_ecg_related_ics = ica.evaluate_all_MNE_ICA_on_segment(
                seg_ref_ecg_signal, seg_eeg_signals, component_amount=PC_AMOUNT_TO_CONSIDER, max_iterations=ICA_MAX_ITERATIONS)
            
            # unpack the computation times (for MNE RawObj creation time, MNE filtering time and MNE ICA fitting time)
            seg_mne_rawObjCreate_times_sec = seg_computation_times_sec[0]
            seg_mne_filt_times_sec = seg_computation_times_sec[1]
            seg_ica_fitting_times_sec = seg_computation_times_sec[2]


            # store ICA results of the segment in session matrix (row-wise)
            sesh_ica_mneRawObjCreate_times_matrix.append(seg_mne_rawObjCreate_times_sec)
            sesh_ica_mneFiltering_times_matrix.append(seg_mne_filt_times_sec)
            sesh_ica_fitting_times_matrix.append(seg_ica_fitting_times_sec)
            sesh_ica_actual_iterations_amount_matrix.append(seg_actual_iterations_amount)
            sesh_ica_seconds_per_iteration_matrix.append(seg_seconds_per_iteration)
            sesh_ica_pTpSNRs_matrix.append(seg_pTp_snrs_dB)
            sesh_ica_rssqSNRs_matrix.append(seg_rssq_snrs_dB)
            sesh_ica_refcorrs_matrix.append(seg_corrs_with_refECG)
            sesh_ica_dist_to_restCorrs_matrix.append(seg_dist_to_restCorrs)
            sesh_ica_explained_variances_matrix.append(seg_variances_explained)
            sesh_ica_ecgICs_matrix.append(seg_ecg_related_ics)

            # store ref_ecg of the segment for easier access
            sesh_ref_ecg.append(seg_ref_ecg_signal)

            # store ica_dicts of the segment for easier access
            sesh_ica_dicts_matrix.append(seg_ica_dicts)


        #######################################################################################################
        #######################################################################################################
        ica_computation_time_naive = time.time() - start_evaluation_time_naive
    

        #######################################################################################################
        print("§§§TEST§§§ ICA Computation for all recordings took in total: §§§")
        print(ica_computation_time_naive, " (evaluation time in sec)")
        #######################################################################################################
    
        #######################################################################################################
        ############################### BUILD COMBINED F_EVAL MATRIX ##########################################
        #######################################################################################################
        
        # extract average computation time and SNRs for each ICA algorithm (with standard deviation)
        # (averaged over all segments)

        sesh_avg_mneRawObjCreate_times_ICAs = []
        sesh_avg_mneFiltering_times_ICAs = []
        sesh_avg_fitting_times_ICAs = []
        sesh_avg_amount_actualIter_ICAs = []
        sesh_avg_seconds_per_iter_ICAs = []
        sesh_avg_pTpSNR_ICAs = []
        sesh_avg_rssqSNR_ICAs= []
        sesh_avg_refcorr_ICAs = []
        sesh_avg_distance_restcorrs_ICAs = []
        sesh_avg_variance_prop_explained_ICAs = []

        for ica_algo in (np.transpose(sesh_ica_mneRawObjCreate_times_matrix)):
            sesh_avg_mneRawObjCreate_times_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))
        
        for ica_algo in (np.transpose(sesh_ica_mneFiltering_times_matrix)):
            sesh_avg_mneFiltering_times_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))
        
        for ica_algo in (np.transpose(sesh_ica_fitting_times_matrix)):
            sesh_avg_fitting_times_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        for ica_algo in (np.transpose(sesh_ica_actual_iterations_amount_matrix)):
            sesh_avg_amount_actualIter_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        for ica_algo in (np.transpose(sesh_ica_seconds_per_iteration_matrix)):
            sesh_avg_seconds_per_iter_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))

          
        for ica_algo in (np.transpose(sesh_ica_pTpSNRs_matrix)):
            sesh_avg_pTpSNR_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        for ica_algo in (np.transpose(sesh_ica_rssqSNRs_matrix)):
            sesh_avg_rssqSNR_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        # extract average correlation of ecg-related IC with ref ECG
        for ica_algo in (np.abs(np.transpose(sesh_ica_refcorrs_matrix))):
            sesh_avg_refcorr_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))

        
        # extract average distance to rest correlation (as secondary metric for correlation strength)
        for ica_algo in (np.abs(np.transpose(sesh_ica_dist_to_restCorrs_matrix))):
            sesh_avg_distance_restcorrs_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        # extract average proportion of data variance (in ICA input data) 
        # explained by ECG-related IC obtained by ICA 
        for ica_algo in (np.abs(np.transpose(sesh_ica_explained_variances_matrix))):
            sesh_avg_variance_prop_explained_ICAs.append((np.average(ica_algo).round(3), np.std(ica_algo).round(3)))


        print("#############################################")
        print("Average MNE Raw Object creation times in sec (together with std)")
        print(pd.DataFrame(sesh_avg_mneRawObjCreate_times_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_mneRawObjCreate_times_ICAs).to_latex())

        print("#############################################")
        print("Average MNE filtering times in sec (together with std)")
        print(pd.DataFrame(sesh_avg_mneFiltering_times_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_mneFiltering_times_ICAs).to_latex())

        print("#############################################")
        print("Average ICA fitting times in sec (together with std)")
        print(pd.DataFrame(sesh_avg_fitting_times_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_fitting_times_ICAs).to_latex())

        print("#############################################")
        print("Average amount of actual ICA iterations performed (together with std)")
        print(pd.DataFrame(sesh_avg_amount_actualIter_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_amount_actualIter_ICAs).to_latex())

        print("#############################################")
        print("Average amount of seconds per iteration needed (together with std)")
        print(pd.DataFrame(sesh_avg_seconds_per_iter_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_seconds_per_iter_ICAs).to_latex())
        
        print("#############################################")
        print("Average ICA pTp-SNRs (together with std)")
        print(pd.DataFrame(sesh_avg_pTpSNR_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_pTpSNR_ICAs).to_latex())

        print("#############################################")
        print("Average ICA rssq-SNRs (together with std)")
        print(pd.DataFrame(sesh_avg_rssqSNR_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_rssqSNR_ICAs).to_latex())

        print("#############################################")
        print("Average ICA Corrs with Ref ECG (together with std)")
        print(pd.DataFrame(sesh_avg_refcorr_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_refcorr_ICAs).to_latex())

        print("#############################################")
        print("Average distance of ICA Corr with Ref ECG to rest of correlation values (together with std)")
        print(pd.DataFrame(sesh_avg_distance_restcorrs_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_distance_restcorrs_ICAs).to_latex())

        print("#############################################")
        print("Average proportion of variance (in the original input data) explained by the ECG-related IC (together with std)")
        print(pd.DataFrame(sesh_avg_variance_prop_explained_ICAs))
        print("-----------------------------------------------")
        print(pd.DataFrame(sesh_avg_variance_prop_explained_ICAs).to_latex())

        
       

if __name__ == "__main__":
    main()