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
        
        sesh_ica_compTimes_matrix = []
        sesh_ica_actual_iterations_amount_matrix = []
        sesh_ica_pTpSNRs_matrix = []
        sesh_ica_rssqSNRs_matrix = []
        sesh_ica_refcorrs_matrix = []
        sesh_ica_dist_to_restCorrs_matrix = []
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
            """
            ##################
            if seg_idx >= 3: 
                break
            ##################
            
            seg_ref_ecg_signal = segment[2]
            seg_eeg_signals = segment[3:]
            seg_ica_dicts, seg_computation_times_sec, seg_actual_iterations_amount, seg_pTp_snrs_dB, seg_rssq_snrs_dB, seg_corrs_with_refECG, seg_dist_to_restCorrs, seg_ecg_related_ics = ica.evaluate_all_MNE_ICA_on_segment(
                seg_ref_ecg_signal, seg_eeg_signals, component_amount=7, max_iterations="auto")
            
            # store ICA results of the segment in session matrix (row-wise)
            sesh_ica_compTimes_matrix.append(seg_computation_times_sec)
            sesh_ica_actual_iterations_amount_matrix.append(seg_actual_iterations_amount)
            sesh_ica_pTpSNRs_matrix.append(seg_pTp_snrs_dB)
            sesh_ica_rssqSNRs_matrix.append(seg_rssq_snrs_dB)
            sesh_ica_refcorrs_matrix.append(seg_corrs_with_refECG)
            sesh_ica_dist_to_restCorrs_matrix.append(seg_dist_to_restCorrs)
            sesh_ica_ecgICs_matrix.append(seg_ecg_related_ics)

            # store ref_ecg of the segment for easier access
            sesh_ref_ecg.append(seg_ref_ecg_signal)

            # store ica_dicts of the segment for easier access
            sesh_ica_dicts_matrix.append(seg_ica_dicts)

        #######################################################################################################
        #######################################################################################################
        

        #######################################################################################################
        ####################################### PEAK DETECTION ################################################
        #######################################################################################################

        # setup 3D matrices to store Peak Detection results of the recording session
        sesh_pd_compTimes_matrix = []
        sesh_pd_TPs_matrix = []
        sesh_pd_FNs_matrix = []
        sesh_pd_FPs_matrix = []
        sesh_pd_displacements_matrix = []
        sesh_pd_JFScores_matrix = []

        # traverse the matrix with all ECG-related ICs (row-wise, i.e. segment-wise) of the recording session and
        # apply peak detection on each IC
        for row_idx, row in enumerate(sesh_ica_ecgICs_matrix):
    
            # setup 2D matrix to store peak detection results of the segment
            seg_pd_compTimes_matrix = []
            seg_pd_TPs_matrix = []
            seg_pd_FNs_matrix = []
            seg_pd_FPs_matrix = []
            seg_pd_displacements_matrix = []
            seg_pd_JFScores_matrix = []

            # ref ecg of the segment
            seg_ref_ecg_signal = sesh_ref_ecg[row_idx]

            
            # apply peak detection and store metrics for each ECG-related IC (ie. executed ICA-Variant) of the segment
            for comp_idx, ecg_component in enumerate(row):
                "TEST"
                print("TEST EVAL of ", comp_idx, " component! In the ", row_idx, "-th Segment")

                peakDetect_dicts = peak_detect.evaluate_all_peak_detect_methods_on_component(ecg_component, seg_ref_ecg_signal)

                # extract and represent metrics of all PD methods on the component as list/array
                comp_pd_compTimes_list = []
                comp_pd_TPs_list = []
                comp_pd_FNs_list = []
                comp_pd_FPs_list = []
                comp_pd_displacements_list = []
                comp_pd_JFScores_list = []

                for method_dict in peakDetect_dicts:
                    metrics_dict = method_dict["metrics_dict"]
                    comp_pd_compTimes_list.append(metrics_dict["compTime_sec"])
                    comp_pd_TPs_list.append(metrics_dict["true_positives"])
                    comp_pd_FNs_list.append(metrics_dict["false_negatives"])
                    comp_pd_FPs_list.append(metrics_dict["false_positives"])
                    comp_pd_displacements_list.append(metrics_dict["peak_displacements_sample"])
                    comp_pd_JFScores_list.append(metrics_dict["jf_score"])


                # store results of peak detection (as row in segment_peak_detect_matrix)
                seg_pd_compTimes_matrix.append(comp_pd_compTimes_list)
                seg_pd_TPs_matrix.append(comp_pd_TPs_list)
                seg_pd_FNs_matrix.append(comp_pd_FNs_list)
                seg_pd_FPs_matrix.append(comp_pd_FPs_list)
                seg_pd_displacements_matrix.append(comp_pd_displacements_list)
                seg_pd_JFScores_matrix.append(comp_pd_JFScores_list)
            

            # store Peak Detection results of the segment in session matrix (row-wise)
            sesh_pd_compTimes_matrix.append(seg_pd_compTimes_matrix)
            sesh_pd_TPs_matrix.append(seg_pd_TPs_matrix)
            sesh_pd_FNs_matrix.append(seg_pd_FNs_matrix)
            sesh_pd_FPs_matrix.append(seg_pd_FPs_matrix)
            sesh_pd_displacements_matrix.append(seg_pd_displacements_matrix)
            sesh_pd_JFScores_matrix.append(seg_pd_JFScores_matrix)

        #######################################################################################################
        #######################################################################################################

        #######################################################################################################
        ############################### BUILD COMBINED F_EVAL MATRIX ##########################################
        #######################################################################################################
        
        # extract average computation time and SNRs for each ICA algorithm (with standard deviation)
        # (averaged over all segments)

        sesh_avg_compTime_ICAs = []
        sesh_avg_pTpSNR_ICAs = []
        sesh_avg_rssqSNR_ICAs= []
        sesh_avg_refcorr_ICAs = []
        sesh_avg_distance_restcorrs_ICAs = []

        for ica_algo in (np.transpose(sesh_ica_compTimes_matrix)):
            sesh_avg_compTime_ICAs.append((np.average(ica_algo), np.std(ica_algo)))
          
        for ica_algo in (np.transpose(sesh_ica_pTpSNRs_matrix)):
            sesh_avg_pTpSNR_ICAs.append((np.average(ica_algo), np.std(ica_algo)))

        for ica_algo in (np.transpose(sesh_ica_rssqSNRs_matrix)):
            sesh_avg_rssqSNR_ICAs.append((np.average(ica_algo), np.std(ica_algo)))

        # extract average correlation of ecg-related IC with ref ECG
        for ica_algo in (np.abs(np.transpose(sesh_ica_refcorrs_matrix))):
            sesh_avg_refcorr_ICAs.append((np.average(ica_algo), np.std(ica_algo)))
        
        # extract average distance to rest correlation (as secondary metric for correlation strength)
        for ica_algo in (np.abs(np.transpose(sesh_ica_dist_to_restCorrs_matrix))):
            sesh_avg_distance_restcorrs_ICAs.append((np.average(ica_algo), np.std(ica_algo)))

        print("Average ICA comp Times (together with std)")
        print(pd.DataFrame(sesh_avg_compTime_ICAs))
        
        print("Average ICA pTp-SNRs (together with std)")
        print(pd.DataFrame(sesh_avg_pTpSNR_ICAs))

        print("Average ICA rssq-SNRs (together with std)")
        print(pd.DataFrame(sesh_avg_rssqSNR_ICAs))

        print("Average ICA Corrs with Ref ECG (together with std)")
        print(pd.DataFrame(sesh_avg_refcorr_ICAs))

        print("Average distance of ICA Corr with Ref ECG to rest of correlation values (together with std)")
        print(pd.DataFrame(sesh_avg_distance_restcorrs_ICAs))

        
        # build combined evaluation matrices (first axis/rows as peak detection methods and second axis/columns
        # ICA-variants)
        sesh_avg_compTime_combined_PD = []
        sesh_avg_jfScore_combined_PD = []
        sesh_avg_TP_combined_PD = []
        sesh_avg_FN_combined_PD = []
        sesh_avg_FP_combined_PD = []
        sesh_avg_sample_discplacements_combined_PD = []
        sesh_avg_amount_discplacements_combined_PD = []
        #sesh_avg_jitterScore_combined_PD = []
        #sesh_avg_F1score_combined_PD = []


        # extract the average peak detection metrics for each ICA x PD combination

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_compTimes_matrix).T):
            ica_variants = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants.append(np.average(ica_variant))
            sesh_avg_compTime_combined_PD.append(ica_variants)
        print("PD x ICA: Average Computation Times in Seconds PD")
        print(pd.DataFrame(np.array(sesh_avg_compTime_combined_PD)))

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_JFScores_matrix).T):
            ica_variants = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants.append(np.average(ica_variant))
            sesh_avg_jfScore_combined_PD.append(ica_variants)
        print("PD x ICA: Average JF-Scores PD")
        print(pd.DataFrame(np.array(sesh_avg_jfScore_combined_PD)))

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_TPs_matrix).T):
            ica_variants = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants.append(np.average(ica_variant))
            sesh_avg_TP_combined_PD.append(ica_variants)        
        print("PD x ICA: Average True Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_TP_combined_PD)))
        
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FNs_matrix).T):
            ica_variants = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants.append(np.average(ica_variant))
            sesh_avg_FN_combined_PD.append(ica_variants)
        print("PD x ICA: Average False Negatives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FN_combined_PD)))

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FPs_matrix).T):
            ica_variants = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants.append(np.average(ica_variant))
            sesh_avg_FP_combined_PD.append(ica_variants)
        print("PD x ICA: Average False Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FP_combined_PD)))

        print("Shape of Displacement Matrix ", np.array(sesh_pd_displacements_matrix, dtype=list).shape)


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            ica_variants_sample_dis = []
            ica_variants_amount_dis = []
            for ica_idx, ica_variant in enumerate(pd_method):
                displacement_lists_fused = []
                displacement_amounts_fused = []
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    displacement_lists_fused += displacement_list
                    displacement_amounts_fused.append(len(displacement_list))
                
                ica_variants_sample_dis.append(np.average(displacement_lists_fused))
                ica_variants_amount_dis.append(np.average(displacement_amounts_fused))

            sesh_avg_sample_discplacements_combined_PD.append(ica_variants_sample_dis)  
            sesh_avg_amount_discplacements_combined_PD.append(ica_variants_amount_dis)  

        print("PD x ICA: Average Displacements in Samples PD")
        print(pd.DataFrame(np.array(sesh_avg_sample_discplacements_combined_PD)))

        print("PD x ICA: Average Amount Displacements PD")
        print(pd.DataFrame(np.array(sesh_avg_amount_discplacements_combined_PD)))



        evaluation_time_naive = time.time() - start_evaluation_time_naive

        print("§§§TEST§§§ Evaluation for all recordings took in total: §§§")
        print(segmentation_time, " (segmentation time in sec)")
        print(evaluation_time_naive, " (evaluation time in sec)")
        # preprocess data appropriately for ICA

        # select ECG-related component (resulting from ICA)

        # employ R-peak detection on timeseries of ECG-related IC

        # evaluate feasibility of all combinations of MNE.ICA x Neurokit.Peak-Detection-Methods 

        # extract average computation time and peak detection metrics for each peak detection algorithm (with standard deviation)
        # (averaged over all segments) 

        sesh_avg_compTime_single_PDs = []
        sesh_avg_jfScore_single_PDs = []
        sesh_avg_TP_single_PDs = []
        sesh_avg_FN_single_PDs = []
        sesh_avg_FP_single_PDs = []
        sesh_avg_sample_discplacements_single_PDs = []
        sesh_avg_amount_discplacements_single_PDs = []

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_compTimes_matrix).T):
            sesh_avg_compTime_single_PDs.append(np.average(pd_method.flatten()))
           
        print("Single PD: Average Computation Times in Seconds PD")
        print(pd.DataFrame(np.array(sesh_avg_compTime_single_PDs)))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_JFScores_matrix).T):
            sesh_avg_jfScore_single_PDs.append(np.average(pd_method.flatten()))
        
        print("Single PD: Average JF-Scores PD")
        print(pd.DataFrame(np.array(sesh_avg_jfScore_single_PDs)))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_TPs_matrix).T):
            sesh_avg_TP_single_PDs.append(np.average(pd_method.flatten()))

        print("Single PD: Average True Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_TP_single_PDs)))
        

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FNs_matrix).T):
            sesh_avg_FN_single_PDs.append(np.average(pd_method.flatten()))

        print("Single PD: Average False Negatives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FN_single_PDs)))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FPs_matrix).T):
            sesh_avg_FP_single_PDs.append(np.average(pd_method.flatten()))

        print("Single PD: Average False Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FP_single_PDs)))

        # extract average displacement amount and average displacements in samples for each pd method
        # (averaged over all segments and ICA-Results (ECG-related ICs))
        
        # TEST
        #print("Shape of Displacement Matrix ", np.array(sesh_pd_displacements_matrix, dtype=list).shape)

        # extract average displacements in samples (averaged over all segments)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            pd_method_sample_displacements = []

            for ica_idx, ica_variant in enumerate(pd_method):
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    pd_method_sample_displacements += displacement_list

            sesh_avg_sample_discplacements_single_PDs.append(np.average(pd_method_sample_displacements))  

        print("Single PD: Average Displacements in Samples PD")
        print(pd.DataFrame(np.array(sesh_avg_sample_discplacements_single_PDs)))


        # now average amount of displacements 
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            pd_method_displacement_amounts = []
            for ica_idx, ica_variant in enumerate(pd_method):
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    pd_method_displacement_amounts.append(len(displacement_list))

            sesh_avg_amount_discplacements_single_PDs.append(np.average(pd_method_displacement_amounts))       

        print("Single PD: Average Amount Displacements PD")
        print(pd.DataFrame(np.array(sesh_avg_amount_discplacements_single_PDs)))

        
       

if __name__ == "__main__":
    main()