"""
    Module description
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import utility
import data.data_offline as data_off
import ica
import peak_detect

SEGMENT_LENGTH_SEC = 60
SEGMENT_AMOUNT = 13
ICA_VAR_TO_EVAL = 8
PEAK_DETECT_VAR_TO_EVAL = 16

# ica parameters
PC_AMOUNT_TO_CONSIDER = 7
ICA_MAX_ITERATIONS = 2

def main():

    start_segmentation_time = time.time()

    segmented_recording_sessions = []

    brainflow_prepo_times = []

    # load offline (brainflow) recording sessions 
    for recording_session in data_off.get_offline_recording_sessions():

        sampling_rate = recording_session.get_description()["sampling_rate"]
        
        (session_data, prepo_time) = recording_session.get_preprocessed_data()
        # keep track of brainflow prepo times
        brainflow_prepo_times.append(prepo_time)
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

        # ensure all recording_sessions have the same amount of segments
        if len(segments) > SEGMENT_AMOUNT:
            segments.pop()

        assert len(segments) == SEGMENT_AMOUNT

        segmented_recording_sessions.append((recording_session, segments))
    
    segmentation_time = time.time() - start_segmentation_time
    
    """ FOR TEST PURPOSES ONLY 
        TEST: if segmentation was done correctly
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
        sesh_ica_pTpSNRs_reference_matrix = []
        sesh_ica_rssqSNRs_reference_matrix = []
        sesh_ica_refcorrs_matrix = []
        sesh_ica_dist_to_restCorrs_matrix = []
        sesh_ica_explained_variances_matrix = []
        sesh_ica_ecgICs_matrix = []


        # setup list of ref ecg (we have one for each segement)
        sesh_ref_ecg = []


        # apply ICA and store metrics for each segment of the recording session
        for seg_idx, segment in enumerate(segments):

            seg_ref_ecg_signal = segment[2]
            seg_eeg_signals = segment[3:]
            seg_ica_dicts, seg_computation_times_sec, seg_actual_iterations_amount, seg_seconds_per_iteration, seg_pTp_snrs_dB, seg_rssq_snrs_dB, seg_SNRs_dB_reference, seg_corrs_with_refECG, seg_dist_to_restCorrs, seg_variances_explained, seg_ecg_related_ics = ica.evaluate_all_MNE_ICA_on_segment(
                seg_ref_ecg_signal, seg_eeg_signals, component_amount=PC_AMOUNT_TO_CONSIDER, max_iterations=ICA_MAX_ITERATIONS)
            
            # unpack the computation times (for MNE RawObj creation time, MNE filtering time and MNE ICA fitting time)
            seg_mne_rawObjCreate_times_sec = seg_computation_times_sec[0]
            seg_mne_filt_times_sec = seg_computation_times_sec[1]
            seg_ica_fitting_times_sec = seg_computation_times_sec[2]

            # unpack list with segment-wise pTp and RSSQ SNRs of reference ECG
            seg_pTp_SNRs_dB_reference = seg_SNRs_dB_reference[0]
            seg_rssq_SNRs_dB_reference = seg_SNRs_dB_reference[1]
            

            # store ICA results of the segment in session matrix (row-wise)
            sesh_ica_mneRawObjCreate_times_matrix.append(seg_mne_rawObjCreate_times_sec)
            sesh_ica_mneFiltering_times_matrix.append(seg_mne_filt_times_sec)
            sesh_ica_fitting_times_matrix.append(seg_ica_fitting_times_sec)
            sesh_ica_actual_iterations_amount_matrix.append(seg_actual_iterations_amount)
            sesh_ica_seconds_per_iteration_matrix.append(seg_seconds_per_iteration)
            sesh_ica_pTpSNRs_matrix.append(seg_pTp_snrs_dB)
            sesh_ica_rssqSNRs_matrix.append(seg_rssq_snrs_dB)
            sesh_ica_pTpSNRs_reference_matrix.append(seg_pTp_SNRs_dB_reference)
            sesh_ica_rssqSNRs_reference_matrix.append(seg_rssq_SNRs_dB_reference)
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
        ####################################### PEAK DETECTION ################################################
        start_pd_evaluation_time_naive = time.time()
        #######################################################################################################
        
        # setup 3D matrices to store Peak Detection results of the recording session
        sesh_pd_compTimes_matrix = []
        sesh_pd_TPs_matrix = []
        sesh_pd_FNs_matrix = []
        sesh_pd_FPs_matrix = []
        sesh_pd_displacements_matrix = []
        sesh_pd_JFScores_matrix = []

        sesh_pd_whole_TPs_matrix = []
        sesh_pd_SE_matrix = []

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

            seg_pd_whole_TPs_matrix = []
            seg_pd_SE_matrix = []

            # ref ecg of the segment
            seg_ref_ecg_signal = sesh_ref_ecg[row_idx]

            
            # apply peak detection and store metrics for each ECG-related IC (ie. executed ICA-Variant) of the segment
            # (hence, traversing ICA variant/ECG-related IC -wise)
            for comp_idx, ecg_component in enumerate(row):

                """ FOR TEST PURPOSES ONLY """
                print("TEST: EVAL of ", comp_idx, " component! In the ", row_idx, "-th Segment")

                peakDetect_dicts = peak_detect.evaluate_all_peak_detect_methods_on_component(ecg_component, seg_ref_ecg_signal)

                # extract and represent metrics of all PD methods on the component as list/array
                comp_pd_compTimes_list = []
                comp_pd_TPs_list = []
                comp_pd_FNs_list = []
                comp_pd_FPs_list = []
                comp_pd_displacements_list = []
                comp_pd_JFScores_list = []

                comp_pd_whole_TPs_list = []
                comp_pd_SE_list = []

                for method_dict in peakDetect_dicts:
                    metrics_dict = method_dict["metrics_dict"]
                    comp_pd_compTimes_list.append(metrics_dict["compTime_sec"])
                    comp_pd_TPs_list.append(metrics_dict["true_positives"])
                    comp_pd_FNs_list.append(metrics_dict["false_negatives"])
                    comp_pd_FPs_list.append(metrics_dict["false_positives"])
                    comp_pd_displacements_list.append(metrics_dict["peak_displacements_sample"])
                    comp_pd_JFScores_list.append(metrics_dict["jf_score"])

                    comp_pd_whole_TPs_list.append(metrics_dict["all_hits"])
                    comp_pd_SE_list.append(metrics_dict["sensitivity"])


                # store results of peak detection (as row in segment_peak_detect_matrix)
                seg_pd_compTimes_matrix.append(comp_pd_compTimes_list)
                seg_pd_TPs_matrix.append(comp_pd_TPs_list)
                seg_pd_FNs_matrix.append(comp_pd_FNs_list)
                seg_pd_FPs_matrix.append(comp_pd_FPs_list)
                seg_pd_displacements_matrix.append(comp_pd_displacements_list)
                seg_pd_JFScores_matrix.append(comp_pd_JFScores_list)

                seg_pd_whole_TPs_matrix.append(comp_pd_whole_TPs_list)
                seg_pd_SE_matrix.append(comp_pd_SE_list)
            

            # store Peak Detection results of the segment in session matrix (row-wise)
            sesh_pd_compTimes_matrix.append(seg_pd_compTimes_matrix)
            sesh_pd_TPs_matrix.append(seg_pd_TPs_matrix)
            sesh_pd_FNs_matrix.append(seg_pd_FNs_matrix)
            sesh_pd_FPs_matrix.append(seg_pd_FPs_matrix)
            sesh_pd_displacements_matrix.append(seg_pd_displacements_matrix)
            sesh_pd_JFScores_matrix.append(seg_pd_JFScores_matrix)

            sesh_pd_whole_TPs_matrix.append(seg_pd_whole_TPs_matrix)
            sesh_pd_SE_matrix.append(seg_pd_SE_matrix)

        #######################################################################################################
        print("§§§TEST§§§ ICA Computation for all recordings took in total: §§§")
        print(ica_computation_time_naive, " (evaluation time in sec)")
        #######################################################################################################
        pd_computation_time_naive = time.time() - start_pd_evaluation_time_naive
        print("§§§TEST§§§ PEAK DETECT Computation for all recordings took in total: §§§")
        print(pd_computation_time_naive, " (evaluation time in sec)")
        #######################################################################################################
        ############################### BUILD COMBINED F_EVAL MATRIX ##########################################
        #######################################################################################################

        # names for pandas dataframes output
        col_labels = ["M", "SD"]
        ica_names = ["picard", "picard_o", "ext_picard", "ext_picard_o", "infomax", "ext_infomax", "fastica_par", "fastica_defl"]
        peak_detect_names = [   "dirty_neurokit",
                                "clean_neurokit",
                                "dirty_pantompkins1985",
                                "clean_pantompkins1985",
                                "dirty_hamilton2002",
                                "clean_hamilton2002",
                                "dirty_elgendi2010",
                                "clean_elgendi2010",
                                "dirty_engzeemod2012",
                                "clean_engzeemod2012",
                                "kalidas2017",
                                "zong2003",
                                "christov2004",
                                "nabian2018",
                                "rodrigues2021",
                                "promac"
                            ]
        
        
        # extract average computation time and SNRs for each ICA algorithm (with standard deviation)
        # (averaged over all segments, rounded to one or two decimals, time converted in millis)

        sesh_avg_mneRawObjCreate_times_millis_ICAs = []
        sesh_avg_mneFiltering_times_millis_ICAs = []
        sesh_avg_fitting_times_millis_ICAs = []
        sesh_avg_amount_actualIter_ICAs = []
        sesh_avg_millis_per_iter_ICAs = []
        sesh_avg_pTpSNR_ICAs = []
        sesh_avg_rssqSNR_ICAs= []
        sesh_avg_pTpSNR_ICAs_reference = []
        sesh_avg_rssqSNR_ICAs_reference = []
        sesh_avg_refcorr_ICAs = []
        sesh_avg_distance_restcorrs_ICAs = []
        sesh_avg_variance_prop_explained_ICAs = []

        for ica_algo in (np.transpose(sesh_ica_mneRawObjCreate_times_matrix)):
            sesh_avg_mneRawObjCreate_times_millis_ICAs.append((np.average(ica_algo * 1000).round(1), np.std(ica_algo * 1000).round(1)))
        
        for ica_algo in (np.transpose(sesh_ica_mneFiltering_times_matrix)):
            sesh_avg_mneFiltering_times_millis_ICAs.append((np.average(ica_algo * 1000).round(1), np.std(ica_algo * 1000).round(1)))
        
        for ica_algo in (np.transpose(sesh_ica_fitting_times_matrix)):
            sesh_avg_fitting_times_millis_ICAs.append((np.average(ica_algo * 1000).round(1), np.std(ica_algo * 1000).round(1)))


        for ica_algo in (np.transpose(sesh_ica_actual_iterations_amount_matrix)):
            sesh_avg_amount_actualIter_ICAs.append((np.average(ica_algo).round(1), np.std(ica_algo).round(1)))


        for ica_algo in (np.transpose(sesh_ica_seconds_per_iteration_matrix)):
            sesh_avg_millis_per_iter_ICAs.append((np.average(ica_algo * 1000).round(1), np.std(ica_algo * 1000).round(1)))

          
        for ica_algo in (np.transpose(sesh_ica_pTpSNRs_matrix)):
            sesh_avg_pTpSNR_ICAs.append((np.average(ica_algo).round(1), np.std(ica_algo).round(1)))


        for ica_algo in (np.transpose(sesh_ica_rssqSNRs_matrix)):
            sesh_avg_rssqSNR_ICAs.append((np.average(ica_algo).round(1), np.std(ica_algo).round(1)))

        for ica_algo in (np.transpose(sesh_ica_pTpSNRs_reference_matrix)):
            sesh_avg_pTpSNR_ICAs_reference.append((np.average(ica_algo).round(1), np.std(ica_algo).round(1)))

        for ica_algo in (np.transpose(sesh_ica_rssqSNRs_reference_matrix)):
            sesh_avg_rssqSNR_ICAs_reference.append((np.average(ica_algo).round(1), np.std(ica_algo).round(1)))



        # extract average correlation of ecg-related IC with ref ECG (only absolute values to measure general corr strength)
        for ica_algo in (np.abs(np.transpose(sesh_ica_refcorrs_matrix))):
            sesh_avg_refcorr_ICAs.append((np.average(ica_algo).round(2), np.std(ica_algo).round(2)))

        
        # extract average distance to rest correlation (as secondary metric for correlation strength)
        for ica_algo in (np.abs(np.transpose(sesh_ica_dist_to_restCorrs_matrix))):
            sesh_avg_distance_restcorrs_ICAs.append((np.average(ica_algo).round(2), np.std(ica_algo).round(2)))


        # extract average proportion of data variance (in ICA input data) in %
        # that is explained by the ECG-related IC obtained by ICA 
        for ica_algo in (np.abs(np.transpose(sesh_ica_explained_variances_matrix))):
            sesh_avg_variance_prop_explained_ICAs.append((np.average(ica_algo * 100).round(1), np.std(ica_algo * 100).round(1)))


        print("Average MNE Raw Object creation times in millis (together with SD)")
        print(pd.DataFrame(sesh_avg_mneRawObjCreate_times_millis_ICAs, index=ica_names, columns=col_labels))

        print("Average MNE filtering times in millis (together with SD)")
        print(pd.DataFrame(sesh_avg_mneFiltering_times_millis_ICAs, index=ica_names, columns=col_labels))

        print("Average ICA fitting times in millis (together with SD)")
        print((pd.DataFrame(sesh_avg_fitting_times_millis_ICAs, index=ica_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))


        print("Average amount of actual ICA iterations performed (together with SD)")
        print((pd.DataFrame(sesh_avg_amount_actualIter_ICAs, index=ica_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))

        print("Average amount of millis per iteration needed (together with SD)")
        print(pd.DataFrame(sesh_avg_millis_per_iter_ICAs, index=ica_names, columns=col_labels))
        
        print("Average ICA pTp-SNRs (together with SD)")
        print((pd.DataFrame(sesh_avg_pTpSNR_ICAs, index=ica_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))

        print("Average ICA rssq-SNRs (together with SD)")
        print(pd.DataFrame(sesh_avg_rssqSNR_ICAs, index=ica_names, columns=col_labels))

        print("Average pTp-SNRs of reference chest-ECG(together with SD)")
        print(pd.DataFrame(sesh_avg_pTpSNR_ICAs_reference, index=ica_names, columns=col_labels))

        print("Average rssq-SNRs of reference chest-ECG (together with SD)")
        print(pd.DataFrame(sesh_avg_rssqSNR_ICAs_reference, index=ica_names, columns=col_labels))

        print("Average ICA Corrs with Ref ECG (together with SD)")
        print((pd.DataFrame(sesh_avg_refcorr_ICAs, index=ica_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))

        print("Average distance of ICA Corr with Ref ECG to rest of correlation values (together with SD)")
        print((pd.DataFrame(sesh_avg_distance_restcorrs_ICAs, index=ica_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))

        print("Average proportion of variance (in the original input data) explained by the ECG-related IC (in % together with SD)")
        print(pd.DataFrame(sesh_avg_variance_prop_explained_ICAs, index=ica_names, columns=col_labels))


        # build combined evaluation matrices (first axis/rows as peak detection methods and second axis/columns
        # ICA-variants)
        sesh_avg_compTime_combined_PD = []
        sesh_avg_compTime_COMBINED = []
        sesh_avg_jfScore_combined_PD = []
        sesh_avg_TP_combined_PD = []
        sesh_avg_FN_combined_PD = []
        sesh_avg_FP_combined_PD = []
        sesh_avg_sample_discplacements_combined_PD = []
        sesh_avg_amount_discplacements_combined_PD = []

        sesh_avg_whole_TPs_combined_PD = []
        sesh_avg_SE_combined_PD = []

        sesh_SD_compTime_combined_PD = []
        sesh_SD_compTime_COMBINED = []
        sesh_SD_jfScore_combined_PD = []
        sesh_SD_TP_combined_PD = []
        sesh_SD_FN_combined_PD = []
        sesh_SD_FP_combined_PD = []
        sesh_SD_sample_discplacements_combined_PD = []
        sesh_SD_amount_discplacements_combined_PD = []

        sesh_SD_whole_TPs_combined_PD = []
        sesh_SD_SE_combined_PD = []


        # extract the average peak detection metrics for each ICA x PD combination
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_compTimes_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant * 1000).round(1))
                ica_variants_SD.append(np.std(ica_variant * 1000).round(1))
            sesh_avg_compTime_combined_PD.append(ica_variants_M)
            sesh_SD_compTime_combined_PD.append(ica_variants_SD)
        print("PD x ICA: MEAN (M) of PD Computation Times in Millis")
        print(pd.DataFrame(np.array(sesh_avg_compTime_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of PD Computation Times in Millis")
        print(pd.DataFrame(np.array(sesh_SD_compTime_combined_PD), index=peak_detect_names, columns=ica_names))

        # add the avg computation times of ICA (fitting) and Peak Detection (employment) together for combined computation time (avg)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_compTimes_matrix).T):
            # add computation time of respective ICA variant fitting times (segment-wise)
            pd_method = np.add(pd_method, np.array(sesh_ica_fitting_times_matrix).T)
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant * 1000).round(1))
                ica_variants_SD.append(np.std(ica_variant * 1000).round(1))
            sesh_avg_compTime_COMBINED.append(ica_variants_M)
            sesh_SD_compTime_COMBINED.append(ica_variants_SD)
        print("PD x ICA: Average (M) of COMBINED Computation Times in Millis")
        print(pd.DataFrame(np.array(sesh_avg_compTime_COMBINED), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of COMBINED Computation Times in Millis")
        print(pd.DataFrame(np.array(sesh_SD_compTime_COMBINED), index=peak_detect_names, columns=ica_names))
        

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_JFScores_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append((np.average(ica_variant).round(1)))
                ica_variants_SD.append(np.std(ica_variant).round(1))
            sesh_avg_jfScore_combined_PD.append(ica_variants_M)
            sesh_SD_jfScore_combined_PD.append(ica_variants_SD)
        print("PD x ICA: Average (M) of JF-Scores PD in %")
        print(pd.DataFrame(np.array(sesh_avg_jfScore_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of JF-Scores PD in %")
        print(pd.DataFrame(np.array(sesh_SD_jfScore_combined_PD), index=peak_detect_names, columns=ica_names))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_TPs_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant).round(1))
                ica_variants_SD.append(np.std(ica_variant).round(1))
            sesh_avg_TP_combined_PD.append(ica_variants_M)       
            sesh_SD_TP_combined_PD.append(ica_variants_SD)       
        print("PD x ICA: Average (M) of True Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_TP_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of True Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_SD_TP_combined_PD), index=peak_detect_names, columns=ica_names))
        

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FNs_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant).round(1))
                ica_variants_SD.append(np.std(ica_variant).round(1))
            sesh_avg_FN_combined_PD.append(ica_variants_M)
            sesh_SD_FN_combined_PD.append(ica_variants_SD)
        print("PD x ICA: Average (M) of False Negatives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FN_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of False Negatives Amount PD")
        print(pd.DataFrame(np.array(sesh_SD_FN_combined_PD), index=peak_detect_names, columns=ica_names))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FPs_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant).round(1))
                ica_variants_SD.append(np.std(ica_variant).round(1))
            sesh_avg_FP_combined_PD.append(ica_variants_M)
            sesh_SD_FP_combined_PD.append(ica_variants_SD)
        print("PD x ICA: Average (M) of False Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_avg_FP_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of False Positives Amount PD")
        print(pd.DataFrame(np.array(sesh_SD_FP_combined_PD), index=peak_detect_names, columns=ica_names))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            ica_variants_sample_dis_M = []
            ica_variants_amount_dis_M = []
            ica_variants_sample_dis_SD = []
            ica_variants_amount_dis_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                displacement_lists_fused = []
                displacement_amounts_fused = []
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    displacement_lists_fused += displacement_list
                    displacement_amounts_fused.append(len(displacement_list))
                
                ica_variants_sample_dis_M.append(np.average(displacement_lists_fused).round(1))
                ica_variants_amount_dis_M.append(np.average(displacement_amounts_fused).round(1))

                ica_variants_sample_dis_SD.append(np.std(displacement_lists_fused).round(1))
                ica_variants_amount_dis_SD.append(np.std(displacement_amounts_fused).round(1))

            sesh_avg_sample_discplacements_combined_PD.append(ica_variants_sample_dis_M)  
            sesh_avg_amount_discplacements_combined_PD.append(ica_variants_amount_dis_M)  
            sesh_SD_sample_discplacements_combined_PD.append(ica_variants_sample_dis_SD)  
            sesh_SD_amount_discplacements_combined_PD.append(ica_variants_amount_dis_SD)  

        print("PD x ICA: Average (M) of Displacements in Samples PD")
        print(pd.DataFrame(np.array(sesh_avg_sample_discplacements_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of Displacements in Samples PD")
        print(pd.DataFrame(np.array(sesh_SD_sample_discplacements_combined_PD), index=peak_detect_names, columns=ica_names))

        print("PD x ICA: Average (M) of Amount Displacements PD")
        print(pd.DataFrame(np.array(sesh_avg_amount_discplacements_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of Amount Displacements PD")
        print(pd.DataFrame(np.array(sesh_SD_amount_discplacements_combined_PD), index=peak_detect_names, columns=ica_names))

        # now extract the amount of all hits (regardless of displacement)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_whole_TPs_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant).round(1))
                ica_variants_SD.append(np.std(ica_variant).round(1))
            sesh_avg_whole_TPs_combined_PD.append(ica_variants_M)       
            sesh_SD_whole_TPs_combined_PD.append(ica_variants_SD)       
        print("PD x ICA: Average (M) of ALL True Positives Amount PD (meaning TP + DP)")
        print(pd.DataFrame(np.array(sesh_avg_whole_TPs_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of ALL True Positives Amount PD (meaning TP + DP)")
        print(pd.DataFrame(np.array(sesh_SD_whole_TPs_combined_PD), index=peak_detect_names, columns=ica_names))

        # now extract the sensitivity 
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_SE_matrix).T):
            ica_variants_M = []
            ica_variants_SD = []
            for ica_idx, ica_variant in enumerate(pd_method):
                ica_variants_M.append(np.average(ica_variant).round(2))
                ica_variants_SD.append(np.std(ica_variant).round(2))
            sesh_avg_SE_combined_PD.append(ica_variants_M)       
            sesh_SD_SE_combined_PD.append(ica_variants_SD)       
        print("PD x ICA: Average (M) of Sensitivity (in %) PD ")
        print(pd.DataFrame(np.array(sesh_avg_SE_combined_PD), index=peak_detect_names, columns=ica_names))
        print("PD x ICA: Standard Deviation (SD) of of Sensitivity (in %) PD ")
        print(pd.DataFrame(np.array(sesh_SD_SE_combined_PD), index=peak_detect_names, columns=ica_names))

        

        evaluation_time_naive = time.time() - start_evaluation_time_naive

        print("§§§TEST§§§ Evaluation for all recordings took in total: §§§")
        print(segmentation_time, " (segmentation time in sec)")
        print(evaluation_time_naive, " (evaluation time in sec)")
        


        # extract average computation time and peak detection metrics for each peak detection algorithm (with standard deviation)
        # (averaged over all segments) 

        sesh_avg_compTime_single_PDs = []
        sesh_avg_jfScore_single_PDs = []
        sesh_avg_TP_single_PDs = []
        sesh_avg_FN_single_PDs = []
        sesh_avg_FP_single_PDs = []
        sesh_avg_sample_discplacements_single_PDs = []
        sesh_avg_amount_discplacements_single_PDs = []

        sesh_avg_whole_TPs__single_PDs = []
        sesh_avg_SE__single_PDs = []

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_compTimes_matrix).T):
            sesh_avg_compTime_single_PDs.append((np.average(pd_method.flatten() * 1000).round(1), np.std(pd_method.flatten() * 1000).round(1)))
           
        print("Single PD: Average Computation Times in Millis PD (with SD)")
        print(pd.DataFrame(np.array(sesh_avg_compTime_single_PDs), index=peak_detect_names, columns=col_labels))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_JFScores_matrix).T):
            sesh_avg_jfScore_single_PDs.append((np.average(pd_method.flatten()).round(1), np.std(pd_method.flatten()).round(1)))
        
        print("Single PD: Average JF-Scores PD in % (with SD)")
        print((pd.DataFrame(np.array(sesh_avg_jfScore_single_PDs), index=peak_detect_names, columns=col_labels)).sort_values(by=col_labels[0], ascending=False))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_TPs_matrix).T):
            sesh_avg_TP_single_PDs.append((np.average(pd_method.flatten()).round(1), np.std(pd_method.flatten()).round(1)))

        print("Single PD: Average True Positives Amount PD (with std)")
        print(pd.DataFrame(np.array(sesh_avg_TP_single_PDs), index=peak_detect_names, columns=col_labels))
        

        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FNs_matrix).T):
            sesh_avg_FN_single_PDs.append((np.average(pd_method.flatten()).round(1), np.std(pd_method.flatten()).round(1)))

        print("Single PD: Average False Negatives Amount PD (with std)")
        print(pd.DataFrame(np.array(sesh_avg_FN_single_PDs), index=peak_detect_names, columns=col_labels))


        for pd_idx, pd_method in enumerate(np.array(sesh_pd_FPs_matrix).T):
            sesh_avg_FP_single_PDs.append((np.average(pd_method.flatten()).round(1), np.std(pd_method.flatten()).round(1)))

        print("Single PD: Average False Positives Amount PD (with std)")
        print(pd.DataFrame(np.array(sesh_avg_FP_single_PDs), index=peak_detect_names, columns=col_labels))

        # extract average displacement amount and average displacements in samples for each pd method
        # (averaged over all segments and ICA-Results (ECG-related ICs))
        

        # extract average displacements in samples (averaged over all segments)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            pd_method_sample_displacements = []

            for ica_idx, ica_variant in enumerate(pd_method):
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    pd_method_sample_displacements += displacement_list

            sesh_avg_sample_discplacements_single_PDs.append((np.average(pd_method_sample_displacements).round(1), np.std(pd_method_sample_displacements).round(1)))

        print("Single PD: Average Displacements in Samples PD (with std)")
        print(pd.DataFrame(np.array(sesh_avg_sample_discplacements_single_PDs), index=peak_detect_names, columns=col_labels))


        # now average amount of displacements 
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_displacements_matrix, dtype=list).T):
            pd_method_displacement_amounts = []
            for ica_idx, ica_variant in enumerate(pd_method):
                for disp_list_idx, displacement_list in enumerate(ica_variant):
                    pd_method_displacement_amounts.append(len(displacement_list))

            sesh_avg_amount_discplacements_single_PDs.append((np.average(pd_method_displacement_amounts).round(1), np.std(pd_method_displacement_amounts).round(1)))         

        print("Single PD: Average Amount Displacements PD (with std)")
        print(pd.DataFrame(np.array(sesh_avg_amount_discplacements_single_PDs), index=peak_detect_names, columns=col_labels))

        # extract amount of all true positives detected (regardless if displaced)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_whole_TPs_matrix).T):
            sesh_avg_whole_TPs__single_PDs.append((np.average(pd_method.flatten()).round(1), np.std(pd_method.flatten()).round(1)))

        print("Single PD: Average of ALL True Positives Amount (meaning TP + DP) (with SD)")
        print(pd.DataFrame(np.array(sesh_avg_whole_TPs__single_PDs), index=peak_detect_names, columns=col_labels))

        # now extract the average sensitivity (in percentage)
        for pd_idx, pd_method in enumerate(np.array(sesh_pd_SE_matrix).T):
            sesh_avg_SE__single_PDs.append((np.average(pd_method.flatten()).round(2), np.std(pd_method.flatten()).round(2)))
        
        print("Single PD: Average Sensitivity PD in % (with SD)")
        print(pd.DataFrame(np.array(sesh_avg_SE__single_PDs), index=peak_detect_names, columns=col_labels))


    # information for data analysis report
    print("RESULTS REPORT FOR")
    print("Proband_03")
    print("ICA_iterations: ", ICA_MAX_ITERATIONS)
    print("PC_considered: ", PC_AMOUNT_TO_CONSIDER)
    print("average brainflow prepo time in sec (per recording): ", np.average(brainflow_prepo_times), np.std(brainflow_prepo_times))
    print("Time_started: " + "16:21")
    print("Date: " + "10.07.2024")  
    print("NOTES:")
    print("Naive Peak Correction set to True")
    print("F1 score was adapted")
       

if __name__ == "__main__":
    main()