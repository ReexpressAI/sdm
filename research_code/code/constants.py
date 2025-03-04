# Copyright Reexpress AI, Inc. All rights reserved.

FORWARD_TYPE_GENAI_WITH_ROUTER_TOKEN_LEVEL_PREDICTION = "genai_with_router_token_level_prediction"
FORWARD_TYPE_FEATURE_EXTRACTION = "feature_extraction"
FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS = "generate_exemplar_vectors"
FORWARD_TYPE_SEQUENCE_LABELING_AND_SENTENCE_LEVEL_PREDICTION = "sequence_labeling_and_sentence_level_prediction"
FORWARD_TYPE_SEQUENCE_LABELING = "sequence_labeling"
FORWARD_TYPE_SENTENCE_LEVEL_PREDICTION = "sentence_level_prediction"
FORWARD_TYPE_TRAIN_RESCALER = "train_rescaler"
FORWARD_TYPE_RESCALE_CACHED_CALIBRATION = "rescale_cached_calibration"
FORWARD_TYPE_SINGLE_PASS_TEST = "single_pass_test"
FORWARD_TYPE_SINGLE_PASS_TEST_ANALYSIS = "single_pass_test_analysis"
# Return the positive and negative contributions separately. This is primarily to use the existing multi-class code
# base for training and evaluating the bounds.
FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS_WITH_SEPARATE_POS_NEG_LOGITS = \
    "generate_exemplar_vectors_with_separate_pos_neg_logits"

# Return the document-level max-pooled vector as the exemplar for the document.
FORWARD_TYPE_GENERATE_EXEMPLAR_VECTORS_DOCUMENT_LEVEL = \
    "generate_exemplar_vectors_document_level"

# This is analogous to the multi-label case, but here, only for binary classification. This allows for a neutral
# class.
FORWARD_TYPE_BINARY_TOKEN_DECOMPOSITION_WITH_NEUTRAL_CLASS = \
    "binary_decomposition_with_neutral_class"

FORWARD_TYPE_GENERATE_EXEMPLAR_GLOBAL_AND_LOCAL_VECTORS_BINARY_TOKEN_DECOMPOSITION_WITH_NEUTRAL_CLASS = \
    "generate_exemplar_vectors_with_token_vectors_from_binary_decomposition_with_neutral_class_combined_with_the_document_vector"


##### Error Messages
ERROR_MESSAGES_NO_THRESHOLD_FOUND = \
    "WARNING: Unable to find a suitable bin threshold to achieve the target class-conditional accuracy."
ERROR_MESSAGES_UNCERTAINTY_STATS_JSON_MALFORMED = \
    "WARNING: The archive for the UncertaintyStatistics class appears to be malformed."

##### SDM constants
q_rescale_offset: int = 2  # This typically should not change.
ood_limit: int = 0  # This typically should not change.
#min_valid_qbin_for_class_conditional_accuracy: int = 1  # typically only this value or higher should be considered
# minReliableCumulativePartitionSize: int = 1000
maxQAvailableFromIndexer: int = 1000 # 150  # This is the max k indexed. Note that this corresponds to the raw q value. Ignored if --use_training_set_max_label_size_as_max_q is used
default_max_hard_bin = 20  # arbitrarily large to handle q up to np.exp(20) = 485165195; i.e., max hard bin is int(np.log(int(np.exp(20))))
##### SDM generation model constants
top_logits_k: int = 3
#####
minProbabilityPrecisionForDisplay: float = 0.01
maxProbabilityPrecisionForDisplay: float = 0.99
probabilityPrecisionStride: float = 0.01

# when adding probability retrieval (see getCDFCategoriesByProbabilityRestrictions() in Swift v1); not currently used
retrieval_probability_tolerance: float = 0.004

balancedAccuracyDescription = \
    "Balanced Accuracy is the average of the Accuracy for each class. It is generally more informative as a single composite metric than overall Accuracy when there is class imbalance."


defaultCdfAlpha: float = 0.95
defaultCdfThresholdTolerance: float = 0.001
defaultQMax: int = 25
minReliablePartitionSize: int = 100  # When the partition size is less than this value, we treat the calibration reliability as the lowest possible. Additional, some additional visual queues can be provided to the user (such as highlighting the size) to draw attention to the user.

defaultDistanceQuantile: float = 0.05

# ModelControl
keyModelDimension = 1000

def floatProbToDisplaySignificantDigits(floatProb: float) -> str:
    intProb = int(floatProb*100.0)
    floored = max(minProbabilityPrecisionForDisplay, min(maxProbabilityPrecisionForDisplay, float(intProb)/100.0))
    return f"{floored:.2f}"  # String(format: "%.2f", floored)


##### ProgramIdentifiers
ProgramIdentifiers_mainProgramName = "Reexpress connect"
ProgramIdentifiers_mainProgramNameShort = "connect"
ProgramIdentifiers_version = "25a_sdm_streamlined"


##### Storage keys
STORAGE_KEY_version = "version"
STORAGE_KEY_uncertaintyModelUUID = "uncertaintyModelUUID"
STORAGE_KEY_alpha = "alpha"
STORAGE_KEY_non_odd_class_conditional_accuracy = "non_odd_class_conditional_accuracy"
STORAGE_KEY_cdfThresholdTolerance = "cdfThresholdTolerance"
STORAGE_KEY_maxQAvailableFromIndexer = "maxQAvailableFromIndexer"
STORAGE_KEY_minReliableCumulativePartitionSize = "minReliableCumulativePartitionSize"
STORAGE_KEY_numberOfClasses = "numberOfClasses"

STORAGE_KEY_q_rescale_offset = "q_rescale_offset"
STORAGE_KEY_ood_limit = "ood_limit"
STORAGE_KEY_exemplar_vector_dimension = "exemplar_vector_dimension"
STORAGE_KEY_embedding_size = "embedding_size"
STORAGE_KEY_calibration_training_stage = "calibration_training_stage"
STORAGE_KEY_calibration_is_ood_indicators = "calibration_is_ood_indicators"
STORAGE_KEY_min_valid_qbin_for_class_conditional_accuracy = "min_valid_qbin_for_class_conditional_accuracy"

STORAGE_KEY_non_odd_thresholds = "non_odd_thresholds"
STORAGE_KEY_trueClass_To_dCDF = "trueClass_To_dCDF"
STORAGE_KEY_train_trueClass_To_dCDF = "train_trueClass_To_dCDF"
STORAGE_KEY_trueClass_To_unrescaledOutputCDF = "trueClass_To_unrescaledOutputCDF"
STORAGE_KEY_trueClass_To_qCumulativeSampleSizeArray = "trueClass_To_qCumulativeSampleSizeArray"
# STORAGE_KEY_trueClass_To_normalized_OutputCDF_non_ood = "trueClass_To_normalized_OutputCDF_non_ood"

STORAGE_KEY_is_gen_ai = "is_gen_ai"
STORAGE_KEY_gen_ai_vocab = "gen_ai_vocab"
STORAGE_KEY_global_embedding_size = "global_embedding_size"
STORAGE_KEY_composition_attributes_size = "composition_attributes_size"
STORAGE_KEY_top_logits_k = "top_logits_k"


# input embedding summary stats
STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_summary_stats = "training_embedding_summary_stats"
STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_mean = "training_embedding_mean"
STORAGE_KEY_SUMMARY_STATS_EMBEDDINGS_training_embedding_std = "training_embedding_std"

STORAGE_KEY_qdfLabelMarginalCategory_To_AcceptanceStatsOutputType = "qdfLabelMarginalCategory_To_AcceptanceStatsOutputType"
STORAGE_KEY_qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_class_AND_predictionConditionalIndicatorCount = "qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_class_AND_predictionConditionalIndicatorCount"
STORAGE_KEY_qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_predictionConditionalIndicatorCount = "qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_predictionConditionalIndicatorCount"
STORAGE_KEY_qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_classConditionalIndicatorCount = "qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_classConditionalIndicatorCount"
STORAGE_KEY_qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_acceptanceIterationN = "qdfLabelMarginalCategory_To_AcceptanceStatsOutputType_acceptanceIterationN"

# global summary statistics:
STORAGE_KEY_globalUncertaintyModelUUID = "globalUncertaintyModelUUID"
STORAGE_KEY_min_valid_qbin_across_iterations = "min_valid_qbin_across_iterations"
STORAGE_KEY_predicted_class_to_bin_to_median_output_magnitude_of_iteration = "predicted_class_to_bin_to_median_output_magnitude_of_iteration"
STORAGE_KEY_cauchy_quantile = "cauchy_quantile"

FILENAME_UNCERTAINTY_STATISTICS = "meta.json"
FILENAME_UNCERTAINTY_STATISTICS_AGGREGATE = "meta_aggregate.json"
FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LABELS = "support_labels.npy"
FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_PREDICTED = "support_predicted.npy"
FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_LOGITS = "support_logits.npy"
STORAGE_KEY_UNCERTAINTY_STATISTICS_SUPPORT_UUID = "support_ids"
FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_UUID = "support_ids.json"
FILENAME_UNCERTAINTY_STATISTICS_SUPPORT_INDEX = "support.npy"
# FILENAME_UNCERTAINTY_STATISTICS_Calibration_sample_size_tensor = "calibration_sample_size_class_"  # suffix is [label].pt

FILENAME_LOCALIZER = "compression_index.pt"  # localizer state dict
# FILENAME_LOCALIZER_PARAMS = "compression_keydict.pt"  # model params

FILENAME_UNCERTAINTY_STATISTICS_calibration_labels_TENSOR = "calibration_labels.pt"
FILENAME_UNCERTAINTY_STATISTICS_calibration_predicted_labels = "calibration_predicted_labels.pt"
STORAGE_KEY_UNCERTAINTY_STATISTICS_calibration_uuids = "calibration_uuids"
FILENAME_UNCERTAINTY_STATISTICS_calibration_uuids = "calibration_uuids.json"
FILENAME_UNCERTAINTY_STATISTICS_calibration_unrescaled_CDFquantiles = "calibration_unrescaled_CDFquantiles.pt"
FILENAME_UNCERTAINTY_STATISTICS_calibration_soft_qbins = "calibration_soft_qbins.pt"

FILENAME_GLOBAL_UNCERTAINTY_STATISTICS_JSON = "global_uncertainty_statistics.json"

DIRNAME_RUNNING_LLM_WEIGHTS_DIR = "non_finalized_llm_weights"

##### CategoryDisplayLabels
labelFull = "Label"
#predictionFull = "Prediction"  # c.f., predictedFull
calibratedProbabilityFull = "Calibrated Probability"
calibrationReliabilityFull = "Calibration Reliability"
predictedFull = "Predicted class"
qFull = "Similarity to Training (q)"
dFull = "Distance to Training (d)"
fFull = "f(x) Magnitude"
sizeFull = "Partition Size (in Calibration)"

# qShort = "Similarity"
# qVar = "q" # this should rarely be used
# dShort = "Distance"
# dVar = "d"
# fShort = "Magnitude" # generally use fFull
# fVar = "f(x)"
# # These two should be used sparingly:
# sizeShort = "Partition Size" # generally use sizeFull
# sizeVar = "size"

# JSON keys
JSON_KEY_UNCERTAINTY = "Uncertainty"
JSON_KEY_UNCERTAINTY_DETAILS = "Uncertainty Details"
# JSON_KEY_UNCERTAINTY_DETAILS_SUMMARY = "Summary"
# JSON_KEY_UNCERTAINTY_DETAILS_VALUES = "Values"
JSON_KEY_UNCERTAINTY_DETAILS_TRAINING_SUPPORT = "Training"


## INPUT JSON
INPUT_JSON_KEY_RETURN_TRAINING_DISTANCES = "return_training_distances"

#### Eval
EVAL_METRICS_KEY__MARGINAL_ACCURACY01 = "marginal_accuracy01"
EVAL_METRICS_KEY__CLASS_CONDITIONAL_ACCURACY01 = "class_conditional_accuracy01"
EVAL_METRICS_KEY__PREDICTION_CONDITIONAL_ACCURACY01 = "prediction_conditional_accuracy01"

### Split labels
SPLIT_LABEL_calibration_during_training = "Calibration (during training)"
