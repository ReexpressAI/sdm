#########################################################################################################
##################### Benchmark model (standard 1-D CNN adaptor) Factcheck train and eval
#########################################################################################################

# First, read the file `paper_baseline_sentiment_experiments.sh` to see how the baseline code is created. The process here is the same as that for the sentiment datasets, but using the factcheck dataset.


cd research_code/code_baseline  # Update with the applicable path (see notes above)

conda activate baseEnv1


RUN_SUFFIX_ID="baseline_conformal_temp_scaling"
MODEL_TYPE="classifier"

DATA_DIR="/data/classification/factcheck" # Update with the applicable path

# 'embedding' field is from the Reexpress one encoder-decoder, and 'attributes' field is from Mixtral. These are concatenated together using the --concat_embeddings_to_attributes flag. We ignore the 'exemplar' field, which is output from the adaptor of Reexpress one.
TRAIN_FILE="${DATA_DIR}/random_shuffle_train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/random_shuffle_calibration.jsonl"
EVAL_FILE="${DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/paper/baselines_codebase/factcheck/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001


python -u reexpress_baseline.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 1 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--concat_embeddings_to_attributes \
--use_balanced_accuracy > ${MODEL_OUTPUT_DIR}/run1.log.txt

# baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.log.txt



OUTPUT_LOGITS_LABEL="eval"
python -u reexpress_baseline.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--concat_embeddings_to_attributes \
--output_logits_file=${MODEL_OUTPUT_DIR}/"${OUTPUT_LOGITS_LABEL}.output_logits.jsonl" \
--output_for_baselines \
--eval_only

# Note that eval output acc is in /baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.log.txt

OUTPUT_LOGITS_LABEL="calibration"
# note the --input_eval_set_file uses the best_iteration_data directory
python -u reexpress_baseline.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file ${MODEL_OUTPUT_DIR}/"best_iteration_data/calibration.jsonl" \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 0 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--concat_embeddings_to_attributes \
--output_logits_file=${MODEL_OUTPUT_DIR}/"${OUTPUT_LOGITS_LABEL}.output_logits.jsonl" \
--output_for_baselines \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.best_iteration_calibration_file.version2.log.txt"

# /baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.test.calibration_file.version2.log.txt


# Calibration and Eval files with output logits:

#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/calibration.output_logits.jsonl
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/eval.output_logits.jsonl

#########################################################################################################
############################################# conformal and temperature scaling baselines -- factcheck
#########################################################################################################

# We include a version of the MIT licensed code of https://github.com/aangelopoulos/conformal-classification at /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines. This does not make any substantive changes to the underlying conformal methods, and exists to simply add a wrapper script `baseline_comp_local.py` to read in the cached output logits from above.

cd /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines  # Update with the applicable path (see notes above)

DATA_DIR="/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/"

CALIBRATION_FILE="${DATA_DIR}/calibration.output_logits.jsonl"
EVAL_FILE="${DATA_DIR}/eval.output_logits.jsonl"

COVERAGE=0.95
L_CRITERION='adaptiveness'
OUTPUT_DIR=${DATA_DIR}/output_logs/
mkdir ${OUTPUT_DIR}

python -u baseline_comp_local.py \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--probability_threshold ${COVERAGE} > ${OUTPUT_DIR}/log_${COVERAGE}_${L_CRITERION}.txt

# /baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//log_0.95_adaptiveness.txt

python -u baseline_comp_local.py \
--calibration_files ${CALIBRATION_FILE} \
--eval_files ${EVAL_FILE} \
--batch_size 50 \
--seed 0 \
--number_of_classes 2 \
--empirical_coverage ${COVERAGE} \
--lambda_criterion ${L_CRITERION} \
--run_aps_baseline \
--probability_threshold ${COVERAGE} > ${OUTPUT_DIR}/log_${COVERAGE}_aps_baseline.txt

# /baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//log_0.95_aps_baseline.txt
