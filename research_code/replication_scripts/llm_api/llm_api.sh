#########################################################################################################
##################### LLM API train and eval
#########################################################################################################

cd research_code/code  # Update with the applicable path

conda activate baseEnv1

RUN_SUFFIX_ID="llm_branching_4options_v1_iterations_final"
MODEL_TYPE="classifier"


DATA_DIR="/mmlu_openai_v2_4option/processed_combined" # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/mmlu_train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/combined_mmlu_devval_mmlu_pro_val.jsonl"

EVAL_LABEL="mmlu_pro_test"
#EVAL_LABEL="mmlu_test"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/llm_api/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path


mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

# Note this only uses the "attributes" field in the input JSON lines files because neither --concat_embeddings_to_attributes nor --use_embeddings are provided as flags.
python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
--class_size 4 \
--seed_value 0 \
--epoch 5 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 > ${MODEL_OUTPUT_DIR}/run1.log.txt

#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/run1.log.txt

## eval on each test set, in-turn:
EVAL_LABEL="mmlu_pro_test"
#EVAL_LABEL="mmlu_test"
EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.jsonl"

python -u reexpress.py \
--input_training_set_file "${TRAIN_FILE}" \
--input_calibration_set_file "${CALIBRATION_FILE}" \
--input_eval_set_file "${EVAL_FILE}" \
--alpha=${ALPHA} \
--class_size 4 \
--seed_value 0 \
--epoch 5 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--router_warm_up_epochs 0 \
--label_error_file=${MODEL_OUTPUT_DIR}/"test.${EVAL_LABEL}.possible_label_errors.jsonl" \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"

### MMLU
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/run1.test.mmlu_test.version2.log.txt
## possible label annotation errors:
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/test.mmlu_test.possible_label_errors.jsonl
#
### MMLU pro (4qa):
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/run1.test.mmlu_pro_test.version2.log.txt
## possible label annotation errors:
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/test.mmlu_pro_test.possible_label_errors.jsonl



## using an older version of the eval script:
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/run1.test.mmlu_test.log.txt
#/llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/run1.test.mmlu_pro_test.log.txt


#########################################################################################################
##################### MMLU PRO -- Analyze candidate error annotations
#########################################################################################################

cd research_code/code  # Update with the applicable path

conda activate baseEnv1


ERROR_FILE="llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/test.mmlu_pro_test.possible_label_errors.jsonl"

python -u llm_branching_examine_annotation_errors.py \
--dataset="mmlu_pro" \
--input_candidate_label_annotation_error_file=${ERROR_FILE} \
--category_restriction="computer science" > ${ERROR_FILE}.cs_subset.log.txt

# llm_api/llm_branching_4options_v1_iterations_final_classifier_0.95_1000/sagemaker/test.mmlu_pro_test.possible_label_errors.jsonl.cs_subset.log.txt

python -u llm_branching_examine_annotation_errors.py \
--dataset="mmlu_pro" \
--input_candidate_label_annotation_error_file=${ERROR_FILE} \
--category_restriction="" > ${ERROR_FILE}.all_questions.log.txt
