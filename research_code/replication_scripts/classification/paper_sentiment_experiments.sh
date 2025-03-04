#########################################################################################################
##################### Sentiment train and eval
#########################################################################################################


cd research_code/code  # Update with the applicable path

conda activate baseEnv1


RUN_SUFFIX_ID="iterations_final"
MODEL_TYPE="classifier"

DATA_DIR="/data/classification/sentiment"  # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/training_set.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.jsonl"

# the suffix __only_emb_ignore_other_fields is not important for the purposes here beyond simply indicating that the JSON objects have some additional fields (exported from a modified development version of the Reexpress one data analysis program) that are not relevant here

EVAL_LABEL="validation_set__only_emb_ignore_other_fields"  # primary test set
EVAL_LABEL="eval_set.__only_emb_ignore_other_fields"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields"  # OOD test set
EVAL_FILE="${DATA_DIR}/embeddings/${EVAL_LABEL}.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/paper/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001

# train:

python -u reexpress.py \
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
--use_embeddings > ${MODEL_OUTPUT_DIR}/run1.log.txt

# a copy of the original log is available in the model dir at paper/sentiment/iterations_final_classifier_0.95_1000/sagemaker/run1.log.txt



############### Run eval using each of the EVAL_FILE's, which are identified by the following (i.e., run three times, commenting out each EVAL_LABEL, in turn):

EVAL_LABEL="validation_set__only_emb_ignore_other_fields"  # primary test set
EVAL_LABEL="eval_set.__only_emb_ignore_other_fields"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields"  # OOD test set
EVAL_FILE="${DATA_DIR}/embeddings/${EVAL_LABEL}.jsonl"

python -u reexpress.py \
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
--use_embeddings \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"

echo ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"

# main test set:
# paper/sentiment/iterations_final_classifier_0.95_1000/sagemaker/run1.test.validation_set__only_emb_ignore_other_fields.version2.log.txt
# small test set:
# paper/sentiment/iterations_final_classifier_0.95_1000/sagemaker/run1.test.eval_set.__only_emb_ignore_other_fields.version2.log.txt
# OOD test set:
# paper/sentiment/iterations_final_classifier_0.95_1000/sagemaker/run1.test.SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.version2.log.txt
