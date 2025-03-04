#########################################################################################################
##################### Factcheck train and eval
#########################################################################################################


cd research_code/code  # Update with the applicable path

conda activate baseEnv1


RUN_SUFFIX_ID="iterations_final"
MODEL_TYPE="classifier"

DATA_DIR="/data/classification/factcheck" # Update with the applicable path

# 'embedding' field is from the Reexpress one encoder-decoder, and 'attributes' field is from Mixtral. These are concatenated together using the --concat_embeddings_to_attributes flag. We ignore the 'exemplar' field, which is output from the adaptor of Reexpress one.
TRAIN_FILE="${DATA_DIR}/random_shuffle_train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/random_shuffle_calibration.jsonl"
EVAL_FILE="${DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl"


ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/factcheck/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path

mkdir -p "${MODEL_OUTPUT_DIR}"


LEARNING_RATE=0.00001


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
--concat_embeddings_to_attributes > ${MODEL_OUTPUT_DIR}/run1.log.txt


# a copy of the original log is available in the model dir at paper/factcheck/iterations_final_classifier_0.95_1000/sagemaker/run1.log.txt



############### Run eval


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
--concat_embeddings_to_attributes \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.eval_file.version2.log.txt"

# eval log for test set:
# paper/factcheck/iterations_final_classifier_0.95_1000/sagemaker/run1.test.eval_file.version2.log.txt
