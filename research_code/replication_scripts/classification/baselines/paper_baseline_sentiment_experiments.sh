#########################################################################################################
##################### Benchmark model (standard 1-D CNN adaptor) Sentiment train and eval
#########################################################################################################


# Constructing the baselines requires some minor changes to the main code, which we describe here to avoid complicating the main code. Note that the output values are available in our archived data/model directory.
# We set kUSE_STANDARD_CROSS_ENTROPY = True
# We use --use_balanced_accuracy
# We set --number_of_random_shuffles=1
# This is inefficient since we calculate distances which are then discarded, but this is a simple way to ensure the code and model are otherwise identical.
# For input to the downstream 3rd party calibration scripts, it is also necessary to save the un-normalized output log probabilities (logits), which are then used for input to downstream 3rd party scripts. This can be achieved by just creating another forward_type that returns the batch_f_positive values. For example, in forward(), add the following, which can then be called to save the output at test-time via output_for_baselines(), below:
#       if forward_type == "baseline_logits_output":
#            return batch_f_positive
#        # before:
#        if forward_type == constants.FORWARD_TYPE_SINGLE_PASS_TEST:
# The following can be placed in utils_test.py and used in place of utils_test.test():
#def output_for_baselines(options, main_device):
#    model = utils_model.load_model_torch(options.model_dir, torch.device("cpu"), load_for_inference=True)
#    test_meta_data, _ = \
#        utils_preprocess.get_metadata_lines(options, options.input_eval_set_file,
#                                            reduce=False,
#                                            use_embeddings=options.use_embeddings,
#                                            concat_embeddings_to_attributes=options.concat_embeddings_to_attributes,
#                                            calculate_summary_stats=False, is_training=False)
#    test_embeddings = test_meta_data["embeddings"].to(main_device)
#    test_labels = torch.tensor(test_meta_data["labels"]).to(main_device)
#    assert test_embeddings.shape[0] == test_labels.shape[0]
#    json_lines = []
#    instance_i = -1
#    for test_embedding, test_label in zip(test_embeddings, test_labels):
#        instance_i += 1
#        true_test_label = test_label.item()
#        output_logits = \
#            model(test_embedding.unsqueeze(0),
#                  forward_type="baseline_logits_output")
#        json_obj = {
#            "id": test_meta_data['uuids'][instance_i],
#            "document": test_meta_data['lines'][instance_i],
#            "logits": output_logits.squeeze(0).detach().cpu().numpy().tolist(),
#            "label": true_test_label,
#            "prediction": torch.argmax(output_logits, -1).item()
#        }
#        json_lines.append(json_obj)
#    if options.output_logits_file != "" and len(json_lines) > 0:
#        utils_model.save_json_lines(options.output_logits_file, json_lines)

# We rename the resulting main file reexpress_baseline.py in a new directory code_baseline, with the additional options of --output_logits_file and --output_for_baselines. The following shows is provided to make clear the hyper-parameters that were used.
        
# Also, when re-calibrating with 3rd party approaches, remember to use the shuffled data in best_iteration_data.
# Internal note: this is ..._gen_baseline




cd research_code/code_baseline  # Update with the applicable path (see notes above)

conda activate baseEnv1


RUN_SUFFIX_ID="baseline_conformal_temp_scaling"
MODEL_TYPE="classifier"

DATA_DIR="/data/classification/sentiment"  # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/training_set.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.jsonl"

EVAL_LABEL="validation_set__only_emb_ignore_other_fields"  # primary test set
EVAL_LABEL="eval_set.__only_emb_ignore_other_fields"  # a small eval set
EVAL_LABEL="SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields"  # OOD test set
EVAL_FILE="${DATA_DIR}/embeddings/${EVAL_LABEL}.jsonl"

ALPHA=0.95
EXEMPLAR_DIMENSION=1000

MODEL_OUTPUT_DIR=/paper/baselines_codebase/sentiment/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_${EXEMPLAR_DIMENSION}"/sagemaker  # Update with the applicable path

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
--use_embeddings \
--use_balanced_accuracy > ${MODEL_OUTPUT_DIR}/run1.log.txt

# a copy of the original log is available in the model dir at sentiment/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.log.txt
# Note that the analysis output in the log file from the rescaler can be ignored, since q and d are never assigned.

# output test sets
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
--use_embeddings \
--output_logits_file=${MODEL_OUTPUT_DIR}/"${EVAL_LABEL}.output_logits.jsonl" \
--output_for_baselines \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"

echo ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"
echo ${MODEL_OUTPUT_DIR}/"${EVAL_LABEL}.output_logits.jsonl"

### Output log files and the cached output logits are available in the directory:
## main test set:
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.test.validation_set__only_emb_ignore_other_fields.version2.log.txt
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/validation_set__only_emb_ignore_other_fields.output_logits.jsonl
## small test set:
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.test.eval_set.__only_emb_ignore_other_fields.version2.log.txt
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/eval_set.__only_emb_ignore_other_fields.output_logits.jsonl
## OOD test set:
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.output_logits.jsonl
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.test.SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.version2.log.txt


# Similarly, in this case, we also need to run eval on the calibration set in order to cache the output logits:
EVAL_LABEL="calibration"
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
--use_embeddings \
--output_logits_file=${MODEL_OUTPUT_DIR}/"${EVAL_LABEL}.output_logits.jsonl" \
--output_for_baselines \
--eval_only > ${MODEL_OUTPUT_DIR}/"run1.test.${EVAL_LABEL}.version2.log.txt"

# cached output logits:
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/calibration.output_logits.jsonl
# log:
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/run1.test.calibration.version2.log.txt


#########################################################################################################
############################################# conformal and temperature scaling baselines -- sentiment
#########################################################################################################

# We include a version of the MIT licensed code of https://github.com/aangelopoulos/conformal-classification at /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines. This does not make any substantive changes to the underlying conformal methods, and exists to simply add a wrapper script `baseline_comp_local.py` to read in the cached output logits from above.


cd /third_party_baselines/conformal/conformal_classification-master_modified_for_2025_baselines  # Update with the applicable path (see notes above)
DATA_DIR="/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker/"

# here, we iteratively re-run, which is inefficient, but the results are the same for the model given the fixed seed

for EVAL_LABEL in "validation_set__only_emb_ignore_other_fields" "eval_set.__only_emb_ignore_other_fields" "SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields"; do
    echo ${EVAL_LABEL}
    CALIBRATION_FILE="${DATA_DIR}/calibration.output_logits.jsonl"
    EVAL_FILE="${DATA_DIR}/${EVAL_LABEL}.output_logits.jsonl"

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
    --probability_threshold ${COVERAGE} > ${OUTPUT_DIR}/${EVAL_LABEL}_log_${COVERAGE}_${L_CRITERION}.txt

    echo ${OUTPUT_DIR}/${EVAL_LABEL}_log_${COVERAGE}_${L_CRITERION}.txt

    python -u baseline_comp_local.py \
    --calibration_files ${CALIBRATION_FILE} \
    --eval_files ${EVAL_FILE} \
    --batch_size 50 \
    --seed 0 \
    --number_of_classes 2 \
    --empirical_coverage ${COVERAGE} \
    --lambda_criterion ${L_CRITERION} \
    --run_aps_baseline \
    --probability_threshold ${COVERAGE} > ${OUTPUT_DIR}/${EVAL_LABEL}_log_${COVERAGE}_aps_baseline.txt

    echo ${OUTPUT_DIR}/${EVAL_LABEL}_log_${COVERAGE}_aps_baseline.txt
done

# The output logs are saved to `output_logs`:
# For validation_set__only_emb_ignore_other_fields
#
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//validation_set__only_emb_ignore_other_fields_log_0.95_adaptiveness.txt
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//validation_set__only_emb_ignore_other_fields_log_0.95_aps_baseline.txt
#
# For eval_set.__only_emb_ignore_other_fields
#
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//eval_set.__only_emb_ignore_other_fields_log_0.95_adaptiveness.txt
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//eval_set.__only_emb_ignore_other_fields_log_0.95_aps_baseline.txt
#
# For SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields
#
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields_log_0.95_adaptiveness.txt
#/baseline_conformal_temp_scaling_classifier_0.95_1000/sagemaker//output_logs//SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields_log_0.95_aps_baseline.txt

