
#########################################################################################################
##################### First, cache force-decoded files for training and evaluating the verificationLayer
#########################################################################################################
cd research_code/code  # Update with the applicable path

conda activate baseEnv1

RUN_SUFFIX_ID="phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined"
MODEL_TYPE="gen_ai"

DATA_DIR="data/sdm_network/sentiment_and_simple_facts_combined"

TRAIN_FILE="${DATA_DIR}/sentiment_rebalanced.jsonl"
CALIBRATION_FILE="${DATA_DIR}/factcheck_combined.jsonl"

EVAL_FILE="${DATA_DIR}/test/exported_ood_eval__only_emb_ignore_other_fields.jsonl"



ALPHA=0.95

MODEL_OUTPUT_DIR=llm_combined/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_"/sagemaker

mkdir -p "${MODEL_OUTPUT_DIR}"

DATA_CACHE_DIR="${DATA_DIR}/force_decoded_init"
mkdir -p "${DATA_CACHE_DIR}"

LEARNING_RATE=0.00001


GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"  # quantized for use with MLX
GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE="phi3.5/lm_head_weights.pt"  # copy of the non-quantized weights of the final linear-layer for training, as Pytorch tensors
EXEMPLAR_DIMENSION=1000
MAX_LENGTH=500


python -u reexpress.py \
--input_training_set_file ${TRAIN_FILE} \
--input_calibration_set_file ${CALIBRATION_FILE} \
--input_eval_set_file ${EVAL_FILE} \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 235 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--use_embeddings \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH} \
--gen_ai_model_lm_head_weights_file=${GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE} \
--max_length=${MAX_LENGTH} \
--gen_ai_vocab 32064 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--is_gen_ai \
--router_warm_up_epochs 0 \
--cache_directory="${DATA_CACHE_DIR}" \
--llmType=0 \
--cache_embeddings_for_classification_with_force_decoded_generation__document_level > ${MODEL_OUTPUT_DIR}/run1.log.txt

# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/run1.log.txt

# also cache the additional eval sets; Note the use of --only_cache_eval

for FILE_TO_CACHE in "eval_set.__only_emb_ignore_other_fields.jsonl" "validation_set__only_emb_ignore_other_fields.jsonl" "SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl"; do
echo ${FILE_TO_CACHE}
python -u reexpress.py \
--input_training_set_file="" \
--input_calibration_set_file="" \
--input_eval_set_file="${DATA_DIR}/test/${FILE_TO_CACHE}" \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 235 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--use_embeddings \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH} \
--gen_ai_model_lm_head_weights_file=${GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE} \
--max_length=${MAX_LENGTH} \
--gen_ai_vocab 32064 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--is_gen_ai \
--router_warm_up_epochs 0 \
--cache_directory="${DATA_CACHE_DIR}" \
--llmType=0 \
--cache_embeddings_for_classification_with_force_decoded_generation__document_level \
--only_cache_eval >> ${MODEL_OUTPUT_DIR}/run1.log.txt
done



#########################################################################################################
##################### Train the verificationLayer over the force-decoded data. As noted
##################### in the paper, this is a binary classification task.
#########################################################################################################

cd research_code/code  # Update with the applicable path

conda activate baseEnv1

RUN_SUFFIX_ID="phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined"
MODEL_TYPE="gen_ai"


DATA_DIR="data/sdm_network/sentiment_and_simple_facts_combined/force_decoded_init"

TRAIN_FILE="${DATA_DIR}/sentiment_rebalanced.jsonl"
CALIBRATION_FILE="${DATA_DIR}/factcheck_combined.jsonl"

EVAL_FILE="${DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl"

ALPHA=0.95

MODEL_OUTPUT_DIR=llm_combined/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_"/sagemaker

mkdir -p "${MODEL_OUTPUT_DIR}"

LEARNING_RATE=0.00001


GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"
GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE="phi3.5/lm_head_weights.pt"
EXEMPLAR_DIMENSION=1000
MAX_LENGTH=500


python -u reexpress.py \
--input_training_set_file ${TRAIN_FILE} \
--input_calibration_set_file ${CALIBRATION_FILE} \
--input_eval_set_file ${EVAL_FILE} \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 235 \
--epoch 50 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--use_embeddings \
--number_of_random_shuffles 10 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH} \
--gen_ai_model_lm_head_weights_file=${GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE} \
--max_length=${MAX_LENGTH} \
--gen_ai_vocab 32064 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--is_gen_ai \
--router_warm_up_epochs 0 \
--llmType=0 \
--init_gen_ai_model > ${MODEL_OUTPUT_DIR}/train_classifier_run1.log.txt

# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/train_classifier_run1.log.txt


#########################################################################################################
##################### Now we run the fine-tuning process for M^pos of the underlying LLM.
## Note that in all subsequent processes, we use the train and calibration data in the
## `best_iteration_data` as D_tr and D_ca, so that we use the same shuffling as that finalized by the
## J=10 iterations of the verificationLayer training process.
#########################################################################################################

cd research_code/code  # Update with the applicable path

conda activate baseEnv1

RUN_SUFFIX_ID="phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined"
MODEL_TYPE="gen_ai"


DATA_DIR="data/sdm_network/sentiment_and_simple_facts_combined/force_decoded_init"

EVAL_FILE="${DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl"

ALPHA=0.95

MODEL_OUTPUT_DIR=llm_combined/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_"/sagemaker

BEST_ITERATION_DATA_DIR="${MODEL_OUTPUT_DIR}/best_iteration_data"
echo "D_tr and D_ca are from the shuffled data of training the verificationLayer"
TRAIN_FILE="${BEST_ITERATION_DATA_DIR}/train.jsonl"
CALIBRATION_FILE="${BEST_ITERATION_DATA_DIR}/calibration.jsonl"


LEARNING_RATE=0.00001


GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"
GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE="phi3.5/lm_head_weights.pt"
EXEMPLAR_DIMENSION=1000
MAX_LENGTH=500


python -u reexpress.py \
--input_training_set_file ${TRAIN_FILE} \
--input_calibration_set_file ${CALIBRATION_FILE} \
--input_eval_set_file="" \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 235 \
--epoch 5 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--use_embeddings \
--number_of_random_shuffles 1 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH} \
--gen_ai_model_lm_head_weights_file=${GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE} \
--max_length=${MAX_LENGTH} \
--gen_ai_vocab 32064 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--is_gen_ai \
--router_warm_up_epochs 0 \
--llmType=0 \
--train_gen_ai_model \
--generation_directory="${MODEL_OUTPUT_DIR}/generation_dir" \
--gen_ai_training_min_beta=0.0 \
--gen_ai_training_max_beta=0.1 > ${MODEL_OUTPUT_DIR}/train_genai_run2.log.txt


#llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/train_genai_run2.log.txt

# The generations created during training are available here for reference:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/generation_dir

# FYI: use this if you want to re-start LLM training from scratch: --reset_gen_ai_model_weights

## The model dir can then be used as the path to experiment with generation with our small MLX generation script:
#adaptor_dir="llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker"
#model.add_adaptors(adaptor_dir)
#
## Similarly, the final running epoch (not best epoch) saved to ${MODEL_OUTPUT_DIR}/non_finalized_llm_weights can be used for comparison:
#adaptor_dir="llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/non_finalized_llm_weights"
#model.add_adaptors(adaptor_dir)

## FYI: To analyze the aforementioned generation files you can use the following script. (Note that the running per-epoch calibration task accuracies in the original saved logs in our archive directory do not take into account parsing errors, which are ignored, since we used the below script and used the former for debugging/analysis. However, this is changed in the current code to account for parsing errors, which is what is normally desired. That is, when you re-run the above, the accuracy values will match those from a_analyze_genai_generation_files.py below.):

FILE_TO_ANALYZE_DIR="llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/generation_dir"

for FILE_TO_ANALYZE_LABEL in "init_" "epoch_1" "epoch_2" "epoch_3" "epoch_4" "epoch_5"; do
echo ${FILE_TO_ANALYZE_LABEL}
FILE_TO_ANALYZE="${FILE_TO_ANALYZE_DIR}/${FILE_TO_ANALYZE_LABEL}_calibration.jsonl"
python -u a_analyze_genai_generation_files.py \
--input_filename ${FILE_TO_ANALYZE}
done


#init_
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.7996713229252259,     out of 6085
#epoch_1
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.8305669679539852,     out of 6085
#epoch_2
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.8414133114215283,     out of 6085
#epoch_3
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.8529170090386196,     out of 6085
#epoch_4
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.8657354149548069,     out of 6085
#epoch_5
#Marginal accuracy, across underlying tasks (i.e., not verification):     mean: 0.8746096959737059,     out of 6085


#########################################################################################################
##################### Eval
## Remember to change generation_directory to avoid overwritting the intermediate files from training
#########################################################################################################

cd research_code/code  # Update with the applicable path

conda activate baseEnv1

RUN_SUFFIX_ID="phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined"
MODEL_TYPE="gen_ai"

DATA_DIR="data/sdm_network/sentiment_and_simple_facts_combined/force_decoded_init"

# Comment/Uncomment each EVAL_FILE_FILE_NAME in turn to run evaluation
EVAL_FILE_FILE_NAME="SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl"  # sentiment OOD
EVAL_FILE_FILE_NAME="eval_set.__only_emb_ignore_other_fields.jsonl"  # sentiment in-domain small test
EVAL_FILE_FILE_NAME="exported_ood_eval__only_emb_ignore_other_fields.jsonl"  # factcheck OOD test
EVAL_FILE_FILE_NAME="validation_set__only_emb_ignore_other_fields.jsonl"  # sentiment main in-domain test
EVAL_FILE="${DATA_DIR}/${EVAL_FILE_FILE_NAME}"

ALPHA=0.95

MODEL_OUTPUT_DIR=llm_combined/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}_"/sagemaker

BEST_ITERATION_DATA_DIR="${MODEL_OUTPUT_DIR}/best_iteration_data"
echo "Continued from shuffled data"
TRAIN_FILE="${BEST_ITERATION_DATA_DIR}/train.jsonl"
CALIBRATION_FILE="${BEST_ITERATION_DATA_DIR}/calibration.jsonl"


LEARNING_RATE=0.00001


GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"
GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE="phi3.5/lm_head_weights.pt"
EXEMPLAR_DIMENSION=1000
MAX_LENGTH=500


python -u reexpress.py \
--input_training_set_file ${TRAIN_FILE} \
--input_calibration_set_file ${CALIBRATION_FILE} \
--input_eval_set_file=${EVAL_FILE} \
--alpha=${ALPHA} \
--class_size 2 \
--seed_value 235 \
--epoch 5 \
--batch_size 50 \
--learning_rate ${LEARNING_RATE} \
--model_dir "${MODEL_OUTPUT_DIR}" \
--use_embeddings \
--number_of_random_shuffles 1 \
--use_training_set_max_label_size_as_max_q \
--warm_up_epochs 0 \
--model_rescaler_training_max_epochs 1000 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH} \
--gen_ai_model_lm_head_weights_file=${GEN_AI_MODEL_LM_HEAD_WEIGHTS_FILE} \
--max_length=${MAX_LENGTH} \
--gen_ai_vocab 32064 \
--exemplar_vector_dimension ${EXEMPLAR_DIMENSION} \
--is_gen_ai \
--router_warm_up_epochs 0 \
--llmType=0 \
--generation_directory="${MODEL_OUTPUT_DIR}/generation_dir_post_hoc" \
--gen_ai_training_min_beta=0.0 \
--gen_ai_training_max_beta=0.1 \
--eval_only \
--eval_gen_ai > ${MODEL_OUTPUT_DIR}/"${EVAL_FILE_FILE_NAME}_eval_genai_sentiment_run1.log.txt"


# NOTE THAT THE LOG FILES INCLUDE BOTH FORCE-DECODED VERIFICATION AND THE ANALYSIS OF THE GENERATED DATA AT THE ORIGINAL TASK LEVEL

# factcheck OOD test:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/exported_ood_eval__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt

# sentiment in-domain small test:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/eval_set.__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt

# sentiment main in-domain test:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/validation_set__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt

# sentiment OOD:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt

# Factcheck:
# llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/exported_ood_eval__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt

#########################################################################################################
##################### For the paper and comparison, we also need to run eval from before fine-tuning.
##################### To do so, we need to copy the model directory and the reset the LLM weights with
##################### --reset_gen_ai_model_weights. We can then run evaluation as in the previous
##################### block. To reduce the size of the archive, we only save the log files in the model directory.
#########################################################################################################

# copy the directory and then reset weights with --reset_gen_ai_model_weights. Next, repeat the process in the previous block.
#cp -r llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_ llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95__no_finetuning

# use the following for the MODEL_OUTPUT_DIR. All other lines are the same as above:
MODEL_OUTPUT_DIR=llm_combined/"${RUN_SUFFIX_ID}_${MODEL_TYPE}_${ALPHA}__no_finetuning"/sagemaker

# NOTE THAT THE LOG FILES INCLUDE BOTH FORCE-DECODED VERIFICATION AND THE ANALYSIS OF THE GENERATED DATA AT THE ORIGINAL TASK LEVEL

## factcheck test:
#/llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95__no_finetuning/sagemaker /sagemaker/exported_ood_eval__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt
#
## sentiment in-domain small test:
#llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95__no_finetuning/sagemaker /eval_set.__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt
#
## sentiment main in-domain test:
#llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95__no_finetuning/sagemaker /validation_set__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt
#
## sentiment OOD:
#llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95__no_finetuning/sagemaker /SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl_eval_genai_sentiment_run1.log.txt


