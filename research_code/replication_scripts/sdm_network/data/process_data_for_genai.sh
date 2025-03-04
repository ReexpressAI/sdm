# It is unlikely you will need these since you can use our provided processed versions, but we include the original data processing for reference.

# To use the following, move the files in `support_code/aux_move_to_main_dir_if_needed` into the main `code` directory.

#########################################################################################################
##################### SENTIMENT: one-time processing for a constant mask for constructing negatives
### This is the same data as in paper_sentiment_experiments.sh
#########################################################################################################
cd research_code/code  # Update with the applicable path

conda activate baseEnv1

GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"  # path to quantized MLX Phi-3.5

DATA_DIR="data/classification/sentiment"

TRAIN_FILE="${DATA_DIR}/training_set.jsonl"
CALIBRATION_FILE="${DATA_DIR}/calibration_set.jsonl"

EVAL_FILE="${DATA_DIR}/embeddings/validation_set__only_emb_ignore_other_fields.jsonl"

EVAL_FILE="${DATA_DIR}/embeddings/eval_set.__only_emb_ignore_other_fields.jsonl"
EVAL_FILE="${DATA_DIR}/embeddings/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl"


OUTPUT_DATA_DIR="data/sdm_network/sentiment/genai_preprocess"
mkdir ${OUTPUT_DATA_DIR}/embeddings

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename ${TRAIN_FILE} \
--output_filename "${OUTPUT_DATA_DIR}/training_set.jsonl" \
--taskCategory 0 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}
       
#Total negative constructions 1746 out of 3414

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename ${CALIBRATION_FILE} \
--output_filename "${OUTPUT_DATA_DIR}/calibration_set.jsonl" \
--taskCategory 0 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 7231 out of 14250

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename "${DATA_DIR}/embeddings/validation_set__only_emb_ignore_other_fields.jsonl" \
--output_filename "${OUTPUT_DATA_DIR}/embeddings/validation_set__only_emb_ignore_other_fields.jsonl" \
--taskCategory 0 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 806 out of 1583

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename "${DATA_DIR}/embeddings/eval_set.__only_emb_ignore_other_fields.jsonl" \
--output_filename "${OUTPUT_DATA_DIR}/embeddings/eval_set.__only_emb_ignore_other_fields.jsonl" \
--taskCategory 0 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 254 out of 488

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename "${DATA_DIR}/embeddings/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl" \
--output_filename "${OUTPUT_DATA_DIR}/embeddings/SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl" \
--taskCategory 0 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}
        
#Total negative constructions 2422 out of 4750



#########################################################################################################
##################### FACTCHECK: one-time processing for a constant mask for constructing negatives
### This is the same data as in paper_factcheck_experiments.sh
#########################################################################################################
cd research_code/code  # Update with the applicable path

conda activate baseEnv1

GEN_AI_MODEL_PATH="phi3.5/microsoft--Phi-3.5-mini-instruct_mlx"  # path to quantized MLX Phi-3.5

DATA_DIR="/data/classification/factcheck" # Update with the applicable path

TRAIN_FILE="${DATA_DIR}/random_shuffle_train.jsonl"
CALIBRATION_FILE="${DATA_DIR}/random_shuffle_calibration.jsonl"
EVAL_FILE="${DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl"


OUTPUT_DATA_DIR="data/sdm_network/factcheck/genai_preprocess"
mkdir ${OUTPUT_DATA_DIR}/random_shuffle

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename ${TRAIN_FILE} \
--output_filename "${OUTPUT_DATA_DIR}/random_shuffle_train.jsonl" \
--taskCategory 1 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 1563 out of 3042

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename ${CALIBRATION_FILE} \
--output_filename "${OUTPUT_DATA_DIR}/random_shuffle_calibration.jsonl" \
--taskCategory 1 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 1563 out of 3043

python -u a_process_reexpress_format_for_genai_verification.py \
--input_filename ${EVAL_FILE} \
--output_filename "${OUTPUT_DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl" \
--taskCategory 1 \
--llmType 0 \
--gen_ai_model_path=${GEN_AI_MODEL_PATH}

#Total negative constructions 122 out of 245

#########################################################################################################
##################### Sentiment+FACTCHECK: balance the training and calibration sets
#########################################################################################################

# Here, we will keep the Sentiment and Factcheck files separate, and one can be used as the initial
# Train and the other as Calibration, since these get randomly shuffled together during the training loop.

cd research_code/code  # Update with the applicable path

conda activate baseEnv1

OUTPUT_DATA_DIR="data/sdm_network/sentiment_and_simple_facts_combined"
mkdir -p ${OUTPUT_DATA_DIR}

TASK0_DATA_DIR="data/sdm_network/sentiment/genai_preprocess"
TASK1_DATA_DIR="data/sdm_network/factcheck/genai_preprocess"


python -u a_process_reexpress_format_for_genai_verification__balance_cal_train_across_tasks.py \
--input_task0_filename0="${TASK0_DATA_DIR}/training_set.jsonl" \
--input_task0_filename1="${TASK0_DATA_DIR}/calibration_set.jsonl" \
--input_task1_filename0="${TASK1_DATA_DIR}/random_shuffle_train.jsonl" \
--input_task1_filename1="${TASK1_DATA_DIR}/random_shuffle_calibration.jsonl" \
--seed_value=0 \
--output_task0_filename="${OUTPUT_DATA_DIR}/sentiment_rebalanced.jsonl" \
--output_task1_filename="${OUTPUT_DATA_DIR}/factcheck_combined.jsonl"

#Task 1: count of class 0: 3126
#Task 1: count of class 1: 2959
#Task 0: count of class 0: 8977
#Task 0: count of class 1: 8687

# for convenience, also add the test sets to the same directory:
mkdir "${OUTPUT_DATA_DIR}/test"

cp ${TASK0_DATA_DIR}/embeddings/"SemEval2017-task4-test.subtask-A.english.txt.binaryevalformat.balanced.txt.__only_emb_ignore_other_fields.jsonl" "${OUTPUT_DATA_DIR}/test/"
cp ${TASK0_DATA_DIR}/embeddings/"eval_set.__only_emb_ignore_other_fields.jsonl" "${OUTPUT_DATA_DIR}/test/"
cp ${TASK0_DATA_DIR}/embeddings/"validation_set__only_emb_ignore_other_fields.jsonl" "${OUTPUT_DATA_DIR}/test/"

cp "${TASK1_DATA_DIR}/exported_ood_eval__only_emb_ignore_other_fields.jsonl" "${OUTPUT_DATA_DIR}/test/"
        
