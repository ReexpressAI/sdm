## It is unlikely these will be needed, but we include for reference how the data was collected from the Azure API. This assumes the openai package is installed and an Azure endpoint has been setup, for which we recommend the official Azure documentation. We include this as it was used in practice, noting that there was some re-starting of some of the data shards; adjust as applicable if re-running. The prompts used are in llm_branching.py.

# To use the following, move the files in `support_code/aux_move_to_main_dir_if_needed` into the main `code` directory.

# Adjust environment variables for output directories, input files, etc. as applicable.

# As noted in the paper, there are some full refusals from the API on the social science questions due to their text triggering the content filter. We treat these as wrong predictions in downstream analyses.

# We also include embeddings for text-embedding-3-large, although we do not use them in the experiments.

# The data archive includes the final processed data in the `processed_combined` directory, with "embedding" the text-embedding-3-large embeddings and "attributes" the length 7 vector used in the experiments.

#########################################################################################################
##################### MMLU for eval
#########################################################################################################

# The file mmlu.csv comes from the link in https://github.com/openai/simple-evals/blob/main/mmlu_eval.py

source setup.sh  # You have to create this. It should export environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


cd research_code/code  # Update with the applicable path

OUTPUT_DIR=/mmlu_openai_v2/eval_data_mmlu
mkdir ${OUTPUT_DIR}

python -u llm_branching.py \
--dataset="mmlu" \
--input_file="/mmlu_openai/mmlu.csv" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} >> ${OUTPUT_DIR}/log.txt

#/mmlu_openai_v2/eval_data_mmlu/log.txt

# continued above in shards:

# partitions of data:
NUM_SHARDS=50
# note starting at 1 not 0
for ((i=1;i<NUM_SHARDS;i++)); do
    echo ${OUTPUT_DIR}/log_${i}.txt
    python -u llm_branching.py \
--dataset="mmlu" \
--input_file="/mmlu_openai/mmlu.csv" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=${i} \
--total_shards=${NUM_SHARDS} > ${OUTPUT_DIR}/log_${i}.txt &
done

# Wait for all background processes to finish
wait

echo "All runs completed."



echo "The above script appends to the output."


OUTPUT_DIR=/mmlu_openai_v2/eval_data_mmlu
OUTPUT_PROCESSED_DIR="/mmlu_openai_v2_4option/eval_data_mmlu"
mkdir -p ${OUTPUT_PROCESSED_DIR}
MODEL=1
# first section
python -u llm_branching_4option_eval.py \
--input_file="${OUTPUT_DIR}/mmlu_eval_depth_0.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}
        
        
NUM_SHARDS=50
# note starting at 22 not 0
for ((i=22;i<NUM_SHARDS;i++)); do
    echo "shard${i}.jsonl"
    python -u llm_branching_4option_eval.py \
--input_file="${OUTPUT_DIR}/mmlu_eval_depth_0_shard${i}.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}
done


        
        
#########################################################################################################
##################### MMLU for dev/val
#########################################################################################################

source setup.sh  # You have to create this. It should export environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


cd research_code/code  # Update with the applicable path

OUTPUT_DIR=/mmlu_openai_v2/val_data_mmlu
mkdir ${OUTPUT_DIR}

python -u llm_branching.py \
--dataset="mmlu_val" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} >> ${OUTPUT_DIR}/log.txt

#val_data_mmlu/log.txt

echo "The above script appends to the output."

OUTPUT_DIR=/mmlu_openai_v2/val_data_mmlu
OUTPUT_PROCESSED_DIR="/mmlu_openai_v2_4option/val_data_mmlu"
mkdir ${OUTPUT_PROCESSED_DIR}
MODEL=1
for FILE_PREFIX in "mmlu_dev_depth_" "mmlu_validation_depth_"; do
python -u llm_branching_4option_eval.py \
--input_file="${OUTPUT_DIR}/${FILE_PREFIX}0.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}
done

#########################################################################################################
##################### MMLU for training
#########################################################################################################

source setup.sh  # You have to create this. It should export environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


cd research_code/code  # Update with the applicable path

OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu
mkdir ${OUTPUT_DIR}
#SHARD=1
#python -u llm_branching.py \
#--dataset="preprocess_mmlu_aux_train" \
#--max_depth=1 \
#--output_dir=${OUTPUT_DIR} \
#--shard=${SHARD} \
#--total_shards=50 >> ${OUTPUT_DIR}/log.txt


echo "The above script appends to the output."


#!/bin/bash

# partitions of data:
NUM_SHARDS=50
# 0 is below
for ((i=1;i<NUM_SHARDS;i++)); do
    echo ${OUTPUT_DIR}/log_${i}.txt
    python -u llm_branching.py \
--dataset="preprocess_mmlu_aux_train" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=${i} \
--total_shards=${NUM_SHARDS} > ${OUTPUT_DIR}/log_${i}.txt &
done

# Wait for all background processes to finish
wait

echo "All runs completed."

# also shard 0
python -u llm_branching.py \
--dataset="preprocess_mmlu_aux_train" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=0 \
--total_shards=${NUM_SHARDS} > ${OUTPUT_DIR}/log_0.txt

#/mmlu_openai_v2/train_data_mmlu/log_0.txt
#
#
#/mmlu_openai_v2/train_data_mmlu


OUTPUT_DIR=/mmlu_openai_v2_incomplete_bu_2025_02_02/train_data_mmlu  # temporary dir to complete run

python -u llm_branching_eval.py \
--input_file="${OUTPUT_DIR}/mmlu_auxiliary_train_depth_0_shard10.jsonl"


OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu
OUTPUT_PROCESSED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/train_data_mmlu"
mkdir ${OUTPUT_PROCESSED_DIR}
MODEL=1
NUM_SHARDS=50
for ((i=0;i<NUM_SHARDS;i++)); do
echo "shard${i}.jsonl"
python -u llm_branching_4option_eval.py \
--input_file="${OUTPUT_DIR}/mmlu_auxiliary_train_depth_0_shard${i}.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}
done


#########################################################################################################
##################### MMLU PRO -- This handles both test and validation
#########################################################################################################

source setup.sh  # You have to create this. It should export environment variables for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT


cd research_code/code  # Update with the applicable path

OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu_pro
mkdir ${OUTPUT_DIR}
SHARD=0
python -u llm_branching.py \
--dataset="mmlu_pro" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=${SHARD} \
--total_shards=50 >> ${OUTPUT_DIR}/log.txt

/mmlu_openai_v2/train_data_mmlu_pro/log.txt
echo "The above script appends to the output."



# partitions of data:
NUM_SHARDS=50
# shard 1 is above
for ((i=1;i<NUM_SHARDS;i++)); do
    echo ${OUTPUT_DIR}/log_${i}.txt
    python -u llm_branching.py \
--dataset="mmlu_pro" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=${i} \
--total_shards=${NUM_SHARDS} > ${OUTPUT_DIR}/log_${i}.txt &
done

# Wait for all background processes to finish
wait

echo "All runs completed."

# rerun of 16
OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu_pro
mkdir ${OUTPUT_DIR}
SHARD=16
python -u llm_branching.py \
--dataset="mmlu_pro" \
--max_depth=1 \
--output_dir=${OUTPUT_DIR} \
--shard=${SHARD} \
--total_shards=50 >> ${OUTPUT_DIR}/log_${SHARD}.txt

#/mmlu_openai_v2/train_data_mmlu_pro/log_16.txt
#
#/mmlu_openai_v2/train_data_mmlu_pro

OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu_pro
# here we fix the dir naming, splitting validation from the eval:
OUTPUT_PROCESSED_DIR="/mmlu_openai_v2_4option/val_data_mmlu_pro"
mkdir -p ${OUTPUT_PROCESSED_DIR}
MODEL=1
python -u llm_branching_4option_eval.py \
--dataset=1 \
--input_file="${OUTPUT_DIR}/mmlu_pro_validation_depth_0_shard0.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}

OUTPUT_DIR=/mmlu_openai_v2/train_data_mmlu_pro
# here we fix the dir naming, splitting validation from the eval:
OUTPUT_PROCESSED_DIR="/mmlu_openai_v2_4option/eval_data_mmlu_pro"
mkdir ${OUTPUT_PROCESSED_DIR}
MODEL=1
NUM_SHARDS=50
for ((i=0;i<NUM_SHARDS;i++)); do
echo "shard${i}.jsonl"
python -u llm_branching_4option_eval.py \
--dataset=1 \
--input_file="${OUTPUT_DIR}/mmlu_pro_test_depth_0_shard${i}.jsonl" \
--model=${MODEL} \
--preprocess_and_save \
--output_dir=${OUTPUT_PROCESSED_DIR}
done



#########################################################################################################
##################### Construct combined data splits
#########################################################################################################

OUTPUT_PROCESSED_COMBINED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/processed_combined"  # this is the directory included in the data archive
mkdir ${OUTPUT_PROCESSED_COMBINED_DIR}

# First, we combine MMLU val and dev and MMLU-Pro val:

cat /Users/a/Documents/projects/data/mmlu_openai_v2_4option/val_data_mmlu/model_1_mmlu_dev_depth_0.jsonl /Users/a/Documents/projects/data/mmlu_openai_v2_4option/val_data_mmlu/model_1_mmlu_validation_depth_0.jsonl /Users/a/Documents/projects/data/mmlu_openai_v2_4option/val_data_mmlu_pro/model_1_mmlu_pro_validation_depth_0_shard0.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/combined_mmlu_devval_mmlu_pro_val.jsonl

cat /Users/a/Documents/projects/data/mmlu_openai_v2_4option/val_data_mmlu/model_1_mmlu_dev_depth_0.jsonl /Users/a/Documents/projects/data/mmlu_openai_v2_4option/val_data_mmlu/model_1_mmlu_validation_depth_0.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/combined_mmlu_devval.jsonl

# Next, we combine MMLU train into one file:
OUTPUT_PROCESSED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/train_data_mmlu"
cat ${OUTPUT_PROCESSED_DIR}/model_1_mmlu_auxiliary_train_depth_0_shard*.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/mmlu_train.jsonl

# Next, we combine the final eval MMLU files
OUTPUT_PROCESSED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/eval_data_mmlu"
cat ${OUTPUT_PROCESSED_DIR}/model_1_mmlu_eval*.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/mmlu_test.jsonl

# Next, we combine the final eval MMLU-pro files
OUTPUT_PROCESSED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/eval_data_mmlu_pro"
cat ${OUTPUT_PROCESSED_DIR}/model_1_mmlu_pro_test*.jsonl > ${OUTPUT_PROCESSED_COMBINED_DIR}/mmlu_pro_test.jsonl



#########################################################################################################
##################### Eval reference
#########################################################################################################

cd research_code/code  # Update with the applicable path

OUTPUT_PROCESSED_COMBINED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/processed_combined"
for ALPHA_PRIME in 0.95 0.9; do
python -u llm_branching_4option_combined_eval.py \
--alpha_prime=${ALPHA_PRIME} \
--input_file ${OUTPUT_PROCESSED_COMBINED_DIR}/combined_mmlu_devval_mmlu_pro_val.jsonl \
--input_file_train ${OUTPUT_PROCESSED_COMBINED_DIR}/mmlu_train.jsonl
done

    #---Evaluating combined_mmlu_devval_mmlu_pro_val.jsonl with alpha'=0.95---
    #Mean accuracy: 0.8216802168021681 out of 1845
    #Mean accuracy (among filtered by verbalized uncertainty): 0.9272151898734177 out of 632
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9094256259204713 out of 1358
    #Total LLM refusals: 18
    #---Evaluating mmlu_train.jsonl with alpha'=0.95---
    #Mean accuracy: 0.9210051881973518 out of 99842
    #Mean accuracy (among filtered by verbalized uncertainty): 0.9828005808127288 out of 48897
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9589901289921243 out of 85199
    #Total LLM refusals: 588
    #---Evaluating combined_mmlu_devval_mmlu_pro_val.jsonl with alpha'=0.9---
    #Mean accuracy: 0.8216802168021681 out of 1845
    #Mean accuracy (among filtered by verbalized uncertainty): 0.8517632994620442 out of 1673
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9027777777777778 out of 1440
    #Total LLM refusals: 18
    #---Evaluating mmlu_train.jsonl with alpha'=0.9---
    #Mean accuracy: 0.9210051881973518 out of 99842
    #Mean accuracy (among filtered by verbalized uncertainty): 0.9345225124261951 out of 96369
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9530960013497554 out of 88905
    #Total LLM refusals: 588



OUTPUT_PROCESSED_COMBINED_DIR="/Users/a/Documents/projects/data/mmlu_openai_v2_4option/processed_combined"
for FILENAME in "mmlu_pro_test.jsonl" "mmlu_test.jsonl"; do
for ALPHA_PRIME in 0.95 0.9; do
python -u llm_branching_4option_combined_eval.py \
--alpha_prime=${ALPHA_PRIME} \
--input_file ${OUTPUT_PROCESSED_COMBINED_DIR}/${FILENAME}
done
done

    #---Evaluating mmlu_pro_test.jsonl with alpha'=0.95---
    #Mean accuracy: 0.6480694624053205 out of 5413
    #Mean accuracy (among filtered by verbalized uncertainty): 0.8568207440811725 out of 887
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.8704235463029433 out of 2786
    #Total LLM refusals: 36
    #---Evaluating mmlu_pro_test.jsonl with alpha'=0.9---
    #Mean accuracy: 0.6480694624053205 out of 5413
    #Mean accuracy (among filtered by verbalized uncertainty): 0.6897466827503016 out of 4145
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.8476536682088566 out of 3026
    #Total LLM refusals: 36
    #---Evaluating mmlu_test.jsonl with alpha'=0.95---
    #Mean accuracy: 0.8323600626691354 out of 14042
    #Mean accuracy (among filtered by verbalized uncertainty): 0.953454992823457 out of 4877
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9208757441905128 out of 10414
    #Total LLM refusals: 146
    #---Evaluating mmlu_test.jsonl with alpha'=0.9---
    #Mean accuracy: 0.8323600626691354 out of 14042
    #Mean accuracy (among filtered by verbalized uncertainty): 0.8651302763918587 out of 12627
    #Mean accuracy (among filtered by the token probability of the answer choice): 0.9092295273118666 out of 11149
    #Total LLM refusals: 146


#########################################################################################################
##################### Analysis
##### We avoid including the raw text of the QA responses to avoid contaminating the test sets, but
##### it is easy to pull up the questions from the Huggingface datasets database either from the
##### original id's or the row index (as below).
#########################################################################################################

dataset = load_dataset("TIGER-Lab/MMLU-Pro")
split_name = "test"
# example:
id_to_find = "computer science_10540"
for row_index in range(len(dataset[split_name])):
    document_id = dataset[split_name][row_index]["category"] + f"_{row_index}"
    if document_id == id_to_find:
        print(dataset[split_name][row_index])

 
 
