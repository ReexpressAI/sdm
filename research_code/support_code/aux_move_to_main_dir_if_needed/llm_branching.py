# Copyright Reexpress AI, Inc. All rights reserved.

# -*- coding: utf-8 -*-

import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs
import pandas
import torch
from collections import namedtuple

from pydantic import BaseModel
from openai import AzureOpenAI
from datasets import load_dataset

DatasetFormatCategory = namedtuple("DatasetFormatCategory",
                          ["mmlu", "mmlu_pro"])
datasetFormatCategories = DatasetFormatCategory(0, 1)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
client_embedding = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


class MultipleChoiceQuestionResponse(BaseModel):
    answer_letter: str
    confidence_in_answer_letter: float
    short_explanation_for_answer_confidence: str

# The JSON schema, if needed:
# print(json.dumps(MultipleChoiceQuestionResponse.model_json_schema(), indent=4))

# The LETTER format is along the lines of https://github.com/openai/simple-evals/blob/main/common.py
QUERY_TEMPLATE_MULTICHOICE = """
Please answer the multiple choice question appearing after the header QUESTION. Please structure your response using the provided JSON format, with the answer letter formatted as follows: '$LETTER' (without quotes) where LETTER is one of ABCD. Please provide your answer and a confidence estimate in your answer as a probability between 0 and 1, where a probability of 0 indicates no confidence and a probability of 1 indicates 100% confidence that your estimated answer is correct. Finally, please provide a short explanation for your answer confidence.
"""

QUERY_TEMPLATE_MULTICHOICE_10 = """
Please answer the multiple choice question appearing after the header QUESTION. Please structure your response using the provided JSON format, with the answer letter formatted as follows: '$LETTER' (without quotes) where LETTER is one of ABCDEFGHIJ. Please provide your answer and a confidence estimate in your answer as a probability between 0 and 1, where a probability of 0 indicates no confidence and a probability of 1 indicates 100% confidence that your estimated answer is correct. Finally, please provide a short explanation for your answer confidence.
"""

def get_logit_attributes(logprobs_content):
    field_checks = ["answer_letter\":\"", "confidence_in_answer_letter\":", "short_explanation_for_answer_confidence\":\""]
    next_indicators = ["\",\"confidence", ",\"short"]

    running_string = ""
    field_probabilities = []
    for _ in range(len(field_checks)):
        field_probabilities.append([])

    field_i = 0
    for position_id, completion_token_position_value in enumerate(logprobs_content):
        for top_token_k, top_token in enumerate(completion_token_position_value.top_logprobs):
            if top_token_k == 0:
                token_prob = np.exp(top_token.logprob)
                running_string += top_token.token
                if field_i == len(field_checks) - 1:
                    valid_parse = field_checks[field_i] in running_string
                else:
                    valid_parse = field_checks[field_i] in running_string and field_checks[field_i + 1] not in running_string
                if valid_parse:
                    field_probabilities[field_i].append(token_prob)
                if field_i < len(next_indicators):
                    if next_indicators[field_i] in running_string:
                        field_i += 1
    field_probability_averages = []
    for field_i in range(len(field_checks)):
        if field_i == len(field_checks) - 1:
            field_probability_averages.append(np.mean(field_probabilities[field_i][1:-1]).item())
        else:
            field_probability_averages.append(np.mean(field_probabilities[field_i][1:-2]).item())
    return field_probability_averages


def get_embedding(client_embedding, document_text):
    embedding_response = client_embedding.embeddings.create(
        model="text-embedding-3-large-2",
        input=document_text,
        encoding_format="float",
        user="embed_llm_branching_v1"
    )
    full_embedding = embedding_response.data[0].embedding
    return full_embedding


def get_document_attributes(client, client_embedding, document_id: str, document_string, max_tokens=512, re_ask=False,
                            previous_responses_string="", model="", datasetFormatCategory=0):
    if datasetFormatCategory == datasetFormatCategories.mmlu:
        query_template = QUERY_TEMPLATE_MULTICHOICE.strip()
    elif datasetFormatCategory == datasetFormatCategories.mmlu_pro:
        query_template = QUERY_TEMPLATE_MULTICHOICE_10.strip()
    else:
        assert False
    total_logprobs_to_consider = 1
    if not re_ask:
        messages_structure = [
                {"role": "system", "content": f"You are a helpful assistant that answers multiple choice questions. {query_template}"},
                {"role": "user",
                 "content": f"{document_string}"}
            ]
    else:
        assert False
        # messages_structure = [
        #         {"role": "system", "content": f"You are a helpful assistant that answers multiple choice questions. {QUERY_TEMPLATE_MULTICHOICE.strip()} {QUERY_TEMPLATE_REASK_SUFFIX.strip()}"},
        #         {"role": "user",
        #          "content": f"{document_string} {previous_responses_string}"}
        #     ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages_structure,
        response_format=MultipleChoiceQuestionResponse,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=total_logprobs_to_consider,
        temperature=0.0,
        user="llm_branching_v1",
        seed=0
    )
    qa_object = completion.choices[0].message.parsed
    reply_text = completion.choices[0].message.content  # this is the JSON as a string
    embedding_input_string = f"PROMPT: {query_template} {document_string} RESPONSE: {reply_text}"
    embedding = get_embedding(client_embedding, embedding_input_string)
    attributes = get_logit_attributes(completion.choices[0].logprobs.content)
    json_obj = {"id": document_id,
                "embedding": embedding,
                "attributes": attributes,
                "reply_text": reply_text,
                "answer_letter": qa_object.answer_letter,
                "confidence_in_answer_letter": qa_object.confidence_in_answer_letter,
                "short_explanation_for_answer_confidence":
                    qa_object.short_explanation_for_answer_confidence,
                }
    return json_obj


def build_previous_response_string(previous_responses_list):
    previous_response_accumulator = []
    for depth, reply_text in enumerate(previous_responses_list):
        previous_response_accumulator.append(f"PREVIOUS_RESPONSE_{depth}: {reply_text}")
    return " ".join(previous_response_accumulator)


def llm_api_controller(client, client_embedding, document_id: str, document_string, max_tokens=512, re_ask=False,
                       previous_responses_string="", model="", datasetFormatCategory=0):
    # modeled after https://github.com/openai/simple-evals/blob/main/sampler/chat_completion_sampler.py
    call_schedule = 0
    while True:
        try:
            completion_json = get_document_attributes(client, client_embedding, document_id, document_string,
                                                      max_tokens=max_tokens, re_ask=re_ask,
                                                      previous_responses_string=previous_responses_string,
                                                      model=model,
                                                      datasetFormatCategory=datasetFormatCategory)
            return completion_json
        except:
            if call_schedule == 4:
                print(
                    f"Get document attributes failed for {document_id} with string {document_string}."
                    f"Returning error structure"
                )
                json_obj = {"id": document_id,
                            "embedding": [],
                            "attributes": [],
                            "reply_text": "",
                            "answer_letter": "X",
                            "confidence_in_answer_letter": 0.0,
                            "short_explanation_for_answer_confidence": "I am unable to answer that question.",
                            }
                return json_obj
            exception_backoff = 2 ** call_schedule + torch.abs(torch.randn(1)).item()
            print(
                f"Get document attributes failed for {document_id} with string {document_string}. "
                f"Retrying {call_schedule} after {exception_backoff} seconds."
            )
            time.sleep(exception_backoff)
            call_schedule += 1


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj["id"])
    return existing_ids


def process_mmlu(options):
    df = pandas.read_csv(options.input_file)
    examples = [row.to_dict() for _, row in df.iterrows()]

    max_depth = options.max_depth
    shard = options.shard
    total_shards = options.total_shards
    accuracy_by_depth = {}
    existing_ids_by_depth = {}
    for depth in range(max_depth):
        accuracy_by_depth[depth] = []
        output_file = os.path.join(options.output_dir, f"mmlu_eval_depth_{depth}_shard{shard}.jsonl")
        if Path(output_file).exists():
            existing_ids_by_depth[depth] = get_existing_ids(output_file)
            print(f"{len(existing_ids_by_depth[depth])} existing ids for depth {depth}")
        else:
            existing_ids_by_depth[depth] = set()
    total_size = len(examples)
    row_indexes = np.arange(total_size)
    split_row_indexes = [int(x) for x in np.array_split(row_indexes, total_shards)[shard].tolist()]
    for row_index in split_row_indexes:
        if row_index % 100 == 0:
            print(f"Currently processing row {row_index} of {len(examples)}")
        previous_responses_list = []
        formatted_choice_texts, answer, document_id = format_mmlu(examples[row_index])
        if document_id in existing_ids_by_depth[0]:
            continue
        for depth in range(max_depth):
            if depth > 0:
                prev_response = build_previous_response_string(previous_responses_list)
                completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                     max_tokens=512, re_ask=True,
                                                     previous_responses_string=prev_response, model=options.model,
                                                     datasetFormatCategory=datasetFormatCategories.mmlu)
            else:
                completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                     max_tokens=512, re_ask=False, model=options.model,
                                                     datasetFormatCategory=datasetFormatCategories.mmlu)
            # add true answer for reference
            completion_json["label"] = answer
            output_file = os.path.join(options.output_dir, f"mmlu_eval_depth_{depth}_shard{shard}.jsonl")
            save_by_appending_json_lines(output_file, [completion_json])
            accuracy_by_depth[depth].append(
                completion_json["label"].lower() == completion_json["answer_letter"].lower())

            previous_responses_list.append(completion_json["short_explanation_for_answer_confidence"])
            if row_index % 100 == 0:
                print(
                    f"Running accuracy depth {depth}: {np.mean(accuracy_by_depth[depth])} out of {len(accuracy_by_depth[depth])}")


def format_mmlu(json_dict):
    question = json_dict["Question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D"]
    choice_i = 0
    formatted_true_answer = ""
    for label in formatted_answer_choices:
        choice_text = json_dict[label]
        formatted_choice_texts.append(f"{label}) {choice_text}\n")
        if label == json_dict["Answer"]:
            formatted_true_answer = label
        choice_i += 1
    assert formatted_true_answer != ""
    assert choice_i == len(formatted_answer_choices)
    document_id = f'{json_dict["Subject"]}_{json_dict["Unnamed: 0"]}'
    return "".join(formatted_choice_texts), formatted_true_answer, document_id


def format_mmlu_hf_datasets(json_dict):
    question = json_dict["question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D"]
    choice_i = 0
    formatted_true_answer = ""
    assert len(json_dict["choices"]) == len(formatted_answer_choices), json_dict["choices"]
    assert json_dict["answer"] in range(len(formatted_answer_choices))
    for formatted_label, choice_text in zip(formatted_answer_choices, json_dict["choices"]):
        formatted_choice_texts.append(f"{formatted_label}) {choice_text}\n")
        if choice_i == json_dict["answer"]:
            formatted_true_answer = formatted_label
        choice_i += 1
    assert formatted_true_answer != ""
    return "".join(formatted_choice_texts), formatted_true_answer


def process_mmlu_val(options):
    dataset = load_dataset("cais/mmlu", "all")
    for split_name in ["dev", "validation"]:
        print(f"Currently processing mmlu: {split_name}")
        max_depth = options.max_depth

        accuracy_by_depth = {}
        existing_ids_by_depth = {}
        for depth in range(max_depth):
            accuracy_by_depth[depth] = []
            output_file = os.path.join(options.output_dir, f"mmlu_{split_name}_depth_{depth}.jsonl")
            if Path(output_file).exists():
                existing_ids_by_depth[depth] = get_existing_ids(output_file)
                print(f"{len(existing_ids_by_depth[depth])} existing ids for depth {depth}")
            else:
                existing_ids_by_depth[depth] = set()

        for row_index in range(len(dataset[split_name])):
            if row_index % 100 == 0:
                print(f"Currently processing row {row_index} of {len(dataset[split_name])}")
            previous_responses_list = []
            formatted_choice_texts, answer = format_mmlu_hf_datasets(dataset[split_name][row_index])
            document_id = dataset[split_name][row_index]["subject"] + f"_{row_index}"

            if document_id in existing_ids_by_depth[0]:
                continue
            for depth in range(max_depth):
                if depth > 0:
                    prev_response = build_previous_response_string(previous_responses_list)
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=True,
                                                         previous_responses_string=prev_response, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu)
                else:
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=False, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu)
                # add true answer for reference
                completion_json["label"] = answer
                output_file = os.path.join(options.output_dir, f"mmlu_{split_name}_depth_{depth}.jsonl")
                save_by_appending_json_lines(output_file, [completion_json])
                accuracy_by_depth[depth].append(completion_json["label"].lower() == completion_json["answer_letter"].lower())

                previous_responses_list.append(completion_json["short_explanation_for_answer_confidence"])
                if row_index % 100 == 0:
                    print(f"Running accuracy depth {depth}: {np.mean(accuracy_by_depth[depth])} out of {len(accuracy_by_depth[depth])}")


def preprocess_mmlu_aux_train(options):
    dataset = load_dataset("cais/mmlu", "all")
    for split_name in ["auxiliary_train"]:
        print(f"Currently processing mmlu: {split_name}")
        max_depth = options.max_depth
        shard = options.shard
        total_shards = options.total_shards
        accuracy_by_depth = {}
        existing_ids_by_depth = {}
        for depth in range(max_depth):
            accuracy_by_depth[depth] = []
            output_file = os.path.join(options.output_dir, f"mmlu_{split_name}_depth_{depth}_shard{shard}.jsonl")
            if Path(output_file).exists():
                existing_ids_by_depth[depth] = get_existing_ids(output_file)
                print(f"{len(existing_ids_by_depth[depth])} existing ids for depth {depth}")
            else:
                existing_ids_by_depth[depth] = set()
        total_size = len(dataset[split_name])
        row_indexes = np.arange(total_size)
        split_row_indexes = [int(x) for x in np.array_split(row_indexes, total_shards)[shard].tolist()]
        for row_index in split_row_indexes:
            if row_index % 100 == 0:
                print(f"Currently processing row {row_index} of {len(dataset[split_name])}")
            previous_responses_list = []
            formatted_choice_texts, answer = format_mmlu_hf_datasets(dataset[split_name][row_index])
            document_id = dataset[split_name][row_index]["subject"] + f"_{row_index}"
            if document_id in existing_ids_by_depth[0]:
                continue
            for depth in range(max_depth):
                if depth > 0:
                    prev_response = build_previous_response_string(previous_responses_list)
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=True,
                                                         previous_responses_string=prev_response, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu)
                else:
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=False, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu)
                # add true answer for reference
                completion_json["label"] = answer
                output_file = os.path.join(options.output_dir, f"mmlu_{split_name}_depth_{depth}_shard{shard}.jsonl")
                save_by_appending_json_lines(output_file, [completion_json])
                accuracy_by_depth[depth].append(completion_json["label"].lower() == completion_json["answer_letter"].lower())

                previous_responses_list.append(completion_json["short_explanation_for_answer_confidence"])
                if row_index % 100 == 0:
                    print(f"Running accuracy depth {depth}: {np.mean(accuracy_by_depth[depth])} out of {len(accuracy_by_depth[depth])}")


def format_mmlu_pro_hf_datasets(json_dict):
    question = json_dict["question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    choice_i = 0
    formatted_true_answer = ""
    assert len(json_dict["options"]) <= len(formatted_answer_choices), \
        f'{len(json_dict["options"])}, {len(formatted_answer_choices)}, {json_dict}'
    if len(json_dict["options"]) < len(formatted_answer_choices):
        print(f"WARNING: option set only contains {len(json_dict['options'])} items: {json_dict}")
    assert json_dict["answer_index"] in range(len(formatted_answer_choices))
    for formatted_label, choice_text in zip(formatted_answer_choices, json_dict["options"]):
        formatted_choice_texts.append(f"{formatted_label}) {choice_text}\n")
        if choice_i == json_dict["answer_index"]:
            formatted_true_answer = formatted_label
            if formatted_true_answer != json_dict["answer"]:
                print(f"WARNING: The index field appears to be mismatched. Reverting to the answer letter. {json_dict}")
                formatted_true_answer = json_dict["answer"]
        choice_i += 1
    assert formatted_true_answer != ""
    return "".join(formatted_choice_texts), formatted_true_answer


def preprocess_mmlu_pro(options):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    for split_name in ["validation", "test"]:
        print(f"Currently processing mmlu: {split_name}")
        max_depth = options.max_depth
        if split_name == "validation":
            if options.shard != 0:
                continue
            shard = options.shard
            total_shards = 1  # validation only has 70 rows
        else:
            shard = options.shard
            total_shards = options.total_shards
        accuracy_by_depth = {}
        existing_ids_by_depth = {}
        for depth in range(max_depth):
            accuracy_by_depth[depth] = []
            output_file = os.path.join(options.output_dir, f"mmlu_pro_{split_name}_depth_{depth}_shard{shard}.jsonl")
            if Path(output_file).exists():
                existing_ids_by_depth[depth] = get_existing_ids(output_file)
                print(f"{len(existing_ids_by_depth[depth])} existing ids for depth {depth}")
            else:
                existing_ids_by_depth[depth] = set()
        total_size = len(dataset[split_name])
        row_indexes = np.arange(total_size)
        split_row_indexes = [int(x) for x in np.array_split(row_indexes, total_shards)[shard].tolist()]
        for row_index in split_row_indexes:
            if row_index % 100 == 0:
                print(f"Currently processing row {row_index} of {len(dataset[split_name])}")
            previous_responses_list = []
            formatted_choice_texts, answer = format_mmlu_pro_hf_datasets(dataset[split_name][row_index])
            document_id = dataset[split_name][row_index]["category"] + f"_{row_index}"
            if document_id in existing_ids_by_depth[0]:
                continue
            for depth in range(max_depth):
                if depth > 0:
                    prev_response = build_previous_response_string(previous_responses_list)
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=True,
                                                         previous_responses_string=prev_response, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu_pro)
                else:
                    completion_json = llm_api_controller(client, client_embedding, document_id, formatted_choice_texts,
                                                         max_tokens=512, re_ask=False, model=options.model,
                                                         datasetFormatCategory=datasetFormatCategories.mmlu_pro)
                # add true answer for reference
                completion_json["label"] = answer
                output_file = os.path.join(options.output_dir, f"mmlu_pro_{split_name}_depth_{depth}_shard{shard}.jsonl")
                save_by_appending_json_lines(output_file, [completion_json])
                accuracy_by_depth[depth].append(completion_json["label"].lower() == completion_json["answer_letter"].lower())

                previous_responses_list.append(completion_json["short_explanation_for_answer_confidence"])
                if row_index % 100 == 0:
                    print(f"Running accuracy depth {depth}: {np.mean(accuracy_by_depth[depth])} out of {len(accuracy_by_depth[depth])}")


def main():
    parser = argparse.ArgumentParser(description="-----[Archive data]-----")
    parser.add_argument("--model", default="gpt-4o-2024-08-06",
                        help="Currently only 'gpt-4o-2024-08-06' or 'gpt-4o-mini'")
    parser.add_argument("--dataset", default="",
                        help="")
    parser.add_argument("--input_file", default="",
                        help="if applicable")
    parser.add_argument("--max_depth", default=1, type=int,
                        help="")
    parser.add_argument("--shard", default=0, type=int,
                        help="")
    parser.add_argument("--total_shards", default=50, type=int,
                        help="")
    parser.add_argument("--output_dir", default="",
                        help="output_dir")
    options = parser.parse_args()

    assert options.model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]
    assert options.max_depth == 1
    print(f"Processing with {options.model}")
    print(f"Note that the running accuracy estimate is an under-estimate since this script does not "
          f"currently parse the output. Use the eval script for evaluation.")
    if options.dataset == "mmlu":
        print(f"Processing MMLU for eval.")
        time.sleep(torch.abs(torch.randn(1)).item())
        process_mmlu(options)
        exit()
    elif options.dataset == "mmlu_val":
        print(f"Processing MMLU val/dev sets.")
        time.sleep(torch.abs(torch.randn(1)).item())
        process_mmlu_val(options)
        exit()
    elif options.dataset == "preprocess_mmlu_aux_train":
        time.sleep(torch.abs(torch.randn(1)).item())
        preprocess_mmlu_aux_train(options)
    elif options.dataset == "mmlu_pro":
        time.sleep(torch.abs(torch.randn(1)).item())
        preprocess_mmlu_pro(options)


if __name__ == "__main__":
    main()

