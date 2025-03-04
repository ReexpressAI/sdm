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
from datasets import load_dataset

DatasetFormatCategory = namedtuple("DatasetFormatCategory",
                          ["mmlu", "mmlu_pro"])
datasetFormatCategories = DatasetFormatCategory(0, 1)


class MultipleChoiceQuestionResponse(BaseModel):
    answer_letter: str
    confidence_in_answer_letter: float
    short_explanation_for_answer_confidence: str


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_candidate_errors(filepath_with_name):
    uuid2idx = {}
    json_lines = []
    line_idx = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            uuid = json_obj["id"]
            # here, we re-process, removing the suffix uuid
            first_marker_index = uuid.find("_")
            assert first_marker_index != -1
            second_marker_index = uuid[first_marker_index+1:].find("_")
            assert second_marker_index != -1
            original_id = uuid[0:first_marker_index + second_marker_index + 1]
            assert original_id not in uuid2idx
            uuid2idx[original_id] = line_idx
            json_lines.append(json_obj)
            line_idx += 1
    return json_lines, uuid2idx


def format_mmlu_pro_hf_datasets(json_dict, verbose=False):
    question = json_dict["question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    choice_i = 0
    formatted_true_answer = ""
    assert len(json_dict["options"]) <= len(formatted_answer_choices), \
        f'{len(json_dict["options"])}, {len(formatted_answer_choices)}, {json_dict}'
    if verbose and len(json_dict["options"]) < len(formatted_answer_choices):
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
    return "".join(formatted_choice_texts), formatted_true_answer, json_dict["question_id"], json_dict["category"]


def preprocess_mmlu_pro(options, json_lines, uuid2idx):
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    unsorted_candidate_errors = []
    for split_name in ["test"]:  # here, only test ["validation", "test"]
        for row_index in range(len(dataset[split_name])):
            formatted_choice_texts, answer, question_id, category = \
                format_mmlu_pro_hf_datasets(dataset[split_name][row_index])
            document_id = dataset[split_name][row_index]["category"] + f"_{row_index}"
            if document_id in uuid2idx:
                unsorted_candidate_errors.append(
                    (json_lines[uuid2idx[document_id]]["prediction_probability__lower"],
                     formatted_choice_texts,
                     answer,
                     json_lines[uuid2idx[document_id]],
                     document_id,
                     question_id,
                     category)
                )

    unsorted_candidate_errors = [y for y in
                                 sorted(unsorted_candidate_errors, key=lambda x: x[0],
                                        reverse=True)]
    assert len(json_lines) == len(unsorted_candidate_errors)
    for candidate_error in unsorted_candidate_errors:
        document_id = candidate_error[4]  # internally used
        question_id = candidate_error[5]  # the original id's
        category = candidate_error[6]
        if options.category_restriction != "":
            if options.category_restriction not in document_id:
                continue
        print(f"--------: START: question_id: {question_id}, category: {category} :: "
              f"index row identifier: {document_id}")
        print(f"{candidate_error[1]}")
        print(f"GROUND-TRUTH-ANSWER: {candidate_error[2]}")
        print(f"SDM: {candidate_error[3]}")
        print(f"--------: END")


def main():
    parser = argparse.ArgumentParser(description="-----[Archive data]-----")
    parser.add_argument("--input_candidate_label_annotation_error_file", default="",
                        help="")
    parser.add_argument("--dataset", default="mmlu_pro",
                        help="currently only mmlu_pro")
    parser.add_argument("--category_restriction", default="computer science",
                        help="")
    options = parser.parse_args()

    json_lines, uuid2idx = get_candidate_errors(options.input_candidate_label_annotation_error_file)

    assert options.dataset == "mmlu_pro"
    preprocess_mmlu_pro(options, json_lines, uuid2idx)


if __name__ == "__main__":
    main()

