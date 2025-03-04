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
import uuid

from llm_branching import DatasetFormatCategory
from llm_branching import datasetFormatCategories


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def format_final_answer_mod_as_soft_one_hot(final_answer_mod, confidence_in_answer_letter, total_choices=4):
    predicted_answer_tensor = torch.zeros(total_choices)
    predicted_answer_tensor[final_answer_mod] = confidence_in_answer_letter
    return predicted_answer_tensor


def format_final_answer_letter(final_answer_letter_str):
    predicted_letter = final_answer_letter_str.lower()
    if len(predicted_letter) != 1:
        assert len(predicted_letter) > 0, final_answer_letter_str
        # assert predicted_letter[0] == "$", final_answer_letter_str
        # there may also be a trailing $
        predicted_letter = predicted_letter.replace("$", "")
        # In rare cases with the weaker models, the text may not be a valid letter
        # (e.g., mistakenly replacing with numbers or
        # roman numerals), but for the purposes here,
        # we do not perform further re-formatting, treating those as wrong predictions.
        if len(predicted_letter) != 1:
            print(f"WARNING: Unexpected token encountered: `{final_answer_letter_str}`. "
                  f"This will be treated as a wrong prediction.")
    return predicted_letter


def eval_lines(options):
    datasetFormatCategory = options.dataset
    filepath_with_name = options.input_file
    alpha_prime = options.alpha_prime
    if datasetFormatCategory == datasetFormatCategories.mmlu:
        formatted_answer_choices = [x.lower() for x in ["A", "B", "C", "D"]]
    elif datasetFormatCategory == datasetFormatCategories.mmlu_pro:
        formatted_answer_choices = [x.lower() for x in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]
    accuracy = []
    accuracy_verbalized_uncertainty_filtered = []
    accuracy_logit_filtered = []
    count_full_llm_refusal = 0
    option_choice_ignored = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            label = json_obj["label"].lower()
            assert label in formatted_answer_choices, json_obj
            true_label_int = formatted_answer_choices.index(label)
            if true_label_int > (options.max_considered_options-1):
                option_choice_ignored += 1
                continue
            if len(json_obj["embedding"]) == 0:
                assert len(json_obj["attributes"]) == 0
                count_full_llm_refusal += 1
                accuracy.append(0)
                continue

            predicted_letter = format_final_answer_letter(json_obj["answer_letter"])
            try:
                predicted_letter_int = formatted_answer_choices.index(predicted_letter)
            except:
                print(f"WARNING: predicted letter has unexpected formatting. Counting as incorrect")
                predicted_letter_int = -1
            accuracy.append(true_label_int == predicted_letter_int)
            if float(json_obj["confidence_in_answer_letter"]) >= alpha_prime:
                accuracy_verbalized_uncertainty_filtered.append(true_label_int == predicted_letter_int)
            # in the current convention, the first attribute is the probability of the answer choice:
            if float(json_obj["attributes"][0]) >= alpha_prime:
                accuracy_logit_filtered.append(true_label_int == predicted_letter_int)

    print(f"Mean accuracy: {np.mean(accuracy)} out of {len(accuracy)}")
    print(f"Mean accuracy (among filtered by verbalized uncertainty): "
          f"{np.mean(accuracy_verbalized_uncertainty_filtered) if len(accuracy_verbalized_uncertainty_filtered) > 0 else 0} "
          f"out of {len(accuracy_verbalized_uncertainty_filtered)}")
    print(f"Mean accuracy (among filtered by the token probability of the answer choice): "
          f"{np.mean(accuracy_logit_filtered) if len(accuracy_logit_filtered) > 0 else 0} "
          f"out of {len(accuracy_logit_filtered)}")
    print(f"Total LLM refusals: {count_full_llm_refusal}")
    print(f"Total skipped documents with more than 4 options: {option_choice_ignored}")


def preprocess_data(options):
    datasetFormatCategory = options.dataset
    filepath_with_name = options.input_file
    if datasetFormatCategory == datasetFormatCategories.mmlu:
        formatted_answer_choices = [x.lower() for x in ["A", "B", "C", "D"]]
    elif datasetFormatCategory == datasetFormatCategories.mmlu_pro:
        formatted_answer_choices = [x.lower() for x in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]
    json_lines = []
    count_null_instances = 0
    count_null_predictions = 0
    option_choice_ignored = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            label = json_obj["label"].lower()
            assert label in formatted_answer_choices
            new_json_obj = {}
            new_json_obj["original_label"] = formatted_answer_choices.index(label)
            true_label_int = formatted_answer_choices.index(label)
            new_json_obj["label"] = true_label_int
            if true_label_int > (options.max_considered_options-1):
                option_choice_ignored += 1
                continue

            # we also add a random suffix to be safe since our code assumes unique id's per instance
            new_json_obj["id"] = json_obj["id"] + "_" + str(uuid.uuid4())

            new_json_obj["document"] = json_obj["reply_text"]
            new_json_obj["refusal"] = False
            if len(json_obj["attributes"]) == 0:
                # These are full refusals from the LLM api, which we always treat as wrong predictions.
                count_null_instances += 1
                new_json_obj["original_prediction"] = -1
                new_json_obj["attributes"] = \
                    torch.zeros(options.expected_output_attribute_size).detach().numpy().tolist()
                new_json_obj["embedding"] = \
                    torch.zeros(options.expected_output_embedding_size).detach().numpy().tolist()
                new_json_obj["refusal"] = True
                new_json_obj["confidence_in_answer_letter"] = 0.0
                new_json_obj["answer_letter_logit"] = 0.0
            else:
                predicted_letter = format_final_answer_letter(json_obj["answer_letter"])
                new_json_obj["confidence_in_answer_letter"] = json_obj["confidence_in_answer_letter"]
                new_json_obj["answer_letter_logit"] = json_obj["attributes"][0]
                try:
                    assert predicted_letter in formatted_answer_choices
                    new_json_obj["original_prediction"] = formatted_answer_choices.index(predicted_letter)
                    predicted_letter_int = formatted_answer_choices.index(predicted_letter)
                    predicted_answer_tensor = \
                        format_final_answer_mod_as_soft_one_hot(predicted_letter_int,
                                                                float(json_obj["confidence_in_answer_letter"]),
                                                                total_choices=4)
                except:
                    print(f"WARNING: predicted letter has unexpected formatting. Setting to null prediction")
                    new_json_obj["original_prediction"] = -1
                    predicted_answer_tensor = torch.zeros(4)
                    count_null_predictions += 1

                new_json_obj["embedding"] = json_obj["embedding"]
                # soft (based on structured verbalized uncertainty) one-hot-indicator of mod letter +
                # attributes
                new_json_obj["attributes"] = predicted_answer_tensor.detach().numpy().tolist() + \
                                             json_obj["attributes"]
            assert len(new_json_obj["attributes"]) == options.expected_output_attribute_size
            assert len(new_json_obj["embedding"]) == options.expected_output_embedding_size
            json_lines.append(new_json_obj)
    print(f"Total null instances: {count_null_instances}")
    print(f"Total null predictions: {count_null_predictions}")
    print(f"Total skipped documents with more than 4 options: {option_choice_ignored}")
    return json_lines


def main():
    parser = argparse.ArgumentParser(description="-----[Archive data]-----")
    parser.add_argument("--input_file", default="",
                        help="")
    parser.add_argument("--alpha_prime", default=0.95, type=float,
                        help="")
    parser.add_argument("--depth", default=0, type=int,
                        help="max 2")
    parser.add_argument("--max_considered_options", default=4, type=int,
                        help="")
    parser.add_argument("--model", default=0, type=int,
                        help="0: mini; 1: gpt-4o")
    parser.add_argument("--expected_output_attribute_size", default=7, type=int,
                        help="")
    parser.add_argument("--expected_output_embedding_size", default=3072, type=int,
                        help="")
    parser.add_argument("--dataset", default=0, type=int,
                        help="0: MMLU; 1: MMLU-pro")
    parser.add_argument("--preprocess_and_save", default=False, action='store_true', help="preprocess_and_save")
    parser.add_argument("--output_dir", default="",
                        help="")

    options = parser.parse_args()
    assert options.depth in [0, 1, 2]
    assert options.model in [0, 1]
    assert options.dataset in [0, 1]
    print(f"{DatasetFormatCategory._fields[options.dataset]}")
    eval_lines(options)
    if options.preprocess_and_save:
        json_lines = preprocess_data(options)
        output_file = os.path.join(options.output_dir, f"model_{options.model}_{Path(options.input_file).name}")
        assert output_file.strip() != options.input_file
        save_json_lines(output_file, json_lines)


if __name__ == "__main__":
    main()

