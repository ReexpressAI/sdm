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
import random

from llm_branching import DatasetFormatCategory
from llm_branching import datasetFormatCategories


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def eval_lines(options, filepath_with_name):
    alpha_prime = options.alpha_prime
    print(f"---Evaluating {Path(filepath_with_name).name} with alpha'={alpha_prime}---")
    accuracy = []
    accuracy_verbalized_uncertainty_filtered = []
    accuracy_logit_filtered = []
    count_full_llm_refusal = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            label = json_obj["original_label"]
            if json_obj["refusal"]:
                count_full_llm_refusal += 1
                accuracy.append(0)
                continue
            accuracy.append(label == json_obj["original_prediction"])
            confidence_in_answer_letter = json_obj["confidence_in_answer_letter"]

            if confidence_in_answer_letter >= alpha_prime:
                accuracy_verbalized_uncertainty_filtered.append(label == json_obj["original_prediction"])
            answer_letter_logit = json_obj["answer_letter_logit"]
            if answer_letter_logit >= alpha_prime:
                accuracy_logit_filtered.append(label == json_obj["original_prediction"])
    print(f"Mean accuracy: {np.mean(accuracy)} out of {len(accuracy)}")
    print(f"Mean accuracy (among filtered by verbalized uncertainty): "
          f"{np.mean(accuracy_verbalized_uncertainty_filtered) if len(accuracy_verbalized_uncertainty_filtered) > 0 else 0} "
          f"out of {len(accuracy_verbalized_uncertainty_filtered)}")
    print(f"Mean accuracy (among filtered by the token probability of the answer choice): "
          f"{np.mean(accuracy_logit_filtered) if len(accuracy_logit_filtered) > 0 else 0} "
          f"out of {len(accuracy_logit_filtered)}")
    print(f"Total LLM refusals: {count_full_llm_refusal}")


def main():
    parser = argparse.ArgumentParser(description="-----[Archive data]-----")
    parser.add_argument("--input_file", default="",
                        help="")
    parser.add_argument("--input_file_train", default="",
                        help="")
    parser.add_argument("--alpha_prime", default=0.95, type=float,
                        help="")

    options = parser.parse_args()
    random.seed(0)
    eval_lines(options, options.input_file)
    if options.input_file_train.strip() != "":
        eval_lines(options, options.input_file_train)


if __name__ == "__main__":
    main()

