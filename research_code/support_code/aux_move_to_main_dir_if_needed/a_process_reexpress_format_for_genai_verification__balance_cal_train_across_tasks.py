# Copyright Reexpress AI, Inc. All rights reserved.

"""

"""
from utils_gen import taskCategories, llmTypes
import argparse
import codecs
import json
import uuid
from os import path
import time
import urllib
import random
import copy
import numpy as np
import glob
import pickle
from datetime import datetime
import torch
from mlx_lm import load


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def balance_by_labels(class_label2json_obj, class_0_size_match, class_1_size_match):
    class_size = 2
    check = class_label2json_obj[0][0]
    for label in range(class_size):
        random.shuffle(class_label2json_obj[label])
    assert class_label2json_obj[0][0] != check
    print(f"shuffled")
    rebalanced = []
    rebalanced.extend(class_label2json_obj[0][0:class_0_size_match])  # class 0
    rebalanced.extend(class_label2json_obj[1][0:class_1_size_match])  # class 1
    return rebalanced


def get_metadata_lines_by_label(filepath_with_name, class_label2json_obj=None, class_size=2):
    if class_label2json_obj is None:
        class_label2json_obj = {}
        for label in range(class_size):
            class_label2json_obj[label] = []

    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            class_label2json_obj[json_obj["label"]].append(json_obj)
    return class_label2json_obj


def main():
    parser = argparse.ArgumentParser(
        description="-----[Construct output JSON lines formatted files for input to Reexpress.]-----")
    parser.add_argument(
        "--input_task0_filename0", required=True,
        help="")
    parser.add_argument(
        "--input_task0_filename1", required=True,
        help="")
    parser.add_argument(
        "--input_task1_filename0", required=True,
        help="")
    parser.add_argument(
        "--input_task1_filename1", required=True,
        help="")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument(
        "--output_task0_filename", default="",
        help="")
    parser.add_argument(
        "--output_task1_filename", default="",
        help="")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    random.seed(options.seed_value)
    rng = np.random.default_rng(seed=options.seed_value)
    class_size = 2
    print(f"Balancing by 'label' (i.e., verification, rather than the original task label), "
          f"assuming Task 1 is smaller and balanced. Assuming {class_size} classes. Assuming class 0 is balanced.")
    task1_class_label2json_obj = get_metadata_lines_by_label(options.input_task1_filename0)
    task1_class_label2json_obj = get_metadata_lines_by_label(options.input_task1_filename1,
                                                             task1_class_label2json_obj)
    for label in range(class_size):
        print(f"Task 1: count of class {label}: {len(task1_class_label2json_obj[label])}")
    task0_class_label2json_obj = get_metadata_lines_by_label(options.input_task0_filename0)
    task0_class_label2json_obj = get_metadata_lines_by_label(options.input_task0_filename1,
                                                             task0_class_label2json_obj)
    for label in range(class_size):
        print(f"Task 0: count of class {label}: {len(task0_class_label2json_obj[label])}")
    class_0_size_match = len(task1_class_label2json_obj[0])
    class_1_size_match = len(task1_class_label2json_obj[1])
    task0_rebalanced = balance_by_labels(task0_class_label2json_obj, class_0_size_match, class_1_size_match)
    save_json_lines(options.output_task0_filename, task0_rebalanced)
    task1_combined = []
    task1_combined.extend(task1_class_label2json_obj[0])  # class 0
    task1_combined.extend(task1_class_label2json_obj[1])  # class 1
    save_json_lines(options.output_task1_filename, task1_combined)


if __name__ == "__main__":
    main()