# Copyright Reexpress AI, Inc. All rights reserved.

"""

"""

import argparse
import codecs
import json
import random
import numpy as np
import torch


def print_summary(header_label, list_to_process, total=None):
    if total is not None and total > 0:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)} "
            f"\t({len(list_to_process)/total})% of {total}")
    else:
        print(
            f"{header_label} \tmean: {np.mean(list_to_process) if len(list_to_process) > 0 else 0}, "
            f"\tout of {len(list_to_process)}")


def get_metadata_lines(filepath_with_name, class_size=2):
    correct_task_predictions = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            correct_task_prediction = json_obj["correct_task_prediction"]
            correct_task_predictions.append(correct_task_prediction)
    print_summary(f"Marginal accuracy, across underlying tasks (i.e., not verification):",
                  correct_task_predictions,
                  total=None)


def main():
    parser = argparse.ArgumentParser(
        description="-----[Analysis of generation files.]-----")
    parser.add_argument(
        "--input_filename", required=True,
        help="")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    random.seed(options.seed_value)
    rng = np.random.default_rng(seed=options.seed_value)
    class_size = 2
    get_metadata_lines(options.input_filename, class_size=class_size)


if __name__ == "__main__":
    main()