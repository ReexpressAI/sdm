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


def get_label_string(label: int, taskCategory: int):
    label_str = ""
    inverse_label_str = ""
    if taskCategory == taskCategories.sentiment:
        if label == 0:
            label_str = '{"sentiment": "negative"}'
            inverse_label_str = '{"sentiment": "positive"}'
        elif label == 1:
            label_str = '{"sentiment": "positive"}'
            inverse_label_str = '{"sentiment": "negative"}'
        else:
            assert False
    elif taskCategory == taskCategories.factcheck:
        if label == 0:
            label_str = '{"correctness": false}'
            inverse_label_str = '{"correctness": true}'
        elif label == 1:
            label_str = '{"correctness": true}'
            inverse_label_str = '{"correctness": false}'
        else:
            assert False
    else:
        assert False
    return label_str, inverse_label_str


def _get_content(construct_negative: bool, label_str: str, inverse_label_str: str,
                       document: str, kSystemPrompt: str, kPrimaryPrompt: str,
                       include_assistant_message: bool = False):
    forced_decoded_response_text = ""
    messages = [
        {"role": "system", "content": f"{kSystemPrompt.strip()}"},
        {"role": "user",
         "content": f"{kPrimaryPrompt.strip()} {document.strip()}"},
    ]
    messages_no_assistant = list(messages)
    if construct_negative:  # given classification is wrong
        updated_label = 0
        if include_assistant_message:
            forced_decoded_response_text = inverse_label_str
            messages.append({"role": "assistant",
                             "content": f"{forced_decoded_response_text}"})
    else:  # given classification is correct
        updated_label = 1
        if include_assistant_message:
            forced_decoded_response_text = label_str
            messages.append({"role": "assistant",
                             "content": f"{forced_decoded_response_text}"})
    return updated_label, messages, forced_decoded_response_text, messages_no_assistant


def get_max_generation_length(class_size, tokenizer, taskCategory):
    max_tokens = 0
    for label in range(class_size):
        label_str, inverse_label_str = get_label_string(label=label, taskCategory=taskCategory)
        for response_str in [label_str, inverse_label_str]:
            tokens = tokenizer.encode(response_str, add_special_tokens=False)
            if len(tokens) > max_tokens:
                max_tokens = len(tokens)
    return max_tokens + 2  # final end symbols


def construct_template(tokenizer, document: str, label: int, taskCategory: int,
                       construct_negative: bool, llmType=None):
    # label is the original task label
    kSystemPrompt = "You are a helpful AI assistant."
    if taskCategory == taskCategories.factcheck:
        kPrimaryPrompt = 'Check the following document for hallucinations and/or factual inaccuracies. Respond using the following JSON: {"correctness": bool}. DOCUMENT:'
    elif taskCategory == taskCategories.sentiment:
        kPrimaryPrompt = 'Classify the sentiment of the following movie review. Respond using the following JSON: {"sentiment": str}. REVIEW:'
    else:
        assert False

    assert taskCategory in [taskCategories.sentiment, taskCategories.factcheck]
    label_str, inverse_label_str = get_label_string(label=label, taskCategory=taskCategory)
    _, messages, _, messages_no_assistant = _get_content(construct_negative=construct_negative,
                                                 label_str=label_str, inverse_label_str=inverse_label_str,
                                                 document=document,
                                                 kSystemPrompt=kSystemPrompt,
                                                 kPrimaryPrompt=kPrimaryPrompt,
                                                 include_assistant_message=True)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_no_assistant = tokenizer.apply_chat_template(
        messages_no_assistant,
        tokenize=False,
        add_generation_prompt=True
    )
    prefix_mask = [1] * len(tokenizer.encode(prompt_no_assistant, add_special_tokens=False))
    if llmType == llmTypes.phiThreePointFive:  # may need to modify final tokens, depending on the gen_ai_model
        assert prompt[-len("<|end|>\n<|assistant|>\n"):] == "<|end|>\n<|assistant|>\n", \
            "ERROR: Unexpected model vocab format."
        # NOTE: This is a quirk of the existing message parsing that we just keep for the purposes here.
        # The main takeaway
        # is that whatever convention is chosen, it's best to keep it the same in training and inference.
        # (And it's generally a good idea to try to stay as close as possible to the convention used when initially
        # training/fine-tuning, if known.)
        # prompt = prompt[:-len("<|end|>\n<|assistant|>\n")].strip()
        # prompt += "<|end|><|endoftext|>"
    elif llmType == llmTypes.phi4:
        # tokenizer.eos_token_ids corresponds to tokenizer.decode([100265]) == '<|im_end|>', so no modification
        # is needed
        assert prompt[-len("<|im_end|>"):] == "<|im_end|>"

    return prompt, prompt_no_assistant, prefix_mask


def get_metadata_lines(filepath_with_name, taskCategory: int, llmType: int, tokenizer=None):
    json_list = []
    total_construct_negative = 0
    max_generation_tokens = get_max_generation_length(2, tokenizer, taskCategory)
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            new_json_obj = {}
            if "info" in json_obj:
                new_json_obj["info"] = json_obj["info"]
            if "group" in json_obj:
                new_json_obj["group"] = json_obj["group"]
            new_json_obj["original_label"] = json_obj["label"]
            new_json_obj["document"] = json_obj["document"]
            new_json_obj["id"] = json_obj["id"]
            construct_negative = bool(torch.randint(2, (1,)).item())
            new_json_obj["construct_negative"] = construct_negative
            if construct_negative:  # given classification is wrong
                new_json_obj["label"] = 0
                total_construct_negative += 1
            else:  # given classification is correct
                new_json_obj["label"] = 1
            new_json_obj["taskCategory"] = taskCategory
            new_json_obj["llmType"] = llmType
            prompt, prompt_no_assistant, prefix_mask = \
                construct_template(tokenizer, document=new_json_obj["document"],
                                   label=new_json_obj["original_label"],
                                   taskCategory=new_json_obj["taskCategory"],
                                   construct_negative=new_json_obj["construct_negative"],
                                   llmType=new_json_obj["llmType"])
            new_json_obj["prompt"] = prompt
            new_json_obj["prompt_no_assistant"] = prompt_no_assistant
            new_json_obj["prefix_mask"] = prefix_mask
            new_json_obj["max_generation_tokens"] = max_generation_tokens
            json_list.append(new_json_obj)
    print(f"Total negative constructions {total_construct_negative} out of {len(json_list)}")
    return json_list


def main():
    parser = argparse.ArgumentParser(
        description="-----[Construct output JSON lines formatted files for input to Reexpress.]-----")

    parser.add_argument(
        "--input_filename", required=True,
        help="")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--taskCategory", default=0, type=int,
                        help="int; 0 for sentiment; 1 for factcheck.")
    parser.add_argument("--llmType", default=0, type=int,
                        help="int; 0 for phi 3.5; 1 for phi4 (not yet tested)")
    parser.add_argument("--gen_ai_model_path", default="",
                        help="")
    parser.add_argument(
        "--output_filename", default="",
        help="")

    options = parser.parse_args()

    # Setting seed
    torch.manual_seed(options.seed_value)
    np.random.seed(options.seed_value)
    # random.seed(options.seed_value)
    # rng = np.random.default_rng(seed=options.seed_value)
    _, tokenizer = load(options.gen_ai_model_path)

    assert options.taskCategory in [taskCategories.sentiment, taskCategories.factcheck]
    assert options.llmType in [llmTypes.phiThreePointFive, llmTypes.phi4]
    json_list = get_metadata_lines(options.input_filename, taskCategory=options.taskCategory,
                                   llmType=options.llmType, tokenizer=tokenizer)
    save_json_lines(options.output_filename, json_list)


if __name__ == "__main__":
    main()