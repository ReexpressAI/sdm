from model import Passthrough

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
import codecs
import json
from scipy.special import softmax

def get_precomputed_model_outputs_as_tensordataset(filename_with_path):
    # Return the original model output as a TensorDataset as input to the existing RAPS code. We also calculate
    # model accuracy for reference.

    prediction_metadata_list = []
    output_logits = []
    overall_accuracy = []
    true_labels = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            prediction_metadata_list.append(json_obj)
            output_logits.append(json_obj["logits"])
            overall_accuracy.append(json_obj["label"] == np.argmax(json_obj["logits"]))
            assert np.argmax(json_obj["logits"]) == json_obj["prediction"]
            true_labels.append(json_obj["label"])
    print(f"Total input documents: {len(output_logits)}")
    print(f"Overall accuracy: {np.mean(overall_accuracy)}")
    dataset = torch.utils.data.TensorDataset(torch.tensor(output_logits), torch.LongTensor(true_labels))
    return dataset, prediction_metadata_list


def convert_sparse_numpy_prediction_set_to_flat_tensor(sparse_numpy_prediction_set, number_of_classes):
    prediction_set = torch.zeros(number_of_classes).bool()  # or BoolTensor
    for class_id in sparse_numpy_prediction_set:
        prediction_set[class_id] = True
    return prediction_set


def get_raps_prediction_sets(val_loader, model,
                             output_type=None, add_model_prediction_to_set=None, number_of_classes=None,
                             prediction_sets_label=None, prediction_metadata_list=None):
    assert prediction_metadata_list is not None, \
        f"prediction_metadata_list is required to add required metadata for downstream analysis"
    prediction_sets_list = []
    overall_accuracy = []
    token_id = 0
    count_of_missing_prediction_in_set = 0
    for i, (x, target) in enumerate(val_loader):
        # compute output
        output, S = model(x)

        for batch_i in range(output.shape[0]):
            output_type_prediction_class = torch.argmax(output[batch_i], dim=0).item()
            query_true_token_label = target[batch_i].item()
            overall_accuracy.append(output_type_prediction_class == query_true_token_label)
            metadata = prediction_metadata_list[token_id]
            assert query_true_token_label == metadata["label"]

            prediction_set = convert_sparse_numpy_prediction_set_to_flat_tensor(S[batch_i], number_of_classes)
            if not prediction_set[output_type_prediction_class]:
                count_of_missing_prediction_in_set += 1
            if add_model_prediction_to_set:
                prediction_set[output_type_prediction_class] = True
            prediction_set_structure_for_token = \
                {"sent_i": metadata["id"],
                 "token_id": token_id,
                 "query_true_token_label": query_true_token_label,
                 "q_hat": None,
                 "n": None,
                 "output_type": output_type,
                 "add_model_prediction_to_set": add_model_prediction_to_set,
                 "prediction_set": prediction_set,
                 "label": prediction_sets_label,
                 "predicted_class": output_type_prediction_class
                 }
            prediction_sets_list.append(prediction_set_structure_for_token)
            token_id += 1
    print(f"===")
    label = f"Test Accuracy"
    print_summary_stats(label, overall_accuracy, only_display_mean=True)
    print(f"Total count of sets missing the predicted value: {count_of_missing_prediction_in_set}")
    return prediction_sets_list


def evaluate_prediction_sets(options, prediction_sets_list, prediction_set_type):
    coverage_by_token = []
    coverage_by_token_size = []
    coverage_by_token_by_class = {}
    coverage_by_token_by_class_by_cardinality = {}
    # Typically Frequentist coverage is evaluated in a class-conditional manner, but
    # here we also include stratification by the predicted class for reference:
    coverage_by_token_by_predicted_class = {}
    coverage_by_token_by_prediction_by_cardinality = {}
    for class_i in range(options.number_of_classes):
        coverage_by_token_by_class[f"true_class_{class_i}"] = []
        coverage_by_token_by_predicted_class[f"predicted_class_{class_i}"] = []
        for cardinality in range(options.number_of_classes + 1):
            coverage_by_token_by_class_by_cardinality[f"true_class_{class_i}_cardinality_{cardinality}"] = []
            coverage_by_token_by_prediction_by_cardinality[f"predicted_class_{class_i}_cardinality_{cardinality}"] = []

    for i, prediction_set_structure_for_token in enumerate(prediction_sets_list):
        query_true_token_label = prediction_set_structure_for_token["query_true_token_label"]
        prediction_set = prediction_set_structure_for_token["prediction_set"]
        coverage_by_token.append(prediction_set[query_true_token_label].item())
        coverage_by_token_by_class[f"true_class_{query_true_token_label}"].append(
            prediction_set[query_true_token_label].item())

        cardinality_of_set = int(torch.sum(prediction_set).item())
        coverage_by_token_size.append([prediction_set[query_true_token_label].item(), cardinality_of_set])

        coverage_by_token_by_class_by_cardinality[
            f"true_class_{query_true_token_label}_cardinality_{cardinality_of_set}"].append(
            prediction_set[query_true_token_label].item())
        if i == 0:
            output_type = prediction_set_structure_for_token["output_type"]
            add_model_prediction_to_set = prediction_set_structure_for_token["add_model_prediction_to_set"]
        else:
            assert output_type == prediction_set_structure_for_token["output_type"]
            assert add_model_prediction_to_set == prediction_set_structure_for_token["add_model_prediction_to_set"]
        predicted_class = prediction_set_structure_for_token["predicted_class"]
        coverage_by_token_by_predicted_class[f"predicted_class_{predicted_class}"].append(
            prediction_set[query_true_token_label].item())
        coverage_by_token_by_prediction_by_cardinality[
            f"predicted_class_{predicted_class}_cardinality_{cardinality_of_set}"].append(
            prediction_set[query_true_token_label].item())

    label = f"All classes: {prediction_set_type}: {output_type}: prediction added to set: {add_model_prediction_to_set}"
    dataset_size = len(prediction_sets_list)
    print(f"dataset size: {dataset_size}")
    print_summary_stats(label, coverage_by_token, only_display_mean=True, total=dataset_size)

    label = f"Cardinality: All classes: {prediction_set_type}: {output_type}: prediction added to set: {add_model_prediction_to_set}"
    print_summary_stats(label, [x[1] for x in coverage_by_token_size], only_display_mean=True, total=dataset_size)

    for cardinality in range(options.number_of_classes+1):
        label = f"All classes, cardinality {cardinality}: {prediction_set_type}: {output_type}: " \
                f"prediction added to set: {add_model_prediction_to_set}"
        print_summary_stats(
            label, [x[0] for x in coverage_by_token_size if x[1] == cardinality], only_display_mean=True,
            total=dataset_size)

    for class_i in range(options.number_of_classes):
        label = f"True class {class_i}: {prediction_set_type}: {output_type}: " \
                f"prediction added to set: {add_model_prediction_to_set}"
        print_summary_stats(label, coverage_by_token_by_class[f"true_class_{class_i}"], only_display_mean=True,
                            total=dataset_size)

        for cardinality in range(options.number_of_classes + 1):
            label = f"Cardinality {cardinality}, True class {class_i}: {prediction_set_type}: {output_type}: " \
                    f"prediction added to set: {add_model_prediction_to_set}"
            print_summary_stats(label,
                                coverage_by_token_by_class_by_cardinality[f"true_class_{class_i}_cardinality_{cardinality}"],
                                only_display_mean=True,
                                extra_indent="\t", total=dataset_size)
    print(f"### Prediction-conditional stratification for reference: ###")
    for class_i in range(options.number_of_classes):
        label = f"Predicted class {class_i}: {prediction_set_type}: {output_type}: " \
                f"prediction added to set: {add_model_prediction_to_set}"
        print_summary_stats(label, coverage_by_token_by_predicted_class[f"predicted_class_{class_i}"],
                            only_display_mean=True, total=dataset_size)

        for cardinality in range(options.number_of_classes + 1):
            label = f"Cardinality {cardinality}, Predicted class {class_i}: {prediction_set_type}: {output_type}: " \
                    f"prediction added to set: {add_model_prediction_to_set}"
            print_summary_stats(label,
                                coverage_by_token_by_prediction_by_cardinality[f"predicted_class_{class_i}_cardinality_{cardinality}"],
                                only_display_mean=True,
                                extra_indent="\t", total=dataset_size)


def print_summary_stats(label, list_of_floats, only_display_mean=False, display_quartiles=True, extra_indent="",
                        total=None):
    print(f"\t{extra_indent}{label}")
    if len(list_of_floats) > 0:
        if only_display_mean:
            if total is not None and total > 0:
                print(f"\t\t{extra_indent}"
                      f"mean: {np.mean(list_of_floats)}; "
                      f"total: {len(list_of_floats)} ({len(list_of_floats)/total})")
            else:
                print(f"\t\t{extra_indent}"
                      f"mean: {np.mean(list_of_floats)}; "
                      f"total: {len(list_of_floats)}")
        else:
            print(f"\t\t{extra_indent}"
                  f"mean: {np.mean(list_of_floats)}; "
                  f"min: {np.min(list_of_floats)}; "
                  f"max: {np.max(list_of_floats)}; "
                  f"std: {np.std(list_of_floats)}; "
                  f"total: {len(list_of_floats)}")
            if display_quartiles:
                first_quartile, third_quartile = np.quantile(list_of_floats, [1/3, 2/3])  # 0.25, 0.75
                print(f"\t\t{extra_indent}"
                      f"median: {np.quantile(list_of_floats, 0.5)}; "
                      f"1/3, 2/3 quartiles: {[first_quartile, third_quartile]}; "
                      f"inter-quartile range: {third_quartile-first_quartile}")
    else:
        print(f"\t\t{extra_indent}Empty list")


def eval_temp_scaling(val_loader, model, probability_threshold=0.95, number_of_classes=2):
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        total_accuracy = []
        constrained_accuracy = []
        constrained_accuracy_unscaled_logits = []

        temperature_scaling_class_conditional_constrained_accuracy = {}
        temperature_scaling_prediction_conditional_constrained_accuracy = {}
        softmax_class_conditional_constrained_accuracy = {}
        softmax_prediction_conditional_constrained_accuracy = {}
        for class_i in range(number_of_classes):
            temperature_scaling_class_conditional_constrained_accuracy[class_i] = []
            temperature_scaling_prediction_conditional_constrained_accuracy[class_i] = []
            softmax_class_conditional_constrained_accuracy[class_i] = []
            softmax_prediction_conditional_constrained_accuracy[class_i] = []

        for i, (x, target) in enumerate(val_loader):
            target = target
            # compute output
            logits, S = model(x)

            logits_numpy = logits.detach().cpu().numpy()
            scores = softmax(logits_numpy/model.T.item(), axis=1)
            softmax_without_temp = softmax(logits_numpy, axis=1)

            for doc_i in range(logits.shape[0]):
                prediction = torch.argmax(logits[doc_i])
                total_accuracy.append(prediction == target[doc_i])
                if softmax_without_temp[doc_i][prediction] >= probability_threshold:
                    constrained_accuracy_unscaled_logits.append(prediction == target[doc_i])
                    softmax_class_conditional_constrained_accuracy[target[doc_i].item()].append(prediction == target[doc_i])
                    softmax_prediction_conditional_constrained_accuracy[prediction.item()].append(prediction == target[doc_i])

                if scores[doc_i][prediction] >= probability_threshold:
                    constrained_accuracy.append(prediction == target[doc_i])
                    temperature_scaling_class_conditional_constrained_accuracy[target[doc_i].item()].append(prediction == target[doc_i])
                    temperature_scaling_prediction_conditional_constrained_accuracy[prediction.item()].append(prediction == target[doc_i])
        print(f"Total Accuracy: {np.mean(total_accuracy)}")
        dataset_size = len(total_accuracy)
        print(
            f"Constrained Accuracy *before* temperature scaling (i.e., just softmax) at threshold={probability_threshold}: "
            f"{np.mean(constrained_accuracy_unscaled_logits)} out of {len(constrained_accuracy_unscaled_logits)} "
            f"({len(constrained_accuracy_unscaled_logits)/dataset_size})")
        print(f"Constrained Accuracy after temperature scaling (temp={model.T.item()}) at threshold={probability_threshold}: "
              f"{np.mean(constrained_accuracy)} out of {len(constrained_accuracy)} ({len(constrained_accuracy)/dataset_size})")

        print(f"### Softmax: Class-conditional stratification at threshold={probability_threshold} ###")
        for class_i in range(number_of_classes):
            label = f"True class {class_i}"
            print_summary_stats(label, softmax_class_conditional_constrained_accuracy[class_i],
                                only_display_mean=True, total=dataset_size)
        print(f"### Softmax: Prediction-conditional stratification at threshold={probability_threshold} ###")
        for class_i in range(number_of_classes):
            label = f"Predicted class {class_i}"
            print_summary_stats(label, softmax_prediction_conditional_constrained_accuracy[class_i],
                                only_display_mean=True, total=dataset_size)
        print(f">>> Temperature Scaling: Class-conditional stratification at threshold={probability_threshold} <<<")
        for class_i in range(number_of_classes):
            label = f"True class {class_i}"
            print_summary_stats(label, temperature_scaling_class_conditional_constrained_accuracy[class_i],
                                only_display_mean=True, total=dataset_size)
        print(f">>> Temperature Scaling: Prediction-conditional stratification at threshold={probability_threshold} <<<")
        for class_i in range(number_of_classes):
            label = f"Predicted class {class_i}"
            print_summary_stats(label, temperature_scaling_prediction_conditional_constrained_accuracy[class_i],
                                only_display_mean=True, total=dataset_size)


def main():
    parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
    parser.add_argument("--batch_size", default=50, type=int, help="batch size")
    parser.add_argument("--seed", default=1776, type=int, help="seed_value")
    # parser.add_argument(
    #     "--training_files", default="",
    #     help="")
    parser.add_argument(
        "--calibration_files", default="",
        help="")
    parser.add_argument(
        "--eval_files", default="",
        help="")

    parser.add_argument("--number_of_classes", default=2, type=int,
                        help="Number of prediction classes. Labels should be in [0, --number_of_classes].")
    parser.add_argument("--empirical_coverage", default=0.9, type=float,
                        help="This should match the 'empirical_coverage' parameter used to construct the "
                             "provided --knn_bound_prediction_sets_stats_file.")
    parser.add_argument("--lambda_criterion", default="size",
                        help="'size' or 'adaptiveness' or float in [0, 1.0]")

    parser.add_argument("--number_formatter", default=".2f", type=str,
                        help="Formatter for main output. Default: '.2f'")
    parser.add_argument("--run_aps_baseline", default=False, action='store_true',
                        help="If --run_aps_baseline, lambda is set to 0 and the option --lambda_criterion is ignored.")
    parser.add_argument("--probability_threshold", default=0.95, type=float,
                        help="probability_threshold")

    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    options = args

    calib_dataset, calib_dataset_prediction_metadata_list = \
        get_precomputed_model_outputs_as_tensordataset(args.calibration_files)

    eval_dataset, eval_dataset_prediction_metadata_list = \
        get_precomputed_model_outputs_as_tensordataset(args.eval_files)

    # Initialize loaders
    calib_loader = \
        torch.utils.data.DataLoader(calib_dataset, batch_size=options.batch_size, shuffle=True, pin_memory=True)
    # Note for the eval dataset, shuffle=False
    val_loader = \
        torch.utils.data.DataLoader(eval_dataset, batch_size=options.batch_size, shuffle=False, pin_memory=True)

    # Get the model
    model_params = {"class_size": options.number_of_classes}
    model = Passthrough(**model_params)
    model.eval()

    allow_zero_sets = False
    # use the randomized version of conformal
    randomized = True
    adaptive_set_type = f"RAPS: lambda_criterion {options.lambda_criterion}"
    lambda_parameter = None
    if options.run_aps_baseline:
        print(f"Running the APS baseline. --options.lambda_criterion is ignored.")
        adaptive_set_type = "APS (i.e., lambda=0)"
        lambda_parameter = 0.0

    print(f"{adaptive_set_type}")
    print(f"empirical_coverage: {options.empirical_coverage} (alpha={1 - options.empirical_coverage})")
    # Conformalize model
    # model = \
    # ConformalModel(model, calib_loader, alpha=0.1, lamda=0, randomized=randomized, allow_zero_sets=allow_zero_sets)
    model = ConformalModel(model, calib_loader, alpha=1 - options.empirical_coverage, lamda=lambda_parameter,
                           lamda_criterion=options.lambda_criterion, randomized=randomized,
                           allow_zero_sets=allow_zero_sets)
    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    validate(val_loader, model, print_bool=True)

    print("Complete!")
    eval_temp_scaling(val_loader, model, probability_threshold=options.probability_threshold,
                      number_of_classes=options.number_of_classes)


    prediction_sets_label = adaptive_set_type
    raps_prediction_sets_list = get_raps_prediction_sets(val_loader, model, output_type="n/a",
                                                         add_model_prediction_to_set=False, #not allow_zero_sets,
                                                         number_of_classes=options.number_of_classes,
                                                         prediction_sets_label=prediction_sets_label,
                                                         prediction_metadata_list=eval_dataset_prediction_metadata_list)


    evaluate_prediction_sets(options, raps_prediction_sets_list, prediction_sets_label)
    total_n_in_eval_set = len(raps_prediction_sets_list)
    print(f"Total N: {total_n_in_eval_set}")


if __name__ == "__main__":
    main()