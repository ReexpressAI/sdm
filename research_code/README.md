# Research Codebase

## Overview

The script `reexpress.py` is the main entry point. It takes as input pre-cached hidden states for an underlying network, adds an sdm activation layer, and constructs an sdm estimator. As noted in the paper, the sdm estimator can then be used as the basis for building reliable multi-stage, sequential LLM pipelines. The code also provides an implementation of an sdm network, incorporating this behavior directly into the LLM architecture and fine-tuning process. 

Note that the code for the sdm estimator over pre-cached hidden states is general-purpose as it can operate over hidden states from essentially any underlying network. In contrast, the current implementation of the sdm network is dependent on implementation details of Phi3.5 for the purposes of efficiently running the research experiments on-device. A more general version of the latter (including for use on CUDA devices) is in development, as noted in the main directory.

## Dependencies

The code has only been fully tested on a 2023 M2 Ultra (76 core) Mac Studio with 128 GB of RAM running macOS Sonoma 14.

The dependencies can be installed into a conda environment via:

> `conda env create -f research_code/dependencies/environment.yml`

> `conda activate baseEnv1`

`mlx-lm` needs to be installed locally, as we have modified the inference code (`mlx-examples_editable/llms/mlx_lm/utils.py`) and phi3 model file (`mlx-examples_editable/llms/mlx_lm/models/phi3.py`). We have lightly pruned that codebase to save space and simplify the presentation; refer to the original mlx-examples repo to add additional models and functionality. Use:

> `cd research_code/dependencies/third_party_code/mlx-examples_editable/llms`

> `pip install -e .`

Note that the research codebase is particularly dependent on the version of FAISS, NumPy, and Torch to work as expected. More recent versions of FAISS may install an incompatible version of NumPy (and ditto with many other Python packages). If you install a package that installs different versions, you might be able to use the following to down/up-grade and recover:

> `pip install torch==2.3.0 numpy==1.26.4`

## Replication of Experiments

The instructions in the `research_code/replication_scripts` directory step through replicating each of the experiments. Note that the scripts require setting directory and file environment variables and are arranged in blocks as step-by-step tutorials. They are **not** structured to be run as fully automated command-line scripts (e.g, as with a global command-line call `./script.sh` in one go).

## Data, Model, and Output Logs Archive

An archive of all data, model, and output log files for the paper's experiments (which are replicated by the provided scripts), is available [here](https://drive.google.com/file/d/1P3gx9njLEA7oqr5VPYFX01TmgfCzJRkT/view?usp=sharing). It is a 20 GB .zip file and approximately 45 GB uncompressed.


## Additional Notes

> The references to Algorithms, Equations, Definitions, etc. in code comments refer to the arXiv version of the paper.

> If needed to replicate the results from scratch, additional files in `support/aux_move_to_main_dir_if_needed` need to be moved to the main directory before using since they (may) have import dependencies. These files are otherwise not needed for general use, so we have moved them to simplify the main code directory. 

> If you want to get a quick sense of the behavior of the sdm activation function, the code in `research_code/support_code/local_scaling_tutorial_code.py` has a version of the sdm activation that you can easily drop into a local interpreter (i.e., without class variables or other dependencies).

> Currently the generative AI code is only set up to work with Phi3.5 and is a rather complicated setup across Pytorch and MLX codebases in order to run the experiments on-device. We run a quantized version of the frozen weights, while updating the weights of M^pos in Float32 using Pytorch. We plan to release a more general purpose Pytorch codebase for the fine-tuning case using CUDA in the future, but we keep our original experiment code as-is here for archival purposes of replicating the paper's experiments.
