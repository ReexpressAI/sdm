# Copyright Reexpress AI, Inc. All rights reserved.

def gen(model, tokenizer, question, task=0, distribution: int = -1, max=300):
    model.switch_distribution(distribution)
    if task == 0:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user",
             "content": f"Classify the sentiment of the following movie review. Respond using the following JSON: {{\"sentiment\": str}}. REVIEW: {question.strip()}"},
        ]
    elif task == 1:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user",
             "content": f"Check the following document for hallucinations and/or factual inaccuracies. Respond using the following JSON: {{\"correctness\": bool}}. DOCUMENT: {question.strip()}"},
        ]
    elif task == 3:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{question.strip()}"},
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(prompt)
    response = generate(model, tokenizer, prompt=prompt, message_text_to_force_decode=None, max_tokens=max,
                        verbose=True)

# run in interpreter:
from mlx_lm import load, generate
import mlx.core as mx
import torch
import numpy as np

# dir to Phi-3.5
model, tokenizer = load("/Users/a/Documents/projects/mlx/repo_started_2025_01_05/phi3.5/microsoft--Phi-3.5-mini-instruct_mlx")

# adaptor dir includes the learned weights
adaptor_dir="/Users/a/Documents/projects/paper_experiments/paper/llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker"
# For reference comparison, running, non-best weights are saved in the inner-directory 'non_finalized_llm_weights': adaptor_dir="/Users/a/Documents/projects/paper_experiments/paper/llm_combined/phi3.5_cached_embeddings_tasksentiment_taskfactcheck_combined_gen_ai_0.95_/sagemaker/non_finalized_llm_weights"
model.add_adaptors(adaptor_dir)

question = "Please compare Python and Swift."
gen(model, tokenizer, question, task=3, distribution=3, max=300)
