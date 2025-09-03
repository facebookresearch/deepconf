"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import time
import pickle
import numpy as np
from datetime import datetime
from helper import equal_func, prepare_prompt, weighted_majority_vote, process_batch_results
import os
import argparse

# Configuration
MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MAX_TOKENS = 64000
DATASET_FILE = "brumo_2025.jsonl"

# Algorithm parameters
TOTAL_BUDGET = 256
WINDOW_SIZE = 2048

def main(qid, rid):
    """
    Main function to process a single question

    Args:
        qid (int): Question ID to process (0-based index)
        rid (str): Run ID for file naming
    """
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Start total timer
    total_start_time = time.time()

    # Load data
    print(f"Loading data from {DATASET_FILE}...")
    data_load_start = time.time()
    with open(DATASET_FILE, 'r', encoding='utf-8') as file:
        data = [json.loads(line.strip()) for line in file]
    data_load_time = time.time() - data_load_start

    print(f"Loaded {len(data)} questions in {data_load_time:.2f} seconds")

    # Validate qid
    if qid >= len(data) or qid < 0:
        raise ValueError(f"Question ID {qid} is out of range (0-{len(data)-1})")

    question_data = data[qid]
    print(f"Processing question {qid}: {question_data['question'][:100]}...")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer_init_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer_init_time = time.time() - tokenizer_init_start
    print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")

    # Initialize vLLM engine
    print("Initializing vLLM engine...")
    llm_init_start = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    llm_init_time = time.time() - llm_init_start
    print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")

    # Prepare prompt for the specific question
    print("Preparing prompt...")
    prompt_prep_start = time.time()
    prompt, ground_truth = prepare_prompt(question_data, tokenizer)
    prompt_prep_time = time.time() - prompt_prep_start
    print(f"Prepared prompt in {prompt_prep_time:.2f} seconds")

    # Process the specific problem
    print(f"Starting processing for question {qid}...")
    processing_start = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all traces at once
    print(f"  Generating {TOTAL_BUDGET} traces for question {qid}...")
    generation_start = time.time()
    sampling_params = SamplingParams(
        n=TOTAL_BUDGET,
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        logprobs=20,
    )

    outputs = llm.generate([prompt], sampling_params)
    generation_time = time.time() - generation_start

    # Process results
    processing_results_start = time.time()
    result = process_batch_results(outputs, ground_truth, window_size=WINDOW_SIZE)
    processing_results_time = time.time() - processing_results_start

    print(f'Min confidences: {result["min_confs"][:10]}...') # Show first 10
    print(f"    Generation completed: {generation_time:.2f}s gen, {processing_results_time:.2f}s proc")

    # Get all traces
    all_traces = result['traces']
    total_tokens = result['total_tokens']

    # Voting for final answer - use all traces with valid answers
    voting_answers = []
    voting_weights = []

    for trace in all_traces:
        if trace['extracted_answer']:
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(1.0)

    print(f'Total voting candidates: {len(voting_answers)}')
    print(f'Sample voting answers: {voting_answers[:5]}') # Show first 5
    print(f'Sample voting weights: {voting_weights[:5]}') # Show first 5

    # Get voted answer
    voted_answer = weighted_majority_vote(voting_answers, voting_weights)
    is_voted_correct = False
    if voted_answer and ground_truth:
        try:
            is_voted_correct = equal_func(voted_answer, ground_truth)
        except:
            is_voted_correct = str(voted_answer) == str(ground_truth)

    processing_time = time.time() - processing_start
    total_time = time.time() - total_start_time

    # Calculate statistics
    correct_traces = sum(1 for trace in all_traces if trace['is_correct'])
    accuracy = correct_traces / len(all_traces) if all_traces else 0

    # Prepare results for this problem
    problem_result = {
        "question_id": qid,
        "run_id": rid,
        "question": question_data['question'],
        "ground_truth": ground_truth,
        "all_traces": all_traces,
        "voted_answer": voted_answer,
        "is_voted_correct": is_voted_correct,
        "accuracy": accuracy,
        "correct_traces_count": correct_traces,
        "token_stats": {
            "total_tokens": total_tokens,
            "total_traces_count": len(all_traces),
            "avg_tokens_per_trace": total_tokens / len(all_traces) if all_traces else 0,
        },
        "timing_stats": {
            "total_execution_time": total_time,
            "data_load_time": data_load_time,
            "tokenizer_init_time": tokenizer_init_time,
            "llm_init_time": llm_init_time,
            "prompt_prep_time": prompt_prep_time,
            "processing_time": processing_time,
            "generation_time": generation_time,
            "processing_results_time": processing_results_time,
        },
        "config": {
            "model_path": MODEL_PATH,
            "total_budget": TOTAL_BUDGET,
            "window_size": WINDOW_SIZE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save result with rid in filename
    result_filename = f"outputs/deepconf_simple_qid{qid}_rid{rid}_{timestamp}.pkl"
    with open(result_filename, 'wb') as f:
        pickle.dump(problem_result, f)

    # Print summary
    print(f"\n=== Question {qid} Summary ===")
    print(f"Run ID: {rid}")
    print(f"Voted answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Correct: {is_voted_correct}")
    print(f"Individual trace accuracy: {correct_traces}/{len(all_traces)} ({accuracy:.1%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Processing time: {processing_time:.2f}s")

    print(f"\n=== Performance Metrics ===")
    if generation_time > 0:
        print(f"Generation throughput: {total_tokens / generation_time:.1f} tokens/second")

    print(f"\nResult saved to {result_filename}")

    return problem_result

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process a single question with simplified DeepConf')
    parser.add_argument('--qid', type=int, help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, help='Run ID for file naming')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = main(args.qid, args.rid)
