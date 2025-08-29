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

# Online algorithm parameters
WARMUP_TRACES = 16
TOTAL_BUDGET = 256
CONFIDENCE_PERCENTILE = 90
WINDOW_SIZE = 2048


BRUMO_DPSK_8B_THRESHOLD = {17: 15.11, 13: 17.37, 25: 16.59, 6: 15.33, 8: 16.31, 4: 14.47, 0: 14.68, 10: 17.32, 20: 14.99, 15: 15.26, 18: 16.52, 7: 16.03, 19: 14.31, 5: 15.63, 11: 14.03, 9: 14.76, 21: 15.02, 3: 18.47, 1: 16.86, 23: 15.13, 22: 17.34, 14: 15.02, 24: 16.44, 28: 16.11, 2: 16.15, 16: 14.78, 27: 14.75, 12: 13.91, 26: 12.9, 29: 13.52}

def main(qid, rid, use_predefined_threshold):
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

    # Warmup phase for this problem
    print(f"  Warmup phase for question {qid}...")
    warmup_gen_start = time.time()
    if use_predefined_threshold:
        pass
    else:
        warmup_params = SamplingParams(
            n=WARMUP_TRACES,
            temperature=0.6,
            top_p=0.95,
            max_tokens=MAX_TOKENS,
            logprobs=20,
        )

        warmup_outputs = llm.generate([prompt], warmup_params)
    warmup_gen_time = time.time() - warmup_gen_start

    # Process warmup results
    warmup_process_start = time.time()
    if use_predefined_threshold:
        warmup_result = {'traces': [],'total_tokens':0}
        conf_bar = BRUMO_DPSK_8B_THRESHOLD[qid]
    else:
        warmup_result = process_batch_results(warmup_outputs, ground_truth, window_size=WINDOW_SIZE)
        print('Warmup min_confs:', warmup_result['min_confs'])
        conf_bar = float(np.percentile(warmup_result['min_confs'], CONFIDENCE_PERCENTILE))
    warmup_process_time = time.time() - warmup_process_start

    print(f"    Warmup completed: {warmup_gen_time:.2f}s gen, {warmup_process_time:.2f}s proc, conf_bar={conf_bar:.3f}")

    # Final phase for this problem
    print(f"  Final phase for question {qid}...")
    final_gen_start = time.time()
    final_params = SamplingParams(
        n=TOTAL_BUDGET - WARMUP_TRACES,
        temperature=0.6,
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        logprobs=20,
        extra_args={'enable_conf': True,
        'window_size': WINDOW_SIZE,
        'threshold': conf_bar}  # Use individual confidence bar as threshold
    )

    final_outputs = llm.generate([prompt], final_params)
    final_gen_time = time.time() - final_gen_start

    # Process final results
    final_process_start = time.time()
    final_result = process_batch_results(final_outputs, ground_truth, window_size=WINDOW_SIZE)
    final_process_time = time.time() - final_process_start
    print('Final min_confs:', final_result['min_confs'])

    print(f"    Final completed: {final_gen_time:.2f}s gen, {final_process_time:.2f}s proc")

    # Get traces for this problem
    warmup_traces = warmup_result['traces']
    final_traces = final_result['traces']

    # Calculate token statistics for this problem
    warmup_tokens = warmup_result['total_tokens']
    final_tokens = final_result['total_tokens']
    budget_tokens = warmup_tokens + final_tokens

    # Apply confidence threshold to final traces
    for trace in final_traces:
        if trace["min_conf"] < conf_bar:
            trace["stop_reason"] = "gconf_threshold"

    # Voting for final answer
    voting_answers = []
    voting_weights = []

    # Add warmup traces above threshold
    for trace in warmup_traces:
        if trace['min_conf'] >= conf_bar and trace['extracted_answer']:
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(trace['min_conf'])
    print('Warmup voting answers:', voting_answers, voting_weights)

    # Add final traces (skip early stopped ones)
    for trace in final_traces:
        if trace.get('stop_reason') == 'gconf_threshold':
            continue
        if trace['extracted_answer']:
            voting_answers.append(trace['extracted_answer'])
            voting_weights.append(trace['min_conf'])

    print('Final voting answers:', voting_answers, voting_weights)

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

    # Prepare results for this problem
    problem_result = {
        "question_id": qid,
        "run_id": rid,
        "question": question_data['question'],
        "ground_truth": ground_truth,
        "conf_bar": conf_bar,
        "warmup_traces": warmup_traces,
        "final_traces": final_traces,
        "voted_answer": voted_answer,
        "is_voted_correct": is_voted_correct,
        "token_stats": {
            "warmup_tokens": warmup_tokens,
            "final_tokens": final_tokens,
            "total_tokens": budget_tokens,
            "warmup_traces_count": len(warmup_traces),
            "final_traces_count": len(final_traces),
            "avg_tokens_per_warmup_trace": warmup_tokens / len(warmup_traces) if warmup_traces else 0,
            "avg_tokens_per_final_trace": final_tokens / len(final_traces) if final_traces else 0,
        },
        "timing_stats": {
            "total_execution_time": total_time,
            "data_load_time": data_load_time,
            "tokenizer_init_time": tokenizer_init_time,
            "llm_init_time": llm_init_time,
            "prompt_prep_time": prompt_prep_time,
            "processing_time": processing_time,
            "warmup_gen_time": warmup_gen_time,
            "warmup_process_time": warmup_process_time,
            "final_gen_time": final_gen_time,
            "final_process_time": final_process_time,
            "warmup_total_time": warmup_gen_time + warmup_process_time,
            "final_total_time": final_gen_time + final_process_time,
        },
        "config": {
            "model_path": MODEL_PATH,
            "warmup_traces": WARMUP_TRACES,
            "total_budget": TOTAL_BUDGET,
            "confidence_percentile": CONFIDENCE_PERCENTILE,
            "window_size": WINDOW_SIZE,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save result with rid in filename
    result_filename = f"outputs/deepconf_qid{qid}_rid{rid}_{timestamp}.pkl"
    with open(result_filename, 'wb') as f:
        pickle.dump(problem_result, f)

    # Print summary
    print(f"\n=== Question {qid} Summary ===")
    print(f"Run ID: {rid}")
    print(f"Voted answer: {voted_answer}")
    print(f"Ground truth: {ground_truth}")
    print(f"Correct: {is_voted_correct}")
    print(f"Confidence bar: {conf_bar:.3f}")
    print(f"Tokens - Warmup: {warmup_tokens}, Final: {final_tokens}, Total: {budget_tokens}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Processing time: {processing_time:.2f}s")

    print(f"\n=== Performance Metrics ===")
    if warmup_gen_time > 0:
        print(f"Warmup throughput: {warmup_tokens / warmup_gen_time:.1f} tokens/second")
    if final_gen_time > 0:
        print(f"Final throughput: {final_tokens / final_gen_time:.1f} tokens/second")
    if (warmup_gen_time + final_gen_time) > 0:
        print(f"Overall throughput: {budget_tokens / (warmup_gen_time + final_gen_time):.1f} tokens/second")

    print(f"\nResult saved to {result_filename}")

    return problem_result

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process a single question with DeepConf')
    parser.add_argument('--qid', type=int, help='Question ID to process (0-based index)')
    parser.add_argument('--rid', type=str, help='Run ID for file naming')
    parser.add_argument('--use_predefined_threshold', action='store_true', help='Use predefined confidence threshold instead of computing from warmup')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = main(args.qid, args.rid, args.use_predefined_threshold)
