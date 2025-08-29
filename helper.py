from collections import Counter
from dynasor.core.evaluator import math_equal
import numpy as np

def extract_answer(text):
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None

def quick_parse(text):
    if '\\text{' in text and '}' in text:
        # Find all occurrences of \text{...} and remove them
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            # Replace \text{content} with just content
            content = text[start + 6:end]  # 6 is length of '\text{'
            text = text[:start] + content + text[end + 1:]
    return text

def equal_func(answer, ground_truth):
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)

def compute_confidence(logprobs):
    """Compute confidence score from logprobs"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def compute_least_grouped(confs, group_size):
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means

def weighted_majority_vote(answers, weights):
    """Perform weighted majority voting"""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def prepare_prompt(question_data, tokenizer):
    """Prepare prompt for a single question"""
    prompt = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    # Format prompt using chat template
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": prompt}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt, ground_truth

def prepare_prompt_gpt(question_data, tokenizer, reasoning_effort):
    prompt = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    
    # Format prompt using chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True
    )
    
    return full_prompt, ground_truth


def process_output(output, ground_truth, window_size):
    """Process a single vLLM output"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence
    confs = compute_confidence(logprobs) if logprobs else []
    sliding_window = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    
    extracted_answer = extract_answer(text)
    
    is_correct = False
    if extracted_answer and ground_truth:
        try:
            is_correct = equal_func(extracted_answer, ground_truth)
        except:
            is_correct = str(extracted_answer) == str(ground_truth)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "group_confs": sliding_window,
        "min_conf": min(sliding_window) if sliding_window else 0,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
    }

def process_batch_results(batch_outputs, ground_truth, window_size):
    """Process batch results from vLLM for a single question"""
    question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    min_confs = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output(output, ground_truth, window_size)
        traces.append(trace_data)
        min_confs.append(trace_data["min_conf"])
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'min_confs': min_confs,
        'ground_truth': ground_truth,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

def process_output_offline(output, ground_truth, window_size):
    """Process a single vLLM output"""
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    
    # Calculate confidence
    confs = compute_confidence(logprobs) if logprobs else []
    
    extracted_answer = extract_answer(text)
    
    is_correct = False
    if extracted_answer and ground_truth:
        try:
            is_correct = equal_func(extracted_answer, ground_truth)
        except:
            is_correct = str(extracted_answer) == str(ground_truth)
    
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
    }


def process_batch_results_offline(batch_outputs, ground_truth, window_size):
    """Process batch results from vLLM for a single question"""
    question_outputs = batch_outputs[0].outputs
    
    # Process all traces for this question
    traces = []
    total_tokens = 0
    
    for output in question_outputs:
        trace_data = process_output_offline(output, ground_truth, window_size)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    
    return {
        'traces': traces,
        'ground_truth': ground_truth,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }