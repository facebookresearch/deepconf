import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from tqdm import tqdm
from helper import equal_func

def parse_filename(filename):
    """Parse filename to extract method, qid, rid, timestamp"""
    # Pattern for simple method
    simple_pattern = r'^deepconf_simple_qid(\d+)_rid([^_]+)_(\d{8}_\d{6})\.pkl$'
    simple_match = re.match(simple_pattern, filename)
    
    if simple_match:
        qid, rid, timestamp = simple_match.groups()
        return 'simple', int(qid), rid, timestamp
    
    # Pattern for original method
    original_pattern = r'^deepconf_qid(\d+)_rid([^_]+)_(\d{8}_\d{6})\.pkl$'
    original_match = re.match(original_pattern, filename)
    
    if original_match:
        qid, rid, timestamp = original_match.groups()
        return 'original', int(qid), rid, timestamp
    
    return None

def calculate_mean_confidence(trace):
    """Calculate mean confidence from confs in a trace"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            return np.mean(confs) if confs else 0.0
        return 0.0
    except Exception as e:
        print(f"Error calculating mean confidence: {e}")
        return 0.0

def calculate_tail_confidence(trace, tail_tokens=2048):
    """Calculate mean confidence from the last N tokens"""
    try:
        if 'confs' in trace and trace['confs']:
            # tail confidence = np.mean(confs[-2048:])
            confs = trace['confs']
            tail_confs = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
            return np.mean(tail_confs) if tail_confs else 0.0
        return 0.0
    except Exception as e:
        print(f"Error calculating tail confidence: {e}")
        return 0.0

def calculate_bottom_window_confidence(trace, window_size=2048, bottom_percent=0.1):
    """Calculate mean confidence from sliding windows, return average of bottom percentile"""
    try:
        if 'confs' in trace and trace['confs']:
            confs = trace['confs']
            if len(confs) < window_size:
                return np.mean(confs)
            
            window_means = []
            
            current_sum = sum(confs[:window_size])
            window_means.append(current_sum / window_size)
            
            for i in range(1, len(confs) - window_size + 1):
                current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
                window_means.append(current_sum / window_size)
            
            if not window_means:
                return 0.0
            
            if bottom_percent == -1:
                return min(window_means)
            
            num_bottom = max(1, int(len(window_means) * bottom_percent))
            if num_bottom == 1:
                return min(window_means)
            else:
                bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
                return np.mean(bottom_means)
        
        return 0.0
    except Exception as e:
        print(f"Error calculating bottom window confidence: {e}")
        return 0.0

def majority_vote(answers):
    """Simple majority voting"""
    if not answers:
        return None
    
    vote_counts = Counter(answers)
    return vote_counts.most_common(1)[0][0]

def weighted_majority_vote(answers, weights):
    """Weighted majority voting"""
    if not answers or not weights or len(answers) != len(weights):
        return None
    
    vote_weights = defaultdict(float)
    for answer, weight in zip(answers, weights):
        vote_weights[answer] += weight
    
    if not vote_weights:
        return None
        
    return max(vote_weights.items(), key=lambda x: x[1])[0]

def filter_top_confidence(traces, confidence_type='mean', top_percent=0.1):
    """Filter traces by top confidence percentage"""
    if not traces:
        return []
    
    # Calculate confidences
    confidences = []
    for trace in traces:
        if confidence_type == 'mean':
            conf = calculate_mean_confidence(trace)
        elif confidence_type == 'tail':
            conf = calculate_tail_confidence(trace)
        elif confidence_type == 'bottom_window':
            conf = calculate_bottom_window_confidence(trace)
        else:
            conf = calculate_mean_confidence(trace)  # default fallback
        confidences.append(conf)
    
    # Get threshold for top percentage
    threshold = np.percentile(confidences, (1 - top_percent) * 100)
    
    # Filter traces
    filtered_traces = []
    for trace, conf in zip(traces, confidences):
        if conf >= threshold:
            filtered_traces.append(trace)
    
    return filtered_traces

def analyze_single_experiment(result_data, ground_truth):
    """Analyze a single experiment with different voting mechanisms"""
    all_traces = result_data.get('all_traces', [])
    
    # Extract valid traces with answers
    valid_traces = [trace for trace in all_traces if trace.get('extracted_answer')]
    
    if not valid_traces:
        return None
    
    # Extract answers for voting
    answers = [trace['extracted_answer'] for trace in valid_traces]
    
    # Calculate different types of confidences
    mean_confidences = [calculate_mean_confidence(trace) for trace in valid_traces]
    tail_confidences = [calculate_tail_confidence(trace) for trace in valid_traces]
    bottom_window_confidences = [calculate_bottom_window_confidence(trace) for trace in valid_traces]
    min_window_confidences = [calculate_bottom_window_confidence(trace, bottom_percent=-1) for trace in valid_traces]
    
    # Voting methods
    voting_results = {}
    
    # 1. Simple majority vote
    majority_answer = majority_vote(answers)
    voting_results['majority'] = {
        'answer': majority_answer,
        'is_correct': check_answer_correctness(majority_answer, ground_truth),
        'num_votes': len(answers)
    }
    
    # 2. Mean confidence weighted vote
    if any(c > 0 for c in mean_confidences):
        mean_weighted_answer = weighted_majority_vote(answers, mean_confidences)
        voting_results['mean_confidence_weighted'] = {
            'answer': mean_weighted_answer,
            'is_correct': check_answer_correctness(mean_weighted_answer, ground_truth),
            'avg_confidence': np.mean(mean_confidences),
            'num_votes': len(answers)
        }
    
    # 3. Tail confidence weighted vote
    if any(c > 0 for c in tail_confidences):
        tail_weighted_answer = weighted_majority_vote(answers, tail_confidences)
        voting_results['tail_confidence_weighted'] = {
            'answer': tail_weighted_answer,
            'is_correct': check_answer_correctness(tail_weighted_answer, ground_truth),
            'avg_confidence': np.mean(tail_confidences),
            'num_votes': len(answers)
        }
    
    # 4. Bottom window confidence weighted vote (10% bottom windows)
    if any(c > 0 for c in bottom_window_confidences):
        bottom_weighted_answer = weighted_majority_vote(answers, bottom_window_confidences)
        voting_results['bottom_window_weighted'] = {
            'answer': bottom_weighted_answer,
            'is_correct': check_answer_correctness(bottom_weighted_answer, ground_truth),
            'avg_confidence': np.mean(bottom_window_confidences),
            'num_votes': len(answers)
        }
    
    # 5. Min window confidence weighted vote
    if any(c > 0 for c in min_window_confidences):
        min_window_answer = weighted_majority_vote(answers, min_window_confidences)
        voting_results['min_window_weighted'] = {
            'answer': min_window_answer,
            'is_correct': check_answer_correctness(min_window_answer, ground_truth),
            'avg_confidence': np.mean(min_window_confidences),
            'num_votes': len(answers)
        }
    
    # 6. Top 10% tail confidence filtered + weighted vote
    top_tail_traces = filter_top_confidence(valid_traces, 'tail', 0.1)
    if top_tail_traces:
        top_tail_answers = [trace['extracted_answer'] for trace in top_tail_traces]
        top_tail_confidences = [calculate_tail_confidence(trace) for trace in top_tail_traces]
        
        if any(c > 0 for c in top_tail_confidences):
            top_tail_answer = weighted_majority_vote(top_tail_answers, top_tail_confidences)
            voting_results['top10_tail_filtered'] = {
                'answer': top_tail_answer,
                'is_correct': check_answer_correctness(top_tail_answer, ground_truth),
                'avg_confidence': np.mean(top_tail_confidences),
                'num_votes': len(top_tail_answers)
            }
    
    # 7. Top 10% bottom window confidence filtered + weighted vote
    top_bottom_traces = filter_top_confidence(valid_traces, 'bottom_window', 0.1)
    if top_bottom_traces:
        top_bottom_answers = [trace['extracted_answer'] for trace in top_bottom_traces]
        top_bottom_confidences = [calculate_bottom_window_confidence(trace) for trace in top_bottom_traces]
        
        if any(c > 0 for c in top_bottom_confidences):
            top_bottom_answer = weighted_majority_vote(top_bottom_answers, top_bottom_confidences)
            voting_results['top10_bottom_window_filtered'] = {
                'answer': top_bottom_answer,
                'is_correct': check_answer_correctness(top_bottom_answer, ground_truth),
                'avg_confidence': np.mean(top_bottom_confidences),
                'num_votes': len(top_bottom_answers)
            }
    
    return voting_results

def check_answer_correctness(answer, ground_truth):
    """Check if answer matches ground truth"""
    if answer is None or ground_truth is None:
        return False
    
    try:
        # Try to use equal_func if available (from the original helper)
        return equal_func(answer, ground_truth)
    except:
        return False

def load_and_analyze_voting(outputs_dir="outputs"):
    """Load results and analyze different voting mechanisms"""
    
    if not os.path.exists(outputs_dir):
        print(f"Directory {outputs_dir} not found!")
        return pd.DataFrame()
    
    all_files = [f for f in os.listdir(outputs_dir) if f.endswith('.pkl')]
    print(f"Found {len(all_files)} pickle files")
    
    all_results = []
    load_errors = []
    
    for filename in tqdm(all_files, desc="Processing files"):
        parsed = parse_filename(filename)
        if not parsed:
            continue
            
        method_type, qid, rid, timestamp = parsed
        filepath = os.path.join(outputs_dir, filename)
        
        try:
            # Load file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Get ground truth
            ground_truth = data.get('ground_truth')
            
            # Analyze voting mechanisms
            voting_analysis = analyze_single_experiment(data, ground_truth)
            
            if voting_analysis:
                # Create result record for each voting method
                for voting_method, result in voting_analysis.items():
                    record = {
                        'method': method_type,
                        'question_id': qid,
                        'run_id': rid,
                        'filename': filename,
                        'voting_method': voting_method,
                        'predicted_answer': result['answer'],
                        'ground_truth': ground_truth,
                        'is_correct': result['is_correct'],
                        'num_votes': result['num_votes'],
                        'avg_confidence': result.get('avg_confidence', 0.0)
                    }
                    all_results.append(record)
            
            # Clean up memory
            del data
                
        except Exception as e:
            load_errors.append((filename, str(e)))
    
    if load_errors:
        print(f"Load errors: {len(load_errors)}")
        for fname, error in load_errors[:5]:
            print(f"  {fname}: {error}")
    
    df = pd.DataFrame(all_results)
    print(f"Successfully processed {len(df)} voting results")
    
    return df

def analyze_voting_performance(df):
    """Analyze performance of different voting mechanisms"""
    
    if df.empty:
        print("No data to analyze!")
        return df
    
    print(f"\n" + "="*60)
    print("VOTING MECHANISMS COMPARISON")
    print("="*60)
    
    # Group by voting method and calculate performance
    voting_summary = df.groupby('voting_method').agg({
        'is_correct': ['count', 'sum', 'mean'],
        'num_votes': 'mean',
        'avg_confidence': 'mean'
    }).round(4)
    
    voting_summary.columns = ['Total_Experiments', 'Correct', 'Accuracy', 'Avg_Votes_Used', 'Avg_Confidence']
    
    print("\nVoting Method Performance:")
    print("-" * 80)
    print(f"{'Method':<25} {'Accuracy':<10} {'Correct/Total':<15} {'Avg_Votes':<12} {'Avg_Conf':<10}")
    print("-" * 80)
    
    for method in voting_summary.index:
        row = voting_summary.loc[method]
        accuracy = row['Accuracy']
        correct = int(row['Correct'])
        total = int(row['Total_Experiments'])
        avg_votes = row['Avg_Votes_Used']
        avg_conf = row['Avg_Confidence']
        
        print(f"{method:<25} {accuracy:<10.1%} {correct}/{total:<13} {avg_votes:<12.1f} {avg_conf:<10.3f}")
    
    # Method comparison by model type
    if df['method'].nunique() > 1:
        print(f"\n" + "-"*50)
        print("PERFORMANCE BY MODEL TYPE")
        print("-"*50)
        
        for model_type in df['method'].unique():
            print(f"\n{model_type.upper()} Model Results:")
            model_df = df[df['method'] == model_type]
            model_summary = model_df.groupby('voting_method')['is_correct'].agg(['count', 'sum', 'mean']).round(4)
            
            for voting_method in model_summary.index:
                row = model_summary.loc[voting_method]
                accuracy = row['mean']
                correct = int(row['sum'])
                total = int(row['count'])
                print(f"  {voting_method:<25}: {accuracy:.1%} ({correct}/{total})")
        
    return df

def main():
    """Main voting analysis function"""
    print("Loading DeepConf results for voting analysis...")
    
    # Load and process results
    df = load_and_analyze_voting()
    
    if df.empty:
        print("No valid results found!")
        return df
    
    # Analyze voting performance
    df = analyze_voting_performance(df)
    
    # Save detailed results
    if not df.empty:
        csv_filename = 'deepconf_voting_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nDetailed results saved to '{csv_filename}'")
        
        # Save summary statistics
        summary_df = df.groupby('voting_method').agg({
            'is_correct': ['count', 'sum', 'mean'],
            'num_votes': 'mean',
            'avg_confidence': 'mean'
        }).round(4)
        
        summary_csv = 'deepconf_voting_summary.csv'
        summary_df.to_csv(summary_csv)
        print(f"Summary statistics saved to '{summary_csv}'")
    
    return df

if __name__ == "__main__":
    results_df = main()