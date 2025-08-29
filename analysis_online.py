import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm

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

def extract_key_metrics(result, method_type, filename, qid, rid):
    """Extract only essential metrics from a single result"""
    try:
        # Basic info
        is_correct = result.get('is_voted_correct', False)
        
        # Token metrics
        token_stats = result.get('token_stats', {})
        if method_type == 'simple':
            total_tokens = token_stats.get('total_tokens', 0)
            total_traces = token_stats.get('total_traces_count', 0)
        else:  # original
            total_tokens = token_stats.get('total_tokens', 0)
            total_traces = token_stats.get('warmup_traces_count', 0) + token_stats.get('final_traces_count', 0)
        
        # Timing metrics
        timing_stats = result.get('timing_stats', {})
        if method_type == 'simple':
            generation_time = timing_stats.get('generation_time', 0)
        else:  # original
            warmup_gen = timing_stats.get('warmup_gen_time', 0)
            final_gen = timing_stats.get('final_gen_time', 0)
            generation_time = warmup_gen + final_gen
        
        return {
            'method': method_type,
            'question_id': qid,
            'run_id': rid,
            'filename': filename,
            'is_voted_correct': is_correct,
            'total_tokens': total_tokens,
            'total_traces': total_traces,
            'generation_time': generation_time,
            'avg_tokens_per_trace': total_tokens / total_traces if total_traces > 0 else 0
        }
        
    except Exception as e:
        print(f"Error extracting metrics from {filename}: {e}")
        return None

def load_and_analyze_results(outputs_dir="outputs"):
    """Load results, extract metrics immediately, and clean up memory"""
    
    if not os.path.exists(outputs_dir):
        print(f"Directory {outputs_dir} not found!")
        return pd.DataFrame()
    
    all_files = [f for f in os.listdir(outputs_dir) if f.endswith('.pkl')]
    print(f"Found {len(all_files)} pickle files")
    
    all_metrics = []
    load_errors = []
    
    for filename in tqdm(all_files, desc="Processing files"):
        parsed = parse_filename(filename)
        if not parsed:
            continue
            
        method_type, qid, rid, timestamp = parsed
        if int(rid) < 0 or int(rid) >= 8:  # Skip invalid run IDs
            continue
            
        filepath = os.path.join(outputs_dir, filename)
        
        try:
            # Load file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Extract only what we need
            metrics = extract_key_metrics(data, method_type, filename, qid, rid)
            
            # Clean up - delete the large data structure immediately
            del data
            
            if metrics:
                all_metrics.append(metrics)
                
        except Exception as e:
            load_errors.append((filename, str(e)))
    
    if load_errors:
        print(f"Load errors: {len(load_errors)}")
        for fname, error in load_errors[:5]:  # Show first 5 errors
            print(f"  {fname}: {error}")
    
    df = pd.DataFrame(all_metrics)
    print(f"Successfully loaded {len(df)} experiments")
    
    return df

def analyze_results(df):
    """Analyze and compare results"""
    
    if df.empty:
        print("No data to analyze!")
        return df
    
    print(f"\n" + "="*50)
    print("DEEPCONF RESULTS ANALYSIS")
    print("="*50)
    
    # Basic counts
    simple_df = df[df['method'] == 'simple']
    original_df = df[df['method'] == 'original']
    
    print(f"Total experiments: {len(df)}")
    print(f"Self-Consistency method: {len(simple_df)} experiments")
    print(f"DeepConf method: {len(original_df)} experiments")
    print(f"Questions covered: {df['question_id'].nunique()}")
    
    # Method comparison
    print(f"\n" + "-"*40)
    print("METHOD COMPARISON")
    print("-"*40)
    
    for method in ['simple', 'original']:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            accuracy = method_data['is_voted_correct'].mean()
            avg_tokens = method_data['total_tokens'].mean()
            avg_time = method_data['generation_time'].mean()
            
            if method == 'simple':
                print(f"Self-Consistency:")
            else:
                print(f"DeepConf:")
            print(f"  Accuracy: {accuracy:.1%} ({method_data['is_voted_correct'].sum()}/{len(method_data)})")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Avg time: {avg_time:.1f}s")
    
    # Delta comparison (Original vs Simple)
    if len(simple_df) > 0 and len(original_df) > 0:
        print(f"\n" + "-"*40)
        print("DELTA ANALYSIS (DeepConf vs Baseline)")
        print("-"*40)
        
        simple_accuracy = simple_df['is_voted_correct'].mean()
        original_accuracy = original_df['is_voted_correct'].mean()
        
        simple_tokens = simple_df['total_tokens'].mean()
        original_tokens = original_df['total_tokens'].mean()
        
        simple_time = simple_df['generation_time'].mean()
        original_time = original_df['generation_time'].mean()
        
        print(f"\nAccuracy delta: {original_accuracy - simple_accuracy:+.1%}")
        print(f"Token delta: {original_tokens - simple_tokens:+.0f} tokens ({(original_tokens - simple_tokens)/simple_tokens:+.1%})")
        print(f"Time delta: {original_time - simple_time:+.1f}s ({(original_time - simple_time)/simple_time:+.1%})")
            
    # Per-question summary
    if df['question_id'].nunique() > 1:
        print(f"\n" + "-"*40)
        print("PER-QUESTION SUMMARY")
        print("-"*40)
        
        question_summary = df.groupby(['question_id', 'method']).agg({
            'is_voted_correct': ['mean', 'count'],
            'total_tokens': 'mean',
            'generation_time': 'mean'
        }).round(3)
        
        print(question_summary)
    
    return df

def main():
    """Main analysis function"""
    print("Loading DeepConf results...")
    
    # Load and process results efficiently
    df = load_and_analyze_results()
    
    if df.empty:
        print("No valid results found!")
        return df
    
    # Analyze results
    df = analyze_results(df)
    
    # Save compact results
    if not df.empty:
        csv_filename = 'deepconf_results.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to '{csv_filename}'")
    
    return df

if __name__ == "__main__":
    results_df = main()