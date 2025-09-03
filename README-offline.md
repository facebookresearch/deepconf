# DeepConf (offline version)

DeepConf is a confidence-based approach for improving model performance while reducing computational costs. This method selects high-confidence responses to achieve better accuracy with significant token savings compared to traditional self-consistency baselines.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for vLLM)
- Git

### Installation

1. **Install Dynasor for judgement**
   ```bash
   git clone https://github.com/hao-ai-lab/Dynasor.git
   cd Dynasor && pip install . && cd -
   ```

2. **Install Python Dependencies**
   ```bash
   pip install transformers numpy pandas tqdm
   ```

3. **Install vLLM (Custom Version)**
   ```bash
    uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
   ```
   
   > **Note:** Make sure to enable vLLM V1.

## 📊 Usage

### Running Experiments

**Traces Generation:**
```bash
python deepconf-offline.py --qid 0 --rid 0
```

### Parameters

- `--qid`: Question ID (range: 0-29)
- `--rid`: Run ID (any integer for experiment tracking)

### Configuration

Modify these settings in the Python files for different experimental configurations:

```python
MODEL_PATH = "openai/gpt-oss-120b"
MAX_TOKENS = 130000
DATASET_FILE = "aime25.jsonl"

TOTAL_BUDGET = 512
WINDOW_SIZE = 2048
REASONING_EFFORT = 'high'
```
## Confidence Metrics

- **Mean Confidence**: Calculate the average confidence across all tokens in the reasoning trace
- **Tail Confidence**: Calculate the average confidence over the final 2048 tokens of the trace  
- **Bottom-10 Confidence**: Apply sliding windows (size 2048) across the entire trace, compute mean confidence for each window, then average the lowest 10% of window scores

## Voting Strategies

1. **Simple Majority Voting**: Standard majority rule without confidence weighting
2. **Confidence-Weighted Voting**: Weight each vote by its respective confidence score (Mean/Tail/Bottom-10)
3. **Filtered Confidence Voting**: Pre-filter high-confidence traces, then apply confidence-weighted voting to the selected subset

### Running Complete Experiments

For statistically valid results, ensure equal numbers of runs across all questions, you can change rid to get different results to compute an average:

```bash
# Example: Run all questions with 8 runs each
for qid in {0..29}; do
    python deepconf-offline.py --qid $qid --rid 0
done
```

### Analysis

After completing all experiments:
```bash
python analysis-offline.py
```

## 📈 Expected Results

Based on our experiments with GPT-OSS-120B on AIME25 dataset (total budget: 512, 1 runs per question):

You should expect accuracy improvement. The accuracy may vary and we recommend run 64 times to reproduce the results in the paper.

```
============================================================
VOTING MECHANISMS COMPARISON
============================================================

Voting Method Performance:
--------------------------------------------------------------------------------
Method                    Accuracy   Correct/Total   Avg_Votes    Avg_Conf  
--------------------------------------------------------------------------------
bottom_window_weighted    96.7%      29/30            481.3        9.031     
majority                  96.7%      29/30            481.3        0.000     
mean_confidence_weighted  96.7%      29/30            481.3        10.902    
min_window_weighted       96.7%      29/30            481.3        8.732     
tail_confidence_weighted  96.7%      29/30            481.3        12.671    
top10_bottom_window_filtered 100.0%     30/30            48.7         9.985     
top10_tail_filtered       100.0%     30/30            48.7         14.441    
```

## License
DeepConf is MIT licensed, as found in the LICENSE file.