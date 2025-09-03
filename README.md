# Offline Results Reproduce

See README-offline.md

# Online Results Reproduce

See README-online.md

# Dataset prepare

Taking aime25 as an example. All needed dataset can be found under MathArena (https://huggingface.co/MathArena)

```
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("MathArena/aime_2025", split="train")

# Convert to JSONL
with open("aime_2025.jsonl", "w", encoding="utf-8") as f:
    for example in dataset:
        entry = {
            "question": example["problem"],
            "answer": str(example["answer"])
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} examples to aime_2025.jsonl")
```

# License
DeepConf is MIT licensed, as found in the LICENSE file.