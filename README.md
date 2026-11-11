# Text-to-SQL with QLoRA on a Free T4: Fine-Tune, Evaluate, Serve with vLLM

Fine-tuning Qwen2.5-1.5B-Instruct with QLoRA raised exact-match accuracy on held-out text-to-SQL questions from **7.5% to 72.0%**, in 11 minutes of training on a free Colab T4. Served with vLLM, the model handles **27 requests/s** on the same GPU.

## Results

### Accuracy (200 held-out test questions)

| Model | Exact match (%) | Valid SQL (%) |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct, zero-shot | 7.5 | 96.0 |
| + QLoRA fine-tune (r=16, 1 epoch) | **72.0** | **97.0** |

- **Exact match**: the generated query equals the reference after normalization (lowercase, whitespace collapsed, trailing `;` removed).
- **Valid SQL**: the query runs without error in SQLite against the question's schema.

The base model already writes valid SQL (96%) but rarely in the dataset's style. Part of the gain is the model learning that style (for example, double-quoted string values), not only better reasoning. Exact match is strict: a correct query written differently counts as wrong, so true accuracy is likely higher than 72%.

### Serving speed (vLLM, merged 16-bit model, single T4)

| Concurrent requests | Requests/s | Output tokens/s | p50 latency | p95 latency |
| --- | --- | --- | --- | --- |
| 1 | 2.1 | 55 | 0.47 s | 0.72 s |
| 16 | 27.0 | 710 | 0.55 s | 0.87 s |

With 16 concurrent requests, throughput rose about 13x while median latency rose only 0.08 s. LLM decoding is memory-bandwidth bound, so vLLM's continuous batching uses GPU compute that would otherwise sit idle. Responses are short (about 26 output tokens on average); longer outputs would give different numbers.

## What I built

A text-to-SQL model: given a table schema and an English question, it returns one SQLite query. I fine-tuned [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with QLoRA on 5,000 examples from [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context), using a free Google Colab T4 GPU, then deployed it with vLLM as an OpenAI-compatible API.

Models on the Hugging Face Hub:

- LoRA adapter: [harimanikkothu/qwen2.5-1.5b-sql-lora](https://huggingface.co/harimanikkothu/qwen2.5-1.5b-sql-lora)
- Merged 16-bit model: [harimanikkothu/qwen2.5-1.5b-sql-merged](https://huggingface.co/harimanikkothu/qwen2.5-1.5b-sql-merged)

## How it works

1. **Data split.** The dataset is shuffled with seed 42: 200 rows for test, 5,000 for training, 200 for validation. The test set is never used in training.
2. **Prompt format.** One chat format is used for training, evaluation, and serving: a system message, then the schema and question as the user message, and the SQL as the assistant reply.
3. **Baseline first.** The base model is scored on the test set before any training.
4. **QLoRA training.** The base model is loaded in 4-bit, and LoRA adapters are trained on top with Unsloth and TRL. Loss is computed only on the SQL answer, not the prompt.
5. **Evaluation.** The fine-tuned model is scored with the same code, on the same test set, with greedy decoding.
6. **Serving.** The merged model is served with vLLM and benchmarked with concurrent requests through the OpenAI client.

Training settings:

| Setting | Value |
| --- | --- |
| LoRA rank / alpha / dropout | 16 / 16 / 0 |
| Target modules | q, k, v, o, gate, up, down projections |
| Learning rate | 2e-4, linear schedule, 10 warmup steps |
| Effective batch size | 16 (4 x 4 gradient accumulation) |
| Epochs / steps | 1 / 313 |
| Optimizer | AdamW 8-bit |
| Precision | fp16 (T4 has no bfloat16) |
| Training time | 10 min 51 s |
| Final validation loss | 0.0405 |

Validation loss fell steadily from 0.063 to 0.041 and flattened at the end, with no sign of overfitting, so one epoch was enough.

## Example

Question: *What venue did the away team Essendon play at?* (schema: `table_name_50`)

```sql
SELECT venue FROM table_name_50 WHERE away_team = "essendon"
```

## Error analysis

<!-- TODO: fill in after reviewing the 56 test questions the fine-tuned model got "wrong". -->
<!-- For each type, give a rough count and one example: -->
<!-- - Correct, but written differently from the reference -->
<!-- - Wrong column or table -->
<!-- - Wrong aggregate or condition (e.g. COUNT vs SUM, > vs >=) -->
<!-- - Quoting or case differences in values -->

## What I'd try next

- **Execution-based scoring:** insert sample rows into each schema and compare query results, so correct queries written differently are no longer counted as wrong.
- **Few-shot baseline:** compare fine-tuning against a prompt with three solved examples, to measure how much fine-tuning adds over prompting.
- **Rank sweep:** train with r = 8, 16, and 64 and compare accuracy and training time.
- **Bigger model:** repeat with Qwen2.5-7B-Instruct on a 24 GB GPU.
- **Quantized serving:** compare AWQ or GPTQ 4-bit serving against 16-bit for speed and accuracy.

## Reproduce

1. Open `01_train_eval.ipynb` in Google Colab with a T4 GPU: data, baseline, QLoRA training, evaluation, and pushing to the Hub.
2. Open `02_serve_vllm.ipynb` in a **separate** Colab notebook: vLLM serving, API evaluation, and benchmarking. Unsloth and vLLM pin different PyTorch versions, so keep them apart.

Library versions: vLLM 0.33.0.
<!-- TODO: add the output of: pip freeze | grep -E "unsloth|trl|peft|transformers|torch" from notebook 1 -->
