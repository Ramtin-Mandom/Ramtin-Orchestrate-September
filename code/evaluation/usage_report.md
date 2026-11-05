# Token usage and cost report

This report covers the full-dataset run (250 evaluation requests) that
produced `output.csv`, with the LLM steps enabled:

```bash
python code/main.py --use-ai-extraction --output output.csv
```

## Model

| Provider | Model | Used for |
|---|---|---|
| OpenAI | `gpt-4o-mini` | (1) interpreting user-level messages into typed cash-flow adjustments (`extraction/message_interpreter.py`); (2) extracting facts about one specific event from its linked message or receipt image (`extraction/event_extractor.py`) |

Both calls use temperature 0, a fixed seed and strict JSON-schema output. The
model never computes amounts or chooses recommendations. Every number in
`output.csv` comes from deterministic code.

## Calls and tokens

| Run | Model calls | Input tokens | Output tokens | Total tokens |
|---|---|---|---|---|
| **Full run, cold cache** (every message and image sent to the model) | **209** | **486,587** | **14,614** | **501,201** |
| Final run that wrote `output.csv` (message interpretations served from `.cache/`; only event-level extractions re-sent) | 47 | 361,531 | 4,950 | 366,481 |

How the cold-cache calls break down:
- **162 calls** to the message interpreter, about 125K input tokens. These are
  short text prompts.
- **47 calls** to the event extractor, about 362K input tokens. Most of those
  tokens are the 16 receipt images, which cost about 19K input tokens each.

Per request (cold cache, 250 requests):
- **0.84 model calls**
- **2,005 tokens** on average (1,946 input and 58 output)

## Estimated cost

Priced at OpenAI's published `gpt-4o-mini` list rates of **$0.15 per 1M
input tokens** and **$0.60 per 1M output tokens** (check current pricing):

| Run | Input cost | Output cost | Total | Per request |
|---|---|---|---|---|
| Full run, cold cache | $0.0730 | $0.0088 | **$0.082** | **$0.00033** |
| Final run (message cache warm) | $0.0542 | $0.0030 | $0.057 | $0.00023 |

The 25-request public sample run (`--evaluate-samples --use-ai-extraction`)
used 22 calls when cold (166,869 input / 1,809 output tokens, about $0.026).

## Effect of the LLM steps (25 public samples)

| Field | Rules only | Rules + LLM |
|---|---|---|
| affordability_status | 80% | **88%** |
| recommended_payment_method | 88% | **96%** |
| earliest_date_for_full_payment | 67% | **72%** |

Without `--use-ai-extraction` the pipeline makes **zero** model calls, costs
nothing and still produces a complete `output.csv`.

No API keys, credentials or other sensitive configuration are included in
this repository. The key is read from the `OPENAI_API_KEY` environment
variable only.
