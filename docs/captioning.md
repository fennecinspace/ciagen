---
title: Captioning
description: Auto-caption images using OpenRouter, OpenAI, or Ollama vision models.
keywords: captioning, image captions, openrouter, openai, ollama, gemini, gpt, llava
---

# Captioning

Generate captions for images using a vision-language model.

## Supported Engines

| Engine | Default Model | Cost | Requires |
|--------|--------------|------|----------|
| `openrouter` | `google/gemini-2.0-flash-001` | Free | API key |
| `openai` | `gpt-4o-mini` | Paid | API key |
| `ollama` | `llava` | Free | Ollama running locally |

## Usage

```python
from ciagen import caption

caption(
    images="data/real/train/images/",
    captions_dir="data/real/train/captions/",
    engine="openrouter",
    model="google/gemini-2.0-flash-001",
    api_key="sk-or-v1-...",  # Get from https://openrouter.ai/keys
)
```

## CLI

```bash
ciagen caption \
    --images data/real/train/images/ \
    --output data/real/train/captions/ \
    --engine openrouter \
    --model google/gemini-2.0-flash-001 \
    --api-key YOUR_KEY
```

Captions are saved as `.txt` files matching each image's filename. Already-captioned images are skipped.
