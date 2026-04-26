# Captioning

Generate captions for images using a vision-language model.

## Supported Engines

| Engine | Default Model | Requires |
|--------|--------------|----------|
| `openai` | `gpt-4o-mini` | API key |
| `ollama` | `llava` | Ollama running locally |

## Usage

```python
from ciagen import caption

caption(
    images="data/real/train/images/",
    captions_dir="data/real/train/captions/",
    engine="ollama",
    model="llava",
)
```

## CLI

```bash
ciagen caption \
    --images data/real/train/images/ \
    --output data/real/train/captions/ \
    --engine openai \
    --model gpt-4o-mini \
    --api-key YOUR_KEY
```

Captions are saved as `.txt` files matching each image's filename. Already-captioned images are skipped.
