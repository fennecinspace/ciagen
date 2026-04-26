# `caption()`

Generate captions for images using a vision-language model.

```python
from ciagen import caption

caption(
    images="data/real/train/images/",
    captions_dir="data/real/train/captions/",
    engine="ollama",
    model="llava",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | `str \| Path` | **required** | Directory containing images to caption |
| `captions_dir` | `str \| Path` | **required** | Directory to save caption `.txt` files |
| `engine` | `str` | `"openai"` | Captioning engine: `openai` or `ollama` |
| `model` | `str` | `"gpt-4o-mini"` | Vision model name |
| `api_key` | `str \| None` | `None` | API key (required for OpenAI) |
| `image_formats` | `list[str] \| None` | `None` | Supported image formats |

## Behavior

- Creates a `.txt` file per image in `captions_dir` with the same stem name
- Skips images that already have a caption file
- Supports both [OpenAI](https://platform.openai.com/) and [Ollama](https://ollama.com/) engines
