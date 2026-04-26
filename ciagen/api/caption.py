from pathlib import Path
from typing import List, Optional

from ciagen.captioning.auto_captioner import AutoCaptioner
from ciagen.utils.io import logger

VALID_ENGINES = frozenset({"openrouter", "openai", "ollama"})


def _validate_caption(
    images: Path,
    captions_dir: Path,
    engine: str,
    api_key: Optional[str],
) -> None:
    if not images.is_dir():
        raise NotADirectoryError(f"Images directory does not exist: {images}")

    if engine not in VALID_ENGINES:
        raise ValueError(f"Invalid engine '{engine}'. Choose from: {', '.join(sorted(VALID_ENGINES))}")

    if engine in ("openai", "openrouter") and not api_key:
        raise ValueError(
            f"api_key is required for {engine} engine. "
            "Get a free key at https://openrouter.ai/keys (for openrouter) "
            "or https://platform.openai.com/api-keys (for openai)"
        )


def caption(
    images: str | Path,
    captions_dir: str | Path,
    engine: str = "openrouter",
    model: str = "google/gemini-2.0-flash-001",
    api_key: Optional[str] = None,
    image_formats: Optional[List[str]] = None,
) -> None:
    """Generate captions for images using a vision model.

    Args:
        images: Directory containing images to caption.
        captions_dir: Directory to save caption text files.
        engine: Captioning engine ('openrouter', 'openai', or 'ollama').
        model: Vision model name.
        api_key: API key (required for openrouter/openai).
        image_formats: Supported image formats.
    """
    images = Path(images)
    captions_dir = Path(captions_dir)

    _validate_caption(images, captions_dir, engine, api_key)

    captioner = AutoCaptioner(
        engine=engine,
        model=model,
        api_key=api_key,
        image_formats=image_formats,
    )
    captioner.caption_directory(images, captions_dir)
    logger.info(f"Captions saved to {captions_dir}")
