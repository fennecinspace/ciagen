from pathlib import Path
from typing import List, Optional

from ciagen.captioning.auto_captioner import AutoCaptioner
from ciagen.utils.io import logger


def caption(
    images: str | Path,
    captions_dir: str | Path,
    engine: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    image_formats: Optional[List[str]] = None,
) -> None:
    """Generate captions for images using a vision model.

    Args:
        images: Directory containing images to caption.
        captions_dir: Directory to save caption text files.
        engine: Captioning engine ('openai' or 'ollama').
        model: Vision model name.
        api_key: API key (required for OpenAI).
        image_formats: Supported image formats.
    """
    captioner = AutoCaptioner(
        engine=engine,
        model=model,
        api_key=api_key,
        image_formats=image_formats,
    )
    captioner.caption_directory(images, captions_dir)
    logger.info(f"Captions saved to {captions_dir}")
