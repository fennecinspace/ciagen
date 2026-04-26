import base64
import glob
import os
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ciagen.utils.io import logger


class AutoCaptioner:
    """Automatically generate captions for images using OpenAI or Ollama vision models."""

    def __init__(
        self,
        engine: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        image_formats: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.model = model
        self.image_formats = image_formats or ["png", "jpg", "jpeg"]

        if engine == "openai":
            import openai

            openai.api_key = api_key

    def __call__(self, paths: dict) -> None:
        """Caption images from Hydra paths dict (backward compat)."""
        to_process = []

        splits = [
            ("real_images", "real_captions"),
            ("val_images", "val_captions"),
            ("test_images", "test_captions"),
        ]
        for split in splits:
            images_key, captions_key = split
            if images_key in paths and captions_key in paths:
                images_path = Path(paths[images_key])
                captions_path = Path(paths[captions_key])
                captions_path.mkdir(parents=True, exist_ok=True)
                to_process.append((images_path, captions_path))

        for images_path, captions_path in to_process:
            self._caption_directory(images_path, captions_path)

    def caption_directory(
        self,
        images_path: str | Path,
        captions_path: str | Path,
    ) -> None:
        """Generate captions for all images in a directory."""
        images_path = Path(images_path)
        captions_path = Path(captions_path)
        captions_path.mkdir(parents=True, exist_ok=True)
        self._caption_directory(images_path, captions_path)

    def _caption_directory(self, images_path: Path, captions_path: Path) -> None:
        images = []
        for fmt in self.image_formats:
            images += glob.glob(str(images_path.absolute()) + f"/*.{fmt}")

        for image_path in tqdm(images, desc="Captioning images"):
            caption_path = os.path.join(captions_path, image_path.split("/")[-1].split(".")[0] + ".txt")

            if not os.path.exists(caption_path):
                try:
                    if self.engine == "ollama":
                        caption = self._ollama_caption(image_path)
                    elif self.engine == "openai":
                        caption = self._openai_caption(image_path)
                    else:
                        raise ValueError(f"Invalid engine: {self.engine}. Use 'ollama' or 'openai'")

                    logger.info(f"Caption for {image_path}: {caption}")
                    with open(caption_path, "w") as f:
                        f.write(caption)
                except Exception as e:
                    logger.error(f"Error captioning {image_path}: {e}")

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _ollama_caption(self, image_path: str) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "What is in this picture?",
                    "images": [image_path],
                },
                {
                    "role": "system",
                    "content": (
                        "Don't say stuff like 'This Image contains' or 'This image Depicts', "
                        "just get directly into the description of the content and the objects. "
                        "Don't repeat the prompt, and don't talk like you're an assistant, "
                        "simply give a description of what's in the image."
                    ),
                },
            ],
        )
        return response["message"]["content"]

    def _openai_caption(self, image_path: str) -> str:
        import openai

        image_base64 = self._image_to_base64(image_path)
        response = openai.chat.completions.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Don't say stuff like 'This Image contains' or 'This image Depicts', "
                                "just get directly into the description of the content of the image. "
                                "Don't repeat the prompt, and don't talk like you're an assistant, "
                                "simply give a description of what's in the image."
                            ),
                        },
                    ],
                },
            ],
        )
        return str(response.choices[0].message.content)
