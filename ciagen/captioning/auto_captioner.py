import base64
import glob
import os
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ciagen.utils.io import logger


class AutoCaptioner:
    """Automatically generate captions for images using OpenRouter or Ollama."""

    def __init__(
        self,
        engine: str = "openrouter",
        model: str = "google/gemini-2.0-flash-001",
        api_key: Optional[str] = None,
        image_formats: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.model = model
        self.image_formats = image_formats or ["png", "jpg", "jpeg"]

        if engine == "openrouter":
            self.api_key = api_key
        elif engine == "ollama":
            try:
                import ollama  # noqa: F401
            except ImportError:
                raise ImportError("ollama package required for captioning. Install with: pip install ollama")

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
                    elif self.engine == "openrouter":
                        caption = self._openrouter_caption(image_path)
                    else:
                        raise ValueError(f"Invalid engine: {self.engine}. Use 'openrouter' or 'ollama'")

                    logger.info(f"Caption for {image_path}: {caption}")
                    with open(caption_path, "w") as f:
                        f.write(caption)
                except Exception as e:
                    import traceback

                    logger.error(f"Error captioning {image_path}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            data = f.read()

        # Detect image format from magic bytes
        if data[:4] == b"\x89PNG":
            mime_type = "image/png"
        elif data[:2] == b"\xff\xd8":
            mime_type = "image/jpeg"
        else:
            mime_type = "image/jpeg"  # default

        return base64.b64encode(data).decode("utf-8"), mime_type

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

    def _openrouter_caption(self, image_path: str) -> str:
        """Call OpenRouter API directly using requests."""
        import requests

        image_base64, mime_type = self._image_to_base64(image_path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fennecinspace/ciagen",
            "X-Title": "CIAGEN",
        }
        payload = {
            "model": self.model,
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        # Debug: print response status and first 500 chars
        print(f"OpenRouter status: {response.status_code}")
        print(f"Response text (first 500): {response.text[:500]}")

        if response.status_code != 200:
            raise ValueError(f"OpenRouter API error {response.status_code}: {response.text[:200]}")

        data = response.json()

        # Verify it's a dict, not a string
        if isinstance(data, str):
            raise ValueError(f"Response is a string, not JSON: {data[:200]}")

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"No choices in response: {data}")

        return data["choices"][0]["message"]["content"]
