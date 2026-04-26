import glob
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("ciagen")


def list_files(
    path: Path | str,
    formats: List[str],
    limit: Optional[int] = None,
) -> List[str]:
    """List files in a directory matching given extensions, optionally limited."""
    images: List[str] = []
    for fmt in formats:
        images += glob.glob(str(Path(path).absolute()) + f"/*.{fmt}")
    return images[:limit] if limit else images


def read_caption(
    caption_path: str,
    prompt_per_line: bool = False,
    extra_empty_caption: bool = False,
) -> List[str]:
    """Read a caption file and return prompts as a list of strings."""
    with open(caption_path, "r") as f:
        if prompt_per_line:
            lines = [line.strip() for line in f.readlines()]
        else:
            text = f.read()
            lines = [text.strip().replace("\n", " ").replace(",", "").replace(".", "")]

    if extra_empty_caption and "" not in lines:
        lines += [""]

    return lines


def create_files_list(image_files: List[str], txt_file_path: str) -> None:
    """Write a list of image file paths to a text file."""
    with open(txt_file_path, "w") as f:
        f.write("\n".join(image_files))


def normalizer(image: Image.Image) -> Image.Image:
    """Normalize image pixel values to the [0, 255] range."""
    img = np.array(image)
    return Image.fromarray(cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX))


def find_common_prefix(str_list: List[str]) -> str:
    return os.path.commonprefix(str_list)


def find_common_suffix(str_list: List[str]) -> str:
    str_list_inv = [x[::-1] for x in str_list]
    return find_common_prefix(str_list_inv)


def contains_word(string: str, words: list) -> bool:
    """Check if any word from the list appears in the string (case-insensitive)."""
    for word in words:
        if word.lower() in string.lower():
            return True
    return False
