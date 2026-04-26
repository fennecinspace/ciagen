import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from diffusers.utils import load_image

from ciagen.extractors import instantiate_extractor
from ciagen.generators import SDCN, NaivePromptGenerator
from ciagen.utils.io import logger, read_caption

torch.backends.cudnn.benchmark = False


def generate(
    source: str | Path,
    output: str | Path,
    extractor: str,
    sd_model: str,
    cn_model: str,
    num_per_image: int = 1,
    seed: Union[int, List[int]] = 34567,
    device: str = "cuda",
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[str] = None,
    quality: int = 30,
    guidance_scale: float = 7.0,
    use_captions: bool = False,
    captions_dir: Optional[str] = None,
    modify_captions: bool = False,
    vocabulary_template: Optional[str] = None,
    generation_size: int = 10,
    cn_extra_settings: Optional[Dict] = None,
    image_formats: Optional[List[str]] = None,
) -> Dict:
    """Generate synthetic images from real images using Stable Diffusion + ControlNet.

    Args:
        source: Directory containing real source images.
        output: Directory to save generated images.
        extractor: Condition extractor name ('canny', 'openpose', 'segmentation', 'mediapipe_face').
        sd_model: HuggingFace model ID for Stable Diffusion.
        cn_model: HuggingFace model ID for ControlNet.
        num_per_image: Number of synthetic images to generate per real image.
        seed: Random seed(s) for generation.
        device: Device to use ('cuda' or 'cpu').
        prompt: Positive prompt(s). If None, captions are required.
        negative_prompt: Negative prompt for generation.
        quality: Number of inference steps.
        guidance_scale: Classifier-free guidance scale.
        use_captions: Whether to use caption files as prompts.
        captions_dir: Directory containing caption files (required if use_captions=True).
        modify_captions: Whether to modify captions using vocabulary substitution.
        vocabulary_template: Name of vocabulary config for prompt modification.
        generation_size: Number of prompt variations to generate.
        cn_extra_settings: Extra settings for ControlNet model loading.
        image_formats: Supported image formats for source images.

    Returns:
        Dictionary with generation results including count and output path.
    """
    if image_formats is None:
        image_formats = ["png", "jpeg", "jpg"]
    if cn_extra_settings is None:
        cn_extra_settings = {}

    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    real_images = []
    for fmt in image_formats:
        real_images += glob.glob(str(source.absolute()) + f"/*.{fmt}")
    real_images.sort()

    if not real_images:
        raise ValueError(f"No images found in {source}")

    if use_captions and captions_dir:
        captions = sorted(glob.glob(str(Path(captions_dir).absolute()) + "/*"))
        if len(captions) != len(real_images):
            raise ValueError(
                f"Caption count ({len(captions)}) doesn't match image count ({len(real_images)})"
            )
    else:
        captions = [None] * len(real_images)

    if isinstance(seed, list):
        seed = seed[0]

    prompt_generator = None
    if modify_captions and vocabulary_template:
        prompt_generator = NaivePromptGenerator(vocabulary_template)

    generator = SDCN(
        sd_model=sd_model,
        control_model=cn_model,
        seed=seed,
        device=device,
        cn_extra_settings=cn_extra_settings,
    )
    ext = instantiate_extractor(extractor)

    logger.info(f"Using extractor: {ext}")
    logger.info(f"Using generator: {generator}")
    logger.info(f"Real dataset size: {len(real_images)}")
    logger.info(f"Output: {output}")

    generated_count = 0
    errors = []

    for idx, image_path in enumerate(real_images):
        try:
            image = load_image(image_path)
            image_name = Path(image_path).stem

            condition = ext.extract(image)

            if use_captions and captions[idx]:
                positive_prompts = [
                    p.lower() for p in read_caption(captions[idx])
                ]
                if modify_captions and prompt_generator:
                    positive_prompts = [
                        _modify_prompt(prompt_generator, p, generation_size)
                        for p in positive_prompts
                    ]
            elif prompt:
                positive_prompts = [prompt] if isinstance(prompt, str) else prompt
            else:
                positive_prompts = [""]

            neg = negative_prompt or ""
            neg_prompts = [neg]

            output_images = generator.gen(
                condition, positive_prompts, neg_prompts,
                quality=quality, guidance_scale=guidance_scale,
            )

            for j, img in enumerate(output_images.images if hasattr(output_images, 'images') else output_images):
                img.save(os.path.join(output, f"{image_name}_{j + 1}.png"))
                generated_count += 1

        except Exception as e:
            logger.warning(f"Image {image_path}: Exception during generation: {e}")
            errors.append((image_path, str(e)))

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(real_images)} images ({idx / len(real_images) * 100:.0f}%)")

    logger.info(f"Generation complete: {generated_count} images generated")

    return {
        "total_generated": generated_count,
        "output_path": str(output),
        "source_images": len(real_images),
        "errors": errors,
    }


def _modify_prompt(generator: NaivePromptGenerator, phrase: str, size: int) -> str:
    modified = generator.prompts(size, phrase)
    cleaned = [p for p in modified if p != phrase]
    if not cleaned:
        logger.info(f'No new prompt for "{phrase}", using original.')
        return phrase
    return random.choice(cleaned)
