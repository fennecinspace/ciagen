# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or any later version. This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details. You should have received a copy of the Lesser GNU
# General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

import glob
import os
import random
from pathlib import Path
from typing import Dict

import torch
from diffusers.utils import load_image
from omegaconf import DictConfig, OmegaConf

from ciagen.extractors import instantiate_extractor
from ciagen.generators import SDCN, NaivePromptGenerator
from ciagen.utils.common import get_model_config, logger, read_caption

# Do not let torch decide on best algorithm (we know better!)
torch.backends.cudnn.benchmark = False


class Generator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, paths: Dict[str, str | Path]) -> None:
        data = self.cfg["data"]

        # Paths and data related work
        _real_path = paths["real"]
        generated_path = paths["generated"]
        real_path_images = paths["real_images"]
        real_path_captions = paths["real_captions"]

        formats = data["image_formats"]
        real_images = []
        for format in formats:
            real_images += [
                *glob.glob(str(Path(real_path_images).absolute()) + f"/*.{format}")
            ]
        real_images.sort()
        real_path_images = real_images

        real_dataset_size = len(real_path_images)

        # Define the prompt
        prompt = self.cfg["prompt"]
        modify_captions = bool(prompt["modify_captions"])
        prompt_generation_size = prompt["generation_size"]

        promp_generator = (
            NaivePromptGenerator(prompt["template"]) if modify_captions else None
        )

        if isinstance(prompt["base"], str):
            positive_prompt = [
                prompt["base"] + " " + prompt["modifier"] + " " + prompt["quality"]
            ]
            negative_prompt = ["".join(prompt["negative_simple"])]
        else:
            positive_prompt = [
                pb + " " + prompt["modifier"] + " " + prompt["quality"]
                for pb in prompt["base"]
            ]
            negative_prompt = ["".join(prompt["negative_simple"])] * len(
                positive_prompt
            )

        # Specify the model and feature extractor. Be aware that ideally both extractor and
        # generator should be using the same feature.

        ## !!! MUST ADD CHECKS FOR MODEL NAMES AND CONFIG CORRECTNESS HERE !!!

        model_data = self.cfg["model"]
        model_to_use = model_data["cn_use"]
        model_to_use_conf = [model_to_use]

        model_to_use_conf = get_model_config(model_to_use, model_data["cn"])

        logger.info(f"Using the following config : {model_to_use_conf}")

        sd_model = model_to_use_conf["sd"]
        cn_model = model_to_use_conf["cn"]
        extractor_name = model_to_use_conf["extractor"]

        if "cn_extra_settings" in model_to_use_conf:
            cn_extra_settings = model_to_use_conf["cn_extra_settings"]
        else:
            cn_extra_settings = {}

        use_captions = bool(model_data["use_captions"])
        # use_labels = bool(model_data['use_labels'])

        if use_captions:

            if not os.path.exists(real_path_captions):
                raise ValueError(
                    f"Cannot find captions in {real_path_captions}. Path does not exist."
                )

            real_path_captions = glob.glob(
                str(Path(real_path_captions).absolute()) + f"/*"
            )
            real_path_captions.sort()
            if len(real_path_captions) != real_dataset_size:
                raise ValueError(
                    "Cannot use a captions dataset of different size! Please verify one or both directories."
                )
            logger.info("Using captions")
        else:
            real_path_captions = []

        # if use_labels:
        #     real_labels_path = glob.glob(str(real_labels_path.absolute()) + f'/*')
        #     real_labels_path.sort()
        #     if len(real_labels_path) != real_dataset_size:
        #         raise Exception("Cannot use a labels dataset of different size!")
        #     logger.info("Using labels")
        # else:
        #     real_labels_path = []

        # cn_model = find_model_name(model_data["cn_use"], model_data["cn"])
        # cn_model = (
        #     cn_model
        #     if cn_model is not None
        #     else "fusing/stable-diffusion-v1-5-controlnet-openpose"
        # )

        seed = model_data["seed"]
        device = model_data["device"]

        generator = SDCN(
            sd_model, cn_model, seed, device=device, cn_extra_settings=cn_extra_settings
        )
        extractor = instantiate_extractor(extractor_name)
        logger.info(f"Using extractor: {extractor}")
        logger.info(f"Using generator: {generator}")

        logger.info(f"Results will be saved to {generated_path}")
        logger.info(f"Real dataset size: {real_dataset_size}")

        # dumps to file:
        metadata_dict = {"gen_config": self.cfg}
        with open(os.path.join(generated_path, "metadata.yaml"), "w") as f:
            OmegaConf.save(metadata_dict, f)

        for idx in range(real_dataset_size):
            image_path = real_path_images[idx]

            caption_path = real_path_captions[idx] if use_captions else None
            # label_path = real_labels_path[idx] if use_labels else None

            try:
                image = load_image(image_path)
                image_name = image_path.split(f"{os.sep}")[-1].split(".")[0]

                # Feature extraction, save also the features.
                feature = extractor.extract(image)

                # Here we use captions if necessary and modify them if necessary.
                if use_captions:
                    positive_prompt = [p.lower() for p in read_caption(caption_path)]
                    negative_prompt = ["".join(prompt["negative_simple"])] * len(
                        positive_prompt
                    )

                    if modify_captions:

                        def modify_prompt(p):
                            modified_list = promp_generator.prompts(
                                prompt_generation_size, p
                            )
                            cleaned_list = list(
                                filter(lambda new_p: new_p != p, modified_list)
                            )
                            if not cleaned_list:
                                logger.info(
                                    f'No new prompt created for caption "{p}", using the same.'
                                )
                                return p
                            return random.choice(cleaned_list)

                        positive_prompt = [modify_prompt(p) for p in positive_prompt]

                # Generate with stable diffusion
                # Clean a little the gpu memory between generations
                output = generator.gen(feature, positive_prompt, negative_prompt)
                logger.info(f"Generated {len(output.images)} images from real sample.")

                # save images
                for j, img in enumerate(output.images):
                    img.save(os.path.join(generated_path, f"{image_name}_{j + 1}.png"))

            except Exception as e:
                logger.warning(
                    f"Image {image_path}: Exception during Extraction/SDCN", e
                )

            if (idx + 1) % 50 == 0:
                logger.info(
                    f"Treated {idx + 1} images ({(idx)/real_dataset_size * 100}%)."
                )
